"""Exercise real model SDKs/Torch with small randomly initialized local weights."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("sentence_transformers")
try:
    from sentence_transformers.sentence_transformer import losses
except ImportError:
    from sentence_transformers import losses
from scripts.research import train_models


def tiny_model(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "positive", "negative"]
    words += [f"{prefix}{i}" for prefix in ("q", "d") for i in range(8)]
    vocab = {word: i for i, word in enumerate(words)}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                   pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", model_max_length=32)
    fast.save_pretrained(path)
    config = BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=16, max_position_embeddings=32, num_labels=1)
    BertForSequenceClassification(config).save_pretrained(path)


@pytest.mark.parametrize("kind", ["retriever", "reranker"])
def test_training_stops_and_tests_the_best_checkpoint(kind, tmp_path, monkeypatch):
    # Real local Transformer weights exercise tokenization, backpropagation, SDK
    # save/load and loss APIs. Flat held-out loss makes early stopping predictable.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    base, output = tmp_path / "base", tmp_path / "trained"
    tiny_model(base)
    instances, snapshots = [], []
    native = st.SentenceTransformer if kind == "retriever" else st.CrossEncoder
    def construct(path, **kwargs):
        model = native(path, **kwargs)
        instances.append(model)
        original_save = model.save
        def save(directory):
            network = model if kind == "retriever" else model.model
            snapshots.append({k: v.detach().clone() for k, v in network.state_dict().items()})
            original_save(directory)
        model.save = save
        return model
    original_ranking = losses.MultipleNegativesRankingLoss
    original_bce = torch.nn.BCEWithLogitsLoss
    class HeldOutLoss(torch.nn.Module):
        def __init__(self, actual):
            super().__init__()
            self.actual = actual
        def forward(self, *args):
            return self.actual(*args) if torch.is_grad_enabled() else torch.tensor(1.)
    monkeypatch.setattr(st, "SentenceTransformer", construct)
    monkeypatch.setattr(st, "CrossEncoder", construct)
    monkeypatch.setattr(losses, "MultipleNegativesRankingLoss", lambda model: HeldOutLoss(original_ranking(model)))
    monkeypatch.setattr(torch.nn, "BCEWithLogitsLoss", lambda: HeldOutLoss(original_bce()))
    monkeypatch.setattr(train_models, "EPOCHS", 8)
    monkeypatch.setattr(train_models, "PATIENCE", 2)
    monkeypatch.setattr(train_models, "LEARNING_RATE", .1)
    settings = {k: {**v, "warmup": 0, "model": str(base)} for k, v in train_models.SETTINGS.items()}
    monkeypatch.setattr(train_models, "SETTINGS", settings)
    if kind == "retriever":
        pairs = [(i, f"q{i}", f"d{i}") for i in range(8)]
    else:
        pairs = [(i, f"q{i}", d, label) for i in range(8) for d, label in (("positive", 1.), ("negative", 0.))]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train_models.train(kind, {split: pairs for split in ("train", "validation", "test")}, output, 42)
    finally:
        torch.set_num_threads(previous_threads)
    report = json.loads((output / "training_report.json").read_text())
    assert len(report["history"]) == 3 and report["test_loss"] == 1.
    assert len(snapshots) == 1 and len(instances) == 2
    restored = instances[1] if kind == "retriever" else instances[1].model
    assert all(torch.equal(restored.state_dict()[k], v) for k, v in snapshots[0].items())
    last = instances[0] if kind == "retriever" else instances[0].model
    assert any(not torch.equal(last.state_dict()[k], v) for k, v in snapshots[0].items())
