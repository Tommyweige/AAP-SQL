"""Train the retriever/reranker with thesis 3.3.2-3.3.3 settings."""
import argparse
import hashlib
import json
from pathlib import Path
import random

SETTINGS = {
    "retriever": {"model": "sentence-transformers/all-mpnet-base-v2", "batch_size": 64, "warmup": 200,
                  "output": "column_retriever_v3_paper_earlystop"},
    "reranker": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "batch_size": 32, "warmup": 100,
                 "output": "cross_encoder_schema_paper_repro"},
}
EPOCHS, PATIENCE, LEARNING_RATE = 30, 4, 2e-5


def train(kind, pairs, output, seed):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Model output is not empty: {output}; choose a fresh --output-root")
    # Heavy packages and model downloads are confined to an explicit training run.
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
    try:
        from sentence_transformers.sentence_transformer import losses
    except ImportError:  # sentence-transformers 5.x uses the original module path.
        from sentence_transformers import losses
    from transformers import get_linear_schedule_with_warmup
    from scripts.research.training_data import question_batches

    torch.manual_seed(seed)
    random.seed(seed)
    settings = SETTINGS[kind]
    if kind == "retriever":
        model = SentenceTransformer(settings["model"])
        network = model
        criterion = losses.MultipleNegativesRankingLoss(model)
    else:
        model = CrossEncoder(settings["model"], num_labels=1)
        network = model.model
        criterion = torch.nn.BCEWithLogitsLoss()
    device = next(network.parameters()).device
    optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE)

    def batches(split, epoch):
        rows = pairs[split]
        if kind == "retriever":
            return list(question_batches(rows, settings["batch_size"], seed + epoch))
        rows = rows[:]
        random.Random(seed + epoch).shuffle(rows)
        return [rows[i:i + settings["batch_size"]] for i in range(0, len(rows), settings["batch_size"])]

    for split in ("train", "validation", "test"):
        if not pairs[split]:
            raise ValueError(f"No {kind} pairs in {split} split")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, settings["warmup"], len(batches("train", 0)) * EPOCHS)

    def batch_loss(rows):
        questions, descriptions = [r[1] for r in rows], [r[2] for r in rows]
        if kind == "retriever":
            features, labels = model.smart_batching_collate([InputExample(texts=[q, d]) for q, d in zip(questions, descriptions)])
            features = [{k: v.to(device) if hasattr(v, "to") else v for k, v in f.items()} for f in features]
            return criterion(features, labels.to(device))
        encoded = model.tokenizer(questions, descriptions, padding=True, truncation=True,
                                  max_length=512, return_tensors="pt")
        logits = network(**{k: v.to(device) for k, v in encoded.items()}).logits.view(-1)
        labels = torch.tensor([r[3] for r in rows], dtype=torch.float, device=device)
        return criterion(logits, labels)

    def evaluate(split):
        network.eval()
        total, seen = 0., 0
        with torch.no_grad():
            for rows in batches(split, 0):
                total += batch_loss(rows).item() * len(rows)
                seen += len(rows)
        return total / seen

    history, best, stale = [], float("inf"), 0
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(EPOCHS):
        network.train()
        total, seen = 0., 0
        for rows in batches("train", epoch):
            optimizer.zero_grad()
            loss = batch_loss(rows)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(rows)
            seen += len(rows)
        validation = evaluate("validation")
        history.append({"epoch": epoch + 1, "train_loss": total / seen, "validation_loss": validation})
        if validation < best:
            best, stale = validation, 0
            model.save(str(output))
        else:
            stale += 1
        print(kind, history[-1], flush=True)
        if stale >= PATIENCE:
            break
    # Restore the selected checkpoint for held-out evaluation.
    if kind == "retriever":
        model = SentenceTransformer(str(output))
        network = model
        criterion = losses.MultipleNegativesRankingLoss(model)
    else:
        model = CrossEncoder(str(output), num_labels=1)
        network = model.model
    device = next(network.parameters()).device
    report = {"settings": settings, "learning_rate": LEARNING_RATE, "max_epochs": EPOCHS,
              "patience": PATIENCE, "seed": seed, "history": history, "test_loss": evaluate("test")}
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("models"))
    parser.add_argument("--mode", choices=["retriever", "reranker", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true", help="Write split IDs, pair counts and provenance without downloading/training models")
    args = parser.parse_args()
    from scripts.research.training_data import question_splits, build_pairs
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
    splits = question_splits(questions, args.seed)
    pairs = {split: build_pairs(questions, descriptions, ids, args.seed) for split, ids in splits.items()}
    manifest = {"question_ids": splits, "seed": args.seed,
                "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                           (("questions", args.questions), ("descriptions", args.descriptions))},
                "pair_counts": {split: {"retriever": len(p[0]), "reranker": len(p[1])} for split, p in pairs.items()}}
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "training_data_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise FileExistsError(f"Training inputs differ from {manifest_path}; choose a fresh --output-root")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.prepare_only:
        for kind, idx in (("retriever", 0), ("reranker", 1)):
            if args.mode in (kind, "all"):
                train(kind, {split: p[idx] for split, p in pairs.items()}, args.output_root / SETTINGS[kind]["output"], args.seed)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
