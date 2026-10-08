"""
Description Loader - Load semantic column descriptions from BIRD metadata.
"""
import json
from pathlib import Path
from typing import Dict, Optional

class DescriptionLoader:
    """Load and cache descriptions from BIRD format tables.json."""

    def __init__(self, tables_json_path: Path):
        """
        Initialize loader with path to tables.json.

        Args:
            tables_json_path: Path to BIRD tables.json file
        """
        self.tables_json_path = tables_json_path
        self.descriptions_cache: Dict[str, Dict[str, str]] = {}
        self._loaded = False

    def _load_if_needed(self):
        """Load data from JSON file if not already loaded."""
        if self._loaded:
            return

        try:
            if not self.tables_json_path.exists():
                print(f"Warning: Description file not found at {self.tables_json_path}")
                self._loaded = True
                return

            with open(self.tables_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for db_entry in data:
                db_id = db_entry['db_id']
                table_names = db_entry['table_names_original']
                orig_cols = db_entry['column_names_original']
                human_cols = db_entry['column_names']

                db_descriptions = {}

                # Iterate through columns to map original name -> human name
                # Both lists should be aligned by index
                for i in range(len(orig_cols)):
                    if i >= len(human_cols):
                        break

                    table_idx, orig_name = orig_cols[i]
                    _, human_name = human_cols[i]

                    # Skip special columns like '*' with table_idx -1
                    if table_idx < 0:
                        continue

                    # Skip if names are identical (no semantic value added)
                    if orig_name.lower() == human_name.lower():
                        continue

                    # Resolve table name
                    if table_idx < len(table_names):
                        table_name = table_names[table_idx]
                        key = f"{table_name}.{orig_name}".lower() # Case-insensitive key
                        db_descriptions[key] = human_name

                self.descriptions_cache[db_id] = db_descriptions

            self._loaded = True

        except Exception as e:
            print(f"Error loading descriptions: {e}")
            self._loaded = True # Prevent retry loop on error

    def get_descriptions(self, db_id: str) -> Dict[str, str]:
        """
        Get descriptions for a specific database.

        Args:
            db_id: Database identifier

        Returns:
            Dictionary mapping "table.column" (lowercase) to description
        """
        self._load_if_needed()
        return self.descriptions_cache.get(db_id, {})

# Global instance
_loader_instance: Optional[DescriptionLoader] = None

def get_description_loader(tables_json_path: Optional[Path] = None) -> DescriptionLoader:
    """Get or create global DescriptionLoader instance."""
    global _loader_instance
    if tables_json_path is None:
        raise ValueError("A metadata path must be supplied explicitly")
    tables_json_path = Path(tables_json_path)
    if _loader_instance is None or _loader_instance.tables_json_path != tables_json_path:
        _loader_instance = DescriptionLoader(tables_json_path)
    return _loader_instance
