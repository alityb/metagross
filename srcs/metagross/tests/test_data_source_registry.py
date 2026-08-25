import copy
import json
import unittest
from pathlib import Path

from srcs.metagross.data_source_registry import validate_registry


REGISTRY = Path("experimental/configs/data_sources_v1.json")


def registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


class DataSourceRegistryTest(unittest.TestCase):
    def test_frozen_registry_is_valid(self):
        validate_registry(registry())

    def test_quarantined_metamon_selfplay_cannot_enter_training(self):
        row = registry()
        row["training_stages"][0]["sources"][0]["id"] = "metamon_pac_base"
        with self.assertRaisesRegex(ValueError, "non-trainable"):
            validate_registry(row)

    def test_weights_must_sum_to_one(self):
        row = registry()
        row["training_stages"][1]["sources"][0]["weight"] = 0.39
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            validate_registry(row)

    def test_remote_source_must_be_pinned(self):
        row = registry()
        source = next(s for s in row["sources"] if s["id"] == "metamon_human_v6")
        source.pop("revision")
        with self.assertRaisesRegex(ValueError, "not revision-pinned"):
            validate_registry(row)

    def test_derived_rows_must_inherit_battle_split(self):
        row = registry()
        row["identity"]["derived_rows_inherit_split"] = False
        with self.assertRaisesRegex(ValueError, "inherit"):
            validate_registry(row)

    def test_selected_remote_artifact_requires_hash(self):
        row = registry()
        source = next(s for s in row["sources"] if s["id"] == "metamon_human_v6")
        source["selected_artifacts"][0]["sha256"] = "unpinned"
        with self.assertRaisesRegex(ValueError, "lacks a SHA-256"):
            validate_registry(row)


if __name__ == "__main__":
    unittest.main()
