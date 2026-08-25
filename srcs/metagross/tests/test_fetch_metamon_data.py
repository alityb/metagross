import json
import unittest
from pathlib import Path

from srcs.metagross.data_source_registry import load_registry
from srcs.metagross.fetch_metamon_data import describe_source


REGISTRY = Path("experimental/configs/data_sources_v1.json")


class FetchMetamonDataTest(unittest.TestCase):
    def test_approved_human_source_is_downloadable(self):
        source = describe_source(load_registry(REGISTRY), "metamon_human_v6")
        self.assertEqual(source["selected_artifacts"][0]["path"], "gen9ou.tar.gz")

    def test_unlicensed_selfplay_is_not_downloadable(self):
        with self.assertRaisesRegex(ValueError, "not approved for training"):
            describe_source(load_registry(REGISTRY), "metamon_pac_base")


if __name__ == "__main__":
    unittest.main()
