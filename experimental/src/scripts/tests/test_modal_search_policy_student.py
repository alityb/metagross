from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.modal_search_policy_student import (  # noqa: E402
    DATASET_GZIP_SHA256,
    DATASET_SHA256,
    R1_SHA256,
    _sha256,
)


class ModalSearchPolicyStudentTests(unittest.TestCase):
    def test_frozen_hashes(self):
        self.assertEqual(len(R1_SHA256), 64)
        self.assertEqual(len(DATASET_SHA256), 64)
        self.assertEqual(len(DATASET_GZIP_SHA256), 64)

    def test_sha256(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact"
            path.write_bytes(b"metagross")
            self.assertEqual(_sha256(path), hashlib.sha256(b"metagross").hexdigest())


if __name__ == "__main__":
    unittest.main()
