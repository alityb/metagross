from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from srcs.metagross import adaptive_ensemble_n100_runner as runner
from srcs.metagross.h2h_audit import _sha256 as real_sha256


class AdaptiveEnsembleN100RunnerTests(unittest.TestCase):
    def test_stored_report_does_not_shadow_live_preflight_function(self):
        root = Path(__file__).resolve().parents[3]
        run_dir = root / "experimental/runs/search_native_stage2_20260809"
        prereg = run_dir / "adaptive-independent-ensemble-n100-preregistration-v10.json"
        authorization = run_dir / "adaptive-independent-ensemble-n100-execution-authorization-v4.json"
        stored_path = run_dir / "adaptive-independent-ensemble-n100-runtime-preflight-v5.json"
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        frozen_runner_hash = json.loads(prereg.read_text(encoding="utf-8"))["source_identity"][
            "n100_runner.py"
        ]

        def bound_sha256(path: Path) -> str:
            if Path(path).resolve() == Path(runner.__file__).resolve():
                return frozen_runner_hash
            return real_sha256(Path(path))

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        with (
            patch.object(runner, "preflight", return_value=stored) as live_preflight,
            patch.object(runner, "_sha256", side_effect=bound_sha256),
            patch.object(runner.os, "kill"),
            patch.object(runner.socket, "create_connection", return_value=Connection()),
            patch.object(runner.os, "execve", side_effect=RuntimeError("exec reached")),
        ):
            with self.assertRaisesRegex(RuntimeError, "exec reached"):
                runner.main(
                    [
                        "--preregistration", str(prereg),
                        "--authorization", str(authorization),
                        "--preflight", str(stored_path),
                    ]
                )
        live_preflight.assert_called_once()


if __name__ == "__main__":
    unittest.main()
