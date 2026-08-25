from pathlib import Path
import pytest
from experimental.src.scripts.cycle43_outcome_blind_watcher import reject_outcome_keys,validate_path
def test_rejects_runner_result_protocol_and_nonwhitelist(tmp_path:Path):
    for rel in ("h2h-runner.log","h2h-result.json","h2h-logs/a.protocol.jsonl","prior-a.log"):
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
        with pytest.raises(RuntimeError):validate_path(tmp_path,p)
def test_accepts_only_receipt_registration_paths(tmp_path:Path):
    for rel in ("move-receipts/a.jsonl","ability-receipts/a.jsonl","engine-receipts/a.json","h2h-registrations/a.json"):
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.touch();validate_path(tmp_path,p)
@pytest.mark.parametrize("key",["winner","win","score","outcome","result","terminal","games"])
def test_rejects_outcome_bearing_keys(key):
    with pytest.raises(RuntimeError):reject_outcome_keys({"safe":{"rows":[{key:1}]}})
