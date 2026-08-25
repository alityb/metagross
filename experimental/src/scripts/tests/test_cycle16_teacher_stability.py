import pytest

from experimental.src.scripts.run_cycle16_teacher_stability import (
    offline_correlation, with_offline_rqid,
)
from srcs.metagross.prior_server import correlated_request_rqid, request_cache_status


def test_offline_rqid_is_monotone_and_namespace_is_not_in_request():
    repaired0, provenance0 = with_offline_rqid({"wait": False}, "cluster", "p1", 0, variant=0)
    repaired1, provenance1 = with_offline_rqid({"wait": False}, "cluster", "p1", 1, variant=0)
    assert repaired1["rqid"] == repaired0["rqid"] + 1
    assert "namespace_sha256" not in repaired0
    assert provenance0["namespace_sha256"] == provenance1["namespace_sha256"]


def test_correlation_variant_changes_only_rqid():
    request = {"active": [{"moves": []}], "side": {"pokemon": []}}
    first, _ = with_offline_rqid(request, "cluster", "p2", 4, variant=0)
    second, _ = with_offline_rqid(request, "cluster", "p2", 4, variant=1)
    assert {key: value for key, value in first.items() if key != "rqid"} == request
    assert {key: value for key, value in second.items() if key != "rqid"} == request
    assert first["rqid"] != second["rqid"]


def test_live_correlation_contract_rejects_mismatch_and_stale():
    assert correlated_request_rqid(True, {"rqid": 7}, 7) == 7
    with pytest.raises(RuntimeError, match="mismatch"):
        correlated_request_rqid(True, {"rqid": 7}, 8)
    with pytest.raises(ValueError, match="stale"):
        request_cache_status(6, 7)


def test_invalid_offline_correlation_fails_closed():
    with pytest.raises(RuntimeError):
        offline_correlation("cluster", "observer", 0, variant=0)
