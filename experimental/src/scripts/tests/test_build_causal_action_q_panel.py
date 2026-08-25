from copy import deepcopy

import json

from scripts.build_causal_action_q_panel import (
    battle_is_in_purpose,
    canonical_decision,
    canonical_observers,
    schema6_history_valid,
    schema6_public_facts,
    scoped_source_groups,
    split_for_purpose,
    snapshot_metrics,
)
from train.shallow_search_residual import battle_split


def snapshot():
    text = [[1, 2], [3, 4], [5, 6]]
    numbers = [[0.0], [1.0], [2.0]]
    illegal = [[False] * 13, [False] * 13, [True] + [False] * 12]
    return {
        "schema": 6,
        "decision_idx": 2,
        "mask_fallback": False,
        "player_information_state": {
            "schema_version": 1,
            "opponent_public_team": [{
                "pokemon": {
                    "name": "Pikachu",
                    "moves": [{"name": "Thunderbolt"}],
                    "item": "unknownitem",
                    "ability": "Static",
                }
            }],
        },
        "player_observation_history": {},
        "text_tokens": text[-1],
        "numbers": numbers[-1],
        "illegal_actions": illegal[-1],
        "trajectory": {
            "mode": "causal-history",
            "observations": 3,
            "transitions": 2,
            "inference_length": 3,
            "action_receipts": [{}, {}],
            "rl2": [[0.0] * 14 for _ in range(3)],
            "time_indices": [0, 1, 2],
            "observation_rows": {
                "text_tokens": text,
                "numbers": numbers,
                "illegal_actions": illegal,
            },
        },
    }


def test_schema6_history_accepts_exact_causal_trajectory():
    assert schema6_history_valid(snapshot())


def test_schema6_history_rejects_noncontiguous_time():
    row = snapshot()
    row["trajectory"]["time_indices"] = [0, 2, 3]
    assert not schema6_history_valid(row)


def test_schema6_history_rejects_negative_decision_index():
    row = snapshot()
    row["decision_idx"] = -1
    assert not schema6_history_valid(row)


def test_schema6_history_rejects_top_level_observation_mismatch():
    row = snapshot()
    row["numbers"] = [99.0]
    assert not schema6_history_valid(row)


def test_schema6_history_rejects_mask_fallback():
    row = deepcopy(snapshot())
    row["mask_fallback"] = True
    assert not schema6_history_valid(row)


def test_schema6_history_accepts_production_context_crop():
    row = snapshot()
    row["decision_idx"] = 130
    for name, value in (
        ("text_tokens", [1, 2]),
        ("numbers", [0.0]),
        ("illegal_actions", [False] * 13),
    ):
        row[name] = value
        row["trajectory"]["observation_rows"][name] = [value] * 128
    row["trajectory"].update({
        "observations": 128,
        "transitions": 127,
        "inference_length": 128,
        "action_receipts": [{}] * 127,
        "rl2": [[0.0] * 14 for _ in range(128)],
        "time_indices": list(range(3, 131)),
    })
    assert schema6_history_valid(row)


def test_schema6_public_facts_compile_only_explicit_public_fields():
    facts = schema6_public_facts(snapshot())
    assert facts.species == frozenset({"pikachu"})
    assert facts.moves == (("pikachu", ("thunderbolt",)),)
    assert facts.items == frozenset()
    assert facts.abilities == frozenset({"pikachu"})


def test_causal_dual_root_is_a_canonical_side_one_decision():
    row = canonical_decision({
        "schema": "metagross-causal-dual-r1-root/v1",
        "identity": {
            "battle_tag": "battle-one",
            "username": "alice",
            "decision_idx": 2,
            "battle_turn": 3,
        },
        "state": "engine-state",
    })
    assert row == {
        "battle_tag": "battle-one",
        "username": "alice",
        "prior_decision_idx": 2,
        "turn": 3,
        "state": "engine-state",
    }


def test_snapshot_metrics_accepts_multiple_isolated_dumps(tmp_path):
    paths = []
    for tag, username in (("battle-one", "alice"), ("battle-two", "bob")):
        row = snapshot()
        row.update({
            "tag": tag,
            "username": username,
            "probs": [1.0 / 13.0] * 13,
        })
        path = tmp_path / f"{username}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        paths.append(path)
    metrics = snapshot_metrics(paths)
    assert set(metrics) == {("one", "alice", 2), ("two", "bob", 2)}


def test_snapshot_metrics_discards_withheld_rows_before_feature_processing(tmp_path):
    admitted = snapshot()
    admitted.update({
        "tag": "battle-admitted",
        "username": "alice",
        "probs": [1.0 / 13.0] * 13,
    })
    withheld = {
        "tag": "battle-withheld",
        "username": "bob",
        # These deliberately malformed feature fields would fail if the row
        # reached policy/history processing.
        "decision_idx": "not-an-index",
        "probs": "not-probabilities",
        "illegal_actions": None,
    }
    path = tmp_path / "snapshots.jsonl"
    path.write_text(json.dumps(admitted) + "\n" + json.dumps(withheld) + "\n")
    metrics = snapshot_metrics(
        [path],
        {str(tmp_path.resolve()): {("admitted", "alice")}},
    )
    assert set(metrics) == {("admitted", "alice", 2)}


def test_canonical_observer_keeps_one_pov_per_physical_battle():
    groups = {"one": {("one", "alice"), ("one", "bob")}}
    selected = canonical_observers(groups, 7)
    assert selected["one"] in groups["one"]
    assert len(selected) == 1


def test_independent_collection_scopes_do_not_merge_reused_battle_tags(tmp_path):
    paths = []
    for collection, username in (("peer", "alice"), ("unguided", "bob")):
        directory = tmp_path / collection
        directory.mkdir()
        path = directory / "roots.jsonl"
        path.write_text(json.dumps({
            "schema": "metagross-causal-dual-r1-root/v1",
            "identity": {
                "battle_tag": "battle-gen9randombattle-1",
                "username": username,
                "decision_idx": 0,
                "battle_turn": 1,
            },
            "state": "state",
        }) + "\n")
        paths.append(path)
    groups, collection_hashes = scoped_source_groups(paths)
    assert len(groups) == 2
    assert len(set(collection_hashes.values())) == 2


def test_panel_purpose_enforces_deterministic_60_20_20_split():
    examples = {}
    index = 0
    while set(examples) != {"train", "calibration", "test"}:
        battle_id = f"battle-{index}"
        examples.setdefault(battle_split(battle_id), battle_id)
        index += 1

    purposes = {
        "training": "train",
        "calibration": "calibration",
        "evaluation": "test",
    }
    for purpose, expected_split in purposes.items():
        assert split_for_purpose(purpose) == expected_split
        for observed_split, battle_id in examples.items():
            assert battle_is_in_purpose(battle_id, purpose) is (
                observed_split == expected_split
            )


def test_unknown_panel_purpose_fails_closed():
    try:
        split_for_purpose("development")
    except ValueError as exc:
        assert "unsupported panel purpose" in str(exc)
    else:
        raise AssertionError("unknown panel purpose was accepted")
