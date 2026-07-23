from __future__ import annotations

import json
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build_mcts_v3_dataset import (  # noqa: E402
    GroupRejected,
    build_group,
    load_dumps,
    map_move_string,
    normalize_tag,
)


def make_dump(tag="gen9randombattle-1", username="learner", decision_idx=0,
              illegal=None, name_table=None, **overrides):
    if illegal is None:
        illegal = [False, False, True, True, False, True, True, True, True,
                   True, True, True, True]
    if name_table is None:
        name_table = {
            "voltswitch": 0,
            "thunderbolt": 1,
            "voltswitch-tera": 9,
            "thunderbolt-tera": 10,
            "switch glimmora": 4,
        }
    row = {
        "schema": 3,
        "tag": tag,
        "decision_idx": decision_idx,
        "battle_turn": decision_idx + 1,
        "username": username,
        "text_tokens": [1, 2, 3],
        "numbers": [0.5, 0.25],
        "illegal_actions": illegal,
        "name_table": name_table,
        "probs": [1.0 / 13] * 13,
    }
    row.update(overrides)
    return row


def make_decision(tag="gen9randombattle-1", username="learner", decision_idx=0,
                  visits=None, selected="voltswitch", **overrides):
    if visits is None:
        visits = {"voltswitch": 0.75, "thunderbolt": 0.25}
    row = {
        "record_type": "decision",
        "battle_tag": tag,
        "username": username,
        "turn": decision_idx + 1,
        "mcts_schema_version": 3,
        "prior_decision_idx": decision_idx,
        "mcts_visits": visits,
        "selected_action": selected,
    }
    row.update(overrides)
    return row


def dumps_by_key(rows):
    return {
        (normalize_tag(r["tag"]), r["username"], r["decision_idx"]): r
        for r in rows
    }


GROUP_KEY = ("gen9randombattle-1", "learner")


class TestNormalizeTag(unittest.TestCase):
    def test_strips_battle_prefix(self):
        self.assertEqual(normalize_tag("battle-gen9randombattle-42"),
                         "gen9randombattle-42")

    def test_leaves_bare_tag(self):
        self.assertEqual(normalize_tag("gen9randombattle-42"),
                         "gen9randombattle-42")


class TestMapMoveString(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(map_move_string("voltswitch", {"voltswitch": 0}),
                         (0, "exact"))

    def test_norm_match(self):
        self.assertEqual(map_move_string("Volt Switch", {"voltswitch": 0}),
                         (0, "norm"))

    def test_unmapped(self):
        idx, how = map_move_string("surf", {"voltswitch": 0})
        self.assertIsNone(idx)
        self.assertEqual(how, "unmapped")

    def test_tera_suffix_distinct(self):
        table = {"voltswitch": 0, "voltswitch-tera": 9}
        self.assertEqual(map_move_string("voltswitch-tera", table)[0], 9)
        self.assertEqual(map_move_string("voltswitch", table)[0], 0)

    def test_ambiguous_norm_raises(self):
        with self.assertRaises(GroupRejected):
            map_move_string("volt switch", {"volt-switch": 0, "voltswitch!": 1})

    def test_form_match_regional_bench(self):
        # server tracks benched Weezing-Galar as base 'weezing'
        table = {"switch weezing": 8, "switch deoxys": 4}
        self.assertEqual(map_move_string("switch weezinggalar", table),
                         (8, "form"))

    def test_form_match_reverse_direction(self):
        table = {"switch ogerponhearthflame": 4}
        self.assertEqual(map_move_string("switch ogerpon", table), (4, "form"))

    def test_form_no_cross_species_match(self):
        # mew is a prefix of mewtwo but exact match wins before form tier
        table = {"switch mew": 4, "switch mewtwo": 5}
        self.assertEqual(map_move_string("switch mewtwo", table), (5, "exact"))
        self.assertEqual(map_move_string("switch mew", table), (4, "exact"))

    def test_form_ambiguous_rejects(self):
        table = {"switch mew": 4, "switch mewtwo": 5}
        with self.assertRaises(GroupRejected):
            map_move_string("switch mewt", table)

    def test_form_tier_not_applied_to_moves(self):
        idx, how = map_move_string("voltswit", {"voltswitch": 0})
        self.assertIsNone(idx)
        self.assertEqual(how, "unmapped")


class TestBuildGroup(unittest.TestCase):
    def test_happy_path(self):
        dumps = dumps_by_key([make_dump(decision_idx=0), make_dump(decision_idx=1)])
        rows = [make_decision(decision_idx=0), make_decision(decision_idx=1)]
        out = build_group(GROUP_KEY, rows, dumps, label=1, match_stats=Counter())
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0]["visit_target_13"][0], 0.75)
        self.assertAlmostEqual(out[0]["visit_target_13"][1], 0.25)
        self.assertEqual(out[0]["selected_action_index"], 0)
        self.assertEqual(out[0]["label"], 1)
        self.assertEqual([r["decision_idx"] for r in out], [0, 1])

    def test_switch_and_tera_mapping(self):
        visits = {"switch glimmora": 0.5, "voltswitch-tera": 0.5}
        dumps = dumps_by_key([make_dump(illegal=[False] * 13)])
        rows = [make_decision(visits=visits, selected="switch glimmora")]
        out = build_group(GROUP_KEY, rows, dumps, None, Counter())
        self.assertAlmostEqual(out[0]["visit_target_13"][4], 0.5)
        self.assertAlmostEqual(out[0]["visit_target_13"][9], 0.5)
        self.assertEqual(out[0]["selected_action_index"], 4)

    def assert_rejected(self, rows, dumps, reason):
        with self.assertRaises(GroupRejected) as ctx:
            build_group(GROUP_KEY, rows, dumps, None, Counter())
        self.assertEqual(ctx.exception.reason, reason)

    def test_missing_dump_row_rejects(self):
        dumps = dumps_by_key([make_dump(decision_idx=0)])
        rows = [make_decision(decision_idx=0), make_decision(decision_idx=1)]
        self.assert_rejected(rows, dumps, "missing_dump_row")

    def test_discontinuous_idxs_reject(self):
        dumps = dumps_by_key([make_dump(decision_idx=0), make_dump(decision_idx=2)])
        rows = [make_decision(decision_idx=0), make_decision(decision_idx=2)]
        self.assert_rejected(rows, dumps, "discontinuous_decision_idxs")

    def test_duplicate_idxs_reject(self):
        dumps = dumps_by_key([make_dump(decision_idx=0)])
        rows = [make_decision(decision_idx=0), make_decision(decision_idx=0)]
        self.assert_rejected(rows, dumps, "duplicate_decision_idx")

    def test_visit_mass_on_illegal_action_rejects(self):
        illegal = [False, True, True, True, False, True, True, True, True,
                   True, True, True, True]
        dumps = dumps_by_key([make_dump(illegal=illegal)])
        self.assert_rejected([make_decision()], dumps, "visit_mass_on_illegal_action")

    def test_unmappable_visit_string_rejects(self):
        dumps = dumps_by_key([make_dump()])
        rows = [make_decision(visits={"surf": 1.0}, selected="surf")]
        self.assert_rejected(rows, dumps, "unmappable_visit_string")

    def test_forced_recharge_is_skipped_not_remapped(self):
        dumps = dumps_by_key([make_dump()])
        rows = [make_decision(visits={"recharge": 0.9, "recharge-tera": 0.1}, selected="recharge")]
        stats = Counter()
        self.assertEqual(build_group(GROUP_KEY, rows, dumps, None, stats), [])
        self.assertEqual(stats["skipped_forced_action_decisions"], 1)

    def test_mixed_forced_and_real_visit_drops_forced_mass(self):
        dumps = dumps_by_key([make_dump()])
        rows = [make_decision(visits={"struggle": 0.5, "voltswitch": 0.5}, selected="voltswitch")]
        stats = Counter()
        out = build_group(GROUP_KEY, rows, dumps, None, stats)
        self.assertEqual(out[0]["visit_target_13"][0], 1.0)
        self.assertEqual(stats["ignored_forced_visit_mass"], 0.5)

    def test_bad_visit_mass_rejects(self):
        dumps = dumps_by_key([make_dump()])
        rows = [make_decision(visits={"voltswitch": 0.5})]
        self.assert_rejected(rows, dumps, "bad_visit_mass")

    def test_wrong_schema_version_rejects(self):
        dumps = dumps_by_key([make_dump()])
        rows = [make_decision(mcts_schema_version=2)]
        self.assert_rejected(rows, dumps, "wrong_schema_version")

    def test_missing_join_key_rejects(self):
        dumps = dumps_by_key([make_dump()])
        row = make_decision()
        del row["prior_decision_idx"]
        self.assert_rejected([row], dumps, "missing_prior_decision_idx")

    def test_selected_zero_target_rejects(self):
        dumps = dumps_by_key([make_dump()])
        rows = [make_decision(visits={"voltswitch": 1.0}, selected="thunderbolt")]
        self.assert_rejected(rows, dumps, "selected_action_zero_target")

    def test_fp_target_agreement_stat(self):
        stats = Counter()
        fp_target = [0.0] * 13
        fp_target[0] = 1.0
        rows = [make_decision(mcts_visit_target_13=fp_target)]
        dumps = dumps_by_key([make_dump()])
        build_group(GROUP_KEY, rows, dumps, None, stats)
        self.assertEqual(stats["fp_target_present"], 1)
        self.assertEqual(stats["fp_target_top1_agree"], 1)


class TestLoadDumps(unittest.TestCase):
    def test_duplicate_key_dropped(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dump.jsonl"
            rows = [make_dump(decision_idx=0), make_dump(decision_idx=0)]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            dumps, stats = load_dumps([path])
            self.assertEqual(dumps, {})
            self.assertEqual(stats["dump_duplicate_key"], 1)

    def test_wrong_schema_skipped(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dump.jsonl"
            path.write_text(json.dumps(make_dump(schema=2)) + "\n")
            dumps, stats = load_dumps([path])
            self.assertEqual(dumps, {})
            self.assertEqual(stats["dump_wrong_schema"], 1)

    def test_namespace_filter_excludes_other_workers(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dump.jsonl"
            rows = [
                make_dump(namespace="w1"),
                make_dump(namespace="w2"),
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            dumps, stats = load_dumps([path], namespace="w1")
            self.assertEqual(len(dumps), 1)
            self.assertEqual(stats["dump_other_namespace"], 1)


class TestCli(unittest.TestCase):
    def run_cli(self, tmp, decision_rows, dump_rows, extra_args=()):
        dump_path = Path(tmp) / "dump.jsonl"
        decision_path = Path(tmp) / "decisions.jsonl"
        out_path = Path(tmp) / "out.jsonl"
        report_path = Path(tmp) / "report.json"
        dump_path.write_text("\n".join(json.dumps(r) for r in dump_rows) + "\n")
        decision_path.write_text(
            "\n".join(json.dumps(r) for r in decision_rows) + "\n")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_mcts_v3_dataset.py"),
             "--decision-log", str(decision_path),
             "--prior-dump", str(dump_path),
             "--output", str(out_path),
             "--report", str(report_path),
             *extra_args],
            capture_output=True, text=True,
        )
        return proc, out_path, report_path

    def test_end_to_end(self):
        with TemporaryDirectory() as tmp:
            dump_rows = [
                make_dump(tag="battle-gen9randombattle-7", decision_idx=i)
                for i in range(2)
            ]
            decision_rows = [
                make_decision(tag="gen9randombattle-7", decision_idx=0),
                make_decision(tag="gen9randombattle-7", decision_idx=1),
                {"record_type": "battle_result",
                 "battle_tag": "gen9randombattle-7", "username": "learner",
                 "winner": "learner", "label": 1},
            ]
            proc, out_path, report_path = self.run_cli(
                tmp, decision_rows, dump_rows,
                ("--require-labels", "--min-admission-rate", "1.0"))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(report_path.read_text())
            self.assertTrue(report["ok"])
            self.assertEqual(report["groups_admitted"], 1)
            self.assertEqual(report["targets_written"], 2)
            records = [json.loads(l) for l in out_path.read_text().splitlines()]
            self.assertTrue(all(r["label"] == 1 for r in records))

    def test_min_admission_rate_fails_closed(self):
        with TemporaryDirectory() as tmp:
            dump_rows = [make_dump(decision_idx=0)]
            decision_rows = [make_decision(decision_idx=0),
                             make_decision(decision_idx=1)]
            proc, _, report_path = self.run_cli(
                tmp, decision_rows, dump_rows,
                ("--min-admission-rate", "1.0"))
            self.assertNotEqual(proc.returncode, 0)
            report = json.loads(report_path.read_text())
            self.assertFalse(report["ok"])
            self.assertEqual(report["rejection_reasons"],
                             {"missing_dump_row": 1})


if __name__ == "__main__":
    unittest.main()
