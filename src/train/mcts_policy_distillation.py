"""Fail-closed Foul Play MCTS policy targets for Gen9 Metamon finetuning.

The decision log has no per-action replay identifier.  ``build_sidecar``
therefore labels a trajectory only when its complete, ordered decision sequence
matches the learner POV's recorded actions.  It intentionally emits no partial
or heuristic joins.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


NUM_ACTIONS = 13
SIDECAR_SCHEMA_VERSION = 1


class ActionMappingError(ValueError):
    """A Foul Play action cannot be represented legally by the learner state."""


def _id(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _move_name(move: object) -> str:
    return _id(getattr(move, "name", move))


def _pokemon_name(pokemon: object) -> str:
    return _id(getattr(pokemon, "name", pokemon))


def _ordered_moves(state: object) -> list[object]:
    moves = getattr(getattr(state, "player_active_pokemon"), "moves")
    values = list(moves.values()) if isinstance(moves, dict) else list(moves)
    return sorted(values, key=_move_name)


def _ordered_switches(state: object) -> list[object]:
    return sorted(list(getattr(state, "available_switches")), key=_pokemon_name)


def legal_action_indices(state: object) -> set[int]:
    moves = _ordered_moves(state)
    switches = _ordered_switches(state)
    if len(moves) > 4 or len(switches) > 5:
        raise ActionMappingError("state exceeds Metamon's 13-action space")
    legal = set(range(4, 4 + len(switches)))
    if not getattr(state, "forced_switch"):
        legal.update(range(len(moves)))
        if getattr(state, "can_tera"):
            legal.update(range(9, 9 + len(moves)))
    return legal


def foul_play_action_to_index(action: str, state: object) -> int:
    """Map a Foul Play action string to Metamon's default 13-action index.

    Foul Play logs move IDs (``thunderbolt``), tera move IDs
    (``thunderbolt-tera``), and switches (``switch zapdos``).  The mapping uses
    Metamon's canonical alphabetical order and rejects malformed, unknown, and
    illegal choices instead of guessing.
    """
    if not isinstance(action, str) or not action.strip():
        raise ActionMappingError("action must be a non-empty string")
    raw = action.strip().lower()
    if raw.startswith("switch "):
        target = _id(raw[7:])
        if not target:
            raise ActionMappingError("switch target is empty")
        matches = [i for i, mon in enumerate(_ordered_switches(state)) if _pokemon_name(mon) == target]
        if len(matches) != 1:
            raise ActionMappingError(f"unknown or ambiguous switch {action!r}")
        index = 4 + matches[0]
    else:
        tera = raw.endswith("-tera")
        move = _id(raw[:-5] if tera else raw)
        if not move:
            raise ActionMappingError(f"invalid move {action!r}")
        matches = [i for i, candidate in enumerate(_ordered_moves(state)) if _move_name(candidate) == move]
        if len(matches) != 1:
            raise ActionMappingError(f"unknown or ambiguous move {action!r}")
        if tera and not getattr(state, "can_tera"):
            raise ActionMappingError(f"tera is unavailable for {action!r}")
        index = matches[0] + (9 if tera else 0)

    if index not in legal_action_indices(state):
        raise ActionMappingError(f"illegal action {action!r} at index {index}")
    return index


def visit_distribution_to_target(visits: dict[str, object], state: object) -> list[float]:
    """Convert sparse MCTS visits to a normalized 13-way policy target."""
    if not isinstance(visits, dict) or not visits:
        raise ActionMappingError("mcts_visits must be a non-empty object")
    target = [0.0] * NUM_ACTIONS
    for action, value in visits.items():
        try:
            mass = float(value)
        except (TypeError, ValueError) as exc:
            raise ActionMappingError(f"invalid visit mass for {action!r}") from exc
        if not math.isfinite(mass) or mass < 0:
            raise ActionMappingError(f"invalid visit mass for {action!r}: {value!r}")
        target[foul_play_action_to_index(action, state)] += mass
    total = sum(target)
    if total <= 0:
        raise ActionMappingError("MCTS visit mass must be positive")
    return [mass / total for mass in target]


def canonical_visit_target_to_target(target: object, state: object) -> list[float]:
    """Validate a versioned capture-side 13-way visit target against a state."""
    if not isinstance(target, list) or len(target) != NUM_ACTIONS:
        raise ActionMappingError(f"canonical target must have {NUM_ACTIONS} actions")
    try:
        target = [float(value) for value in target]
    except (TypeError, ValueError) as exc:
        raise ActionMappingError("canonical target contains a non-numeric mass") from exc
    if any(not math.isfinite(value) or value < 0 for value in target):
        raise ActionMappingError("canonical target contains an invalid mass")
    legal = legal_action_indices(state)
    if any(value > 0 and index not in legal for index, value in enumerate(target)):
        raise ActionMappingError("canonical target assigns mass to an illegal action")
    total = sum(target)
    if total <= 0:
        raise ActionMappingError("canonical target mass must be positive")
    return [value / total for value in target]


def explicit_policy_target(record: dict[str, Any], state: object) -> list[float]:
    """Validate a v2 capture record with a stable parser timestep identity."""
    selected = record.get("canonical_selected_action_index")
    if not isinstance(selected, int) or selected not in legal_action_indices(state):
        raise ActionMappingError("canonical selected action is missing or illegal")
    return canonical_visit_target_to_target(record.get("mcts_visit_target_13"), state)


def _load_trajectory(path: Path) -> dict[str, Any]:
    if path.suffix == ".lz4":
        import lz4.frame

        with lz4.frame.open(path, "rb") as handle:
            return json.loads(handle.read().decode("utf-8"))
    return json.loads(path.read_text())


def _pov_from_filename(path: Path) -> tuple[str, str] | None:
    name = path.name
    if name.endswith(".json.lz4"):
        name = name[:-9]
    elif name.endswith(".json"):
        name = name[:-5]
    else:
        return None
    try:
        battle_tag, _, replay_fields = name.split("_", 2)
        pov = replay_fields.removeprefix("Unrated_").split("_vs_", 1)[0]
    except ValueError:
        return None
    return battle_tag, pov


def _trajectory_index(parsed_root: Path) -> dict[tuple[str, str], Path]:
    indexed: dict[tuple[str, str], Path] = {}
    ambiguous: set[tuple[str, str]] = set()
    for path in parsed_root.rglob("*.json*"):
        key = _pov_from_filename(path)
        if key is None:
            continue
        if key in indexed:
            ambiguous.add(key)
        else:
            indexed[key] = path
    for key in ambiguous:
        del indexed[key]
    return indexed


def build_trajectory_index(parsed_root: Path, output: Path) -> dict[str, int]:
    """Persist the parser's stable learner-POV identity for later sidecar joins."""
    indexed = _trajectory_index(parsed_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for (battle_tag, learner_pov), path in sorted(indexed.items()):
            raw = _load_trajectory(path)
            timesteps = len(raw.get("actions", [])) - 1
            if timesteps < 0:
                raise ValueError(f"{path}: missing actions")
            handle.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "battle_tag": battle_tag,
                        "learner_pov": learner_pov,
                        "trajectory": str(path.resolve().relative_to(parsed_root.resolve())),
                        "timesteps": timesteps,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return {"trajectories": len(indexed)}


def load_trajectory_index(path: Path, parsed_root: Path) -> dict[tuple[str, str], Path]:
    """Load a parser-produced identity manifest without accepting duplicate POVs."""
    indexed: dict[tuple[str, str], Path] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        key = (row.get("battle_tag"), row.get("learner_pov"))
        relative = row.get("trajectory")
        if row.get("schema_version") != 1 or not all(isinstance(value, str) for value in (*key, relative)):
            raise ValueError(f"{path}:{line_number}: invalid trajectory identity")
        trajectory = (parsed_root / relative).resolve()
        if parsed_root.resolve() not in trajectory.parents or not trajectory.is_file() or key in indexed:
            raise ValueError(f"{path}:{line_number}: unsafe or duplicate trajectory identity")
        indexed[key] = trajectory
    return indexed


def _load_states(raw_states: Iterable[dict[str, Any]]) -> list[object]:
    from metamon.interface import UniversalState

    return [UniversalState.from_dict(dict(state)) for state in raw_states]


def build_sidecar(
    decision_logs: Iterable[Path],
    parsed_root: Path,
    output: Path,
    trajectory_index: Path | None = None,
) -> dict[str, int]:
    """Build verified targets from complete decision sequences only.

    A decision is accepted only when the exact ``(battle_tag, username)`` POV
    has one parsed trajectory, both sequences have equal length, every selected
    Foul Play action equals the parsed action at the same position, and every
    visit action is legal in that parsed state.  This avoids unsafe joins by
    turn number, action name, or state similarity.
    """
    trajectory_by_pov = (
        load_trajectory_index(trajectory_index, parsed_root)
        if trajectory_index is not None
        else _trajectory_index(parsed_root)
    )
    rows_by_pov: dict[tuple[str, str], list[tuple[Path, int, dict[str, Any]]]] = defaultdict(list)
    invalid_rows = 0
    for log_path in decision_logs:
        for line_number, line in enumerate(log_path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            if row.get("record_type") != "decision":
                continue
            battle_tag, username = row.get("battle_tag"), row.get("username")
            if not isinstance(battle_tag, str) or not isinstance(username, str):
                invalid_rows += 1
                continue
            rows_by_pov[(battle_tag, username)].append((log_path, line_number, row))

    output.parent.mkdir(parents=True, exist_ok=True)
    accepted, rejected = 0, 0
    rejection_reasons: Counter[str] = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for pov, decisions in sorted(rows_by_pov.items()):
            trajectory = trajectory_by_pov.get(pov)
            if trajectory is None:
                rejected += len(decisions)
                rejection_reasons["missing_or_ambiguous_learner_pov"] += len(decisions)
                continue
            try:
                raw = _load_trajectory(trajectory)
                states = _load_states(raw["states"][:-1])
                actions = list(raw["actions"][:-1])
                explicit_schema = all(row.get("mcts_schema_version") == 2 for _, _, row in decisions)
                if explicit_schema:
                    if trajectory_index is None:
                        rejected += len(decisions)
                        rejection_reasons["explicit_trajectory_index_required"] += len(decisions)
                        continue
                    ordered = sorted(decisions, key=lambda item: item[2].get("mcts_decision_seq", -1))
                    sequences = [row.get("mcts_decision_seq") for _, _, row in ordered]
                    if sequences != list(range(len(ordered))) or len(ordered) != len(states):
                        rejected += len(decisions)
                        rejection_reasons["explicit_sequence_mismatch"] += len(decisions)
                        continue
                    for timestep, (source, line_number, row) in enumerate(ordered):
                        try:
                            target = explicit_policy_target(row, states[timestep])
                        except ActionMappingError:
                            rejected += 1
                            rejection_reasons["explicit_illegal_or_unmappable_target"] += 1
                            continue
                        record = {
                            "schema_version": SIDECAR_SCHEMA_VERSION,
                            "trajectory": str(trajectory.relative_to(parsed_root.resolve())),
                            "timestep": timestep,
                            "target": target,
                            "source": {"path": str(source), "line": line_number},
                        }
                        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                        accepted += 1
                    continue
                if len(decisions) != len(states) or len(actions) != len(states):
                    rejected += len(decisions)
                    rejection_reasons["decision_action_count_mismatch"] += len(decisions)
                    continue
                records = []
                for timestep, ((source, line_number, row), state, chosen) in enumerate(zip(decisions, states, actions)):
                    selected = foul_play_action_to_index(row["selected_action"], state)
                    if selected != chosen:
                        raise ActionMappingError("selected action differs from parsed action")
                    target = visit_distribution_to_target(row["mcts_visits"], state)
                    records.append(
                        {
                            "schema_version": SIDECAR_SCHEMA_VERSION,
                            "trajectory": str(trajectory.relative_to(parsed_root.resolve())),
                            "timestep": timestep,
                            "target": target,
                            "source": {"path": str(source), "line": line_number},
                        }
                    )
            except ActionMappingError as exc:
                rejected += len(decisions)
                if str(exc) == "selected action differs from parsed action":
                    rejection_reasons["selected_action_mismatch"] += len(decisions)
                else:
                    rejection_reasons["illegal_or_unmappable_action"] += len(decisions)
                continue
            except (KeyError, TypeError, ValueError, OSError) as exc:
                rejected += len(decisions)
                rejection_reasons[type(exc).__name__] += len(decisions)
                continue
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            accepted += len(records)
    result = {"accepted": accepted, "rejected": rejected, "invalid_rows": invalid_rows}
    result.update({f"rejected_{reason}": count for reason, count in sorted(rejection_reasons.items())})
    return result


def load_sidecar(path: Path) -> dict[str, dict[int, list[float]]]:
    targets: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        if row.get("schema_version") != SIDECAR_SCHEMA_VERSION:
            raise ValueError(f"{path}:{line_number}: unsupported sidecar schema")
        trajectory, timestep, target = row.get("trajectory"), row.get("timestep"), row.get("target")
        if not isinstance(trajectory, str) or not isinstance(timestep, int):
            raise ValueError(f"{path}:{line_number}: invalid trajectory key")
        if not isinstance(target, list) or len(target) != NUM_ACTIONS:
            raise ValueError(f"{path}:{line_number}: target must have {NUM_ACTIONS} actions")
        target = [float(value) for value in target]
        if timestep in targets[trajectory] or not math.isclose(sum(target), 1.0, abs_tol=1e-6):
            raise ValueError(f"{path}:{line_number}: duplicate or unnormalized target")
        targets[trajectory][timestep] = target
    return targets


def add_distillation_loss(total_loss, probs, targets, target_mask, coefficient: float):
    """Add masked cross-entropy from target visits; disabled mode is an exact no-op."""
    if coefficient <= 0:
        return total_loss
    import torch
    import amago

    legal = ~target_mask["illegal_actions"].bool()
    target = targets * legal.to(targets.dtype)
    mass = target.sum(dim=-1, keepdim=True)
    valid = target_mask["valid"].bool() & (mass > 0) & ((targets * (~legal)).sum(dim=-1, keepdim=True) == 0)
    target = target / mass.clamp_min(torch.finfo(target.dtype).eps)
    ce = -(target * probs.clamp_min(torch.finfo(probs.dtype).eps).log()).sum(dim=-1, keepdim=True)
    return total_loss + coefficient * amago.utils.masked_avg(ce, valid.expand_as(ce))


def install_mcts_policy_distillation(sidecar_path: str, coefficient: float) -> None:
    """Install the optional dataset and agent patches used by the finetune runner."""
    if coefficient <= 0:
        raise ValueError("MCTS policy distillation requires a positive coefficient")
    import gin
    import torch
    import metamon.rl.custom_agent as ca
    import metamon.rl.dataset_config as dc
    import metamon.rl.metamon_to_amago as m2a

    sidecar = Path(sidecar_path).resolve()
    target_by_trajectory = load_sidecar(sidecar)
    base_cls = m2a.MetamonAMAGODataset

    class MCTSTargetDataset(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.parsed_replay_dset.max_seq_len is not None:
                raise ValueError("MCTS policy sidecars require unsliced parsed trajectories")

        def sample_random_trajectory(self):
            dataset = self.parsed_replay_dset
            index = random.randrange(len(dataset))
            filename = Path(dataset.filenames[index])
            data = dataset[index]
            rl_data = self._process_data(data)
            root = Path(getattr(dataset, "dset_root", ""))
            try:
                key = str(filename.relative_to(root))
            except ValueError:
                key = filename.name
            labels = target_by_trajectory.get(key, {})
            length = rl_data.actions.shape[0] + 1
            targets = torch.zeros((length, NUM_ACTIONS), dtype=torch.float32)
            valid = torch.zeros((length, 1), dtype=torch.bool)
            for timestep, target in labels.items():
                if timestep >= length - 1:
                    raise ValueError(f"sidecar timestep {timestep} is outside {key}")
                targets[timestep] = torch.tensor(target)
                valid[timestep] = True
            rl_data.obs["mcts_policy_target"] = targets
            rl_data.obs["mcts_policy_target_mask"] = valid
            return rl_data

    class MCTSPolicyDistillationAgent(ca.MetamonFinetuneAgent):
        def __init__(self, *args, mcts_policy_coeff: float = coefficient, **kwargs):
            super().__init__(*args, **kwargs)
            self.mcts_policy_coeff = mcts_policy_coeff

        def forward(self, batch, log_step: bool):
            total_loss = super().forward(batch, log_step)
            if self.mcts_policy_coeff <= 0 or "mcts_policy_target" not in batch.obs:
                return total_loss
            encoded = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
            state_rep, _ = self.traj_encoder(encoded, time_idxs=batch.time_idxs, hidden_state=None)
            probs = self.actor(
                state_rep,
                straight_from_obs={key: batch.obs[key] for key in self.pass_obs_keys_to_actor},
            ).probs[:, :-1]
            valid = batch.obs["mcts_policy_target_mask"][:, :-1] & ~batch.obs["missing_action_mask"][:, :-1]
            state_valid = (~(batch.rl2s == self.pad_val).all(-1, keepdim=True)).bool()[:, 1:]
            before = total_loss
            total_loss = add_distillation_loss(
                total_loss,
                probs,
                batch.obs["mcts_policy_target"][:, :-1].unsqueeze(-2),
                {
                    "valid": (valid & state_valid).unsqueeze(-2),
                    "illegal_actions": batch.obs["illegal_actions"][:, :-1].unsqueeze(-2),
                },
                self.mcts_policy_coeff,
            )
            if log_step:
                self.update_info["MCTS Policy Loss"] = (total_loss - before).detach()
            return total_loss

    gin.external_configurable(MCTSPolicyDistillationAgent, module="custom_agent")
    ca.MCTSPolicyDistillationAgent = MCTSPolicyDistillationAgent
    m2a.MetamonAMAGODataset = MCTSTargetDataset
    dc.MetamonAMAGODataset = MCTSTargetDataset
