from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.sample_types import TrainingSample
from metagross.model.state import EncodedState, stack_encoded


def load_annotations(annotation_dir: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    root = Path(annotation_dir)
    annotations: dict[tuple[str, int], dict[str, Any]] = {}
    if not root.exists():
        return annotations
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        battle_id = str(payload.get("battle_id") or payload.get("replay_id") or path.stem)
        if isinstance(payload.get("turns"), list):
            for turn in payload["turns"]:
                if isinstance(turn, dict) and "turn" in turn:
                    annotations[(battle_id, int(turn["turn"]))] = turn
        elif isinstance(payload.get("decisions"), dict):
            for turn_text, decision in payload["decisions"].items():
                if isinstance(decision, dict):
                    annotations[(battle_id, int(turn_text))] = decision
    return annotations


class ReplayDataset(Dataset[dict[str, Any]]):
    """Small-dataset eager loader (used when n_files <= 10)."""

    def __init__(
        self,
        data_dir: str | Path = "data/parsed",
        annotation_dir: str | Path = "data/annotations",
        n_samples: int | None = None,
        shuffle: bool = True,
    ) -> None:
        self.annotations = load_annotations(annotation_dir)
        self.samples: list[TrainingSample] = []
        # Only load from the given dir, not sibling dirs, to avoid cross-contamination
        root = Path(data_dir)
        paths = sorted(root.glob("*.pkl")) if root.exists() else []
        for path in paths:
            try:
                loaded = pickle.load(path.open("rb"))
            except Exception:
                continue
            self.samples.extend(s for s in loaded if isinstance(s, TrainingSample))
            if n_samples is not None and len(self.samples) >= n_samples:
                self.samples = self.samples[:n_samples]
                break
        if shuffle:
            random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _make_item(self.samples[index], self.annotations)


class StreamingReplayDataset(torch.utils.data.IterableDataset):  # type: ignore[type-arg]
    """Streaming dataset — loads one pkl file at a time, constant RAM usage.
    
    Each worker gets a disjoint slice of files so there is no duplication.
    With n workers and 45 files, each worker processes 45/n files per epoch.
    """

    def __init__(
        self,
        data_dirs: list[str | Path],
        annotation_dir: str | Path = "data/annotations",
        shuffle_files: bool = True,
        shuffle_within: bool = True,
    ) -> None:
        self.annotations = load_annotations(annotation_dir)
        self.shuffle_files = shuffle_files
        self.shuffle_within = shuffle_within
        self._all_files: list[Path] = []
        for d in data_dirs:
            root = Path(d)
            if root.exists():
                self._all_files.extend(sorted(root.glob("*.pkl")))
        if shuffle_files:
            random.shuffle(self._all_files)

    def __len__(self) -> int:
        # Estimate: count files × 100K samples/file (avoids reading all files)
        return len(self._all_files) * 100_000

    def __iter__(self):  # type: ignore[override]
        worker_info = torch.utils.data.get_worker_info()
        files = list(self._all_files)
        if worker_info is not None:
            # Split files evenly across workers
            files = files[worker_info.id :: worker_info.num_workers]
        if self.shuffle_files:
            random.shuffle(files)
        for path in files:
            try:
                batch = pickle.load(path.open("rb"))
            except Exception:
                continue
            samples = [s for s in batch if isinstance(s, TrainingSample)]
            if self.shuffle_within:
                random.shuffle(samples)
            for sample in samples:
                yield _make_item(sample, self.annotations)


def _make_item(sample: TrainingSample, annotations: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    annotation = annotations.get((sample.battle_id, sample.turn))
    v_rlm = 0.0
    belief_target = None
    has_annotation = annotation is not None
    if annotation:
        v_rlm = float(annotation.get("v_rlm", annotation.get("V_rlm", 0.0)))
        belief_target = annotation.get("belief_posterior") or annotation.get("posterior")
    return {
        "encoded_state": sample.encoded_state,
        "action": int(sample.human_action),
        "outcome": float(sample.outcome),
        "v_rlm": v_rlm,
        "belief_target": belief_target,
        "true_opponent_team": sample.true_opponent_team,
        "has_annotation": has_annotation,
        "battle_id": sample.battle_id,
        "turn": sample.turn,
        "generation": getattr(sample, "generation", 9),
    }


class MultiGenReplayDataset(ReplayDataset):
    pass


def _parsed_roots(data_dir: Path) -> list[Path]:
    if any(char in str(data_dir) for char in "*?[]"):
        return [path for path in sorted(data_dir.parent.glob(data_dir.name)) if path.is_dir()]
    roots = [data_dir] if data_dir.exists() else []
    if data_dir.name == "parsed":
        roots.extend(path for path in sorted(data_dir.parent.glob("parsed_*")) if path.is_dir())
        synthetic = data_dir.parent / "synthetic"
        if synthetic.exists():
            roots.append(synthetic)
    elif data_dir.name == "synthetic":
        parsed = data_dir.parent / "parsed"
        if parsed.exists():
            roots.append(parsed)
        roots.extend(path for path in sorted(data_dir.parent.glob("parsed_*")) if path.is_dir())
    return sorted(set(roots))


def _belief_entropy(target: Any) -> float:
    if not isinstance(target, dict):
        return 0.0
    entropies = []
    for entries in target.values():
        if not isinstance(entries, list) or not entries:
            continue
        probs = [max(0.0, float(entry.get("probability", 0.0))) for entry in entries if isinstance(entry, dict)]
        total = sum(probs)
        if total <= 0:
            continue
        normalized = [prob / total for prob in probs]
        entropies.append(-sum(prob * np.log(max(prob, 1e-12)) for prob in normalized))
    return float(np.mean(entropies)) if entropies else 0.0


def collate_replay_samples(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    batch = list(records)
    model_batch = stack_encoded([record["encoded_state"] for record in batch])
    actions = np.asarray([record["action"] for record in batch], dtype=np.int64)
    policy_targets = np.zeros((len(batch), 14), dtype=np.float32)
    for row, action in enumerate(actions):
        if 0 <= action < 14:
            policy_targets[row, action] = 1.0
    model_batch.update(
        {
            "actions": actions,
            "outcomes": np.asarray([record["outcome"] for record in batch], dtype=np.float32),
            "v_rlm": np.asarray([record["v_rlm"] for record in batch], dtype=np.float32),
            "belief_entropy": np.asarray([_belief_entropy(record.get("belief_target")) for record in batch], dtype=np.float32),
            "policy_targets": policy_targets,
            "has_annotation": np.asarray([record["has_annotation"] for record in batch], dtype=np.bool_),
            "metadata": [(record["battle_id"], record["turn"]) for record in batch],
        }
    )
    return model_batch


def torch_targets(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "actions": torch.as_tensor(batch["actions"], dtype=torch.long, device=device),
        "outcomes": torch.as_tensor(batch["outcomes"], dtype=torch.float32, device=device),
        "v_rlm": torch.as_tensor(batch["v_rlm"], dtype=torch.float32, device=device),
        "belief_entropy": torch.as_tensor(batch["belief_entropy"], dtype=torch.float32, device=device),
        "policy_targets": torch.as_tensor(batch["policy_targets"], dtype=torch.float32, device=device),
        "has_annotation": torch.as_tensor(batch.get("has_annotation", np.ones(len(batch["actions"]))), dtype=torch.bool, device=device),
    }
