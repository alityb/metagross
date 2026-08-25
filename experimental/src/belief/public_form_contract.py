"""Source-pinned Pokemon Showdown public battle/cosmetic form contract."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from belief.causal_protocol_bridge import norm


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "experimental/src/scripts/export_showdown_public_form_contract.cjs"


class PublicFormContractError(ValueError):
    pass


@dataclass(frozen=True)
class PublicFormContract:
    mapping: Mapping[str, str]
    authority: Mapping[str, str]

    def canonical(self, value: Any) -> str:
        species = norm(value)
        return self.mapping.get(species, species)


def load_public_form_contract() -> PublicFormContract:
    completed = subprocess.run(
        ["node", str(HELPER)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PublicFormContractError("Showdown form helper emitted invalid JSON") from exc
    if payload.get("schema") != "metagross-showdown-public-form-contract/v1":
        raise PublicFormContractError("unexpected Showdown form schema")
    rows = payload.get("rows")
    if not isinstance(rows, list) or payload.get("mapping_count") != len(rows):
        raise PublicFormContractError("invalid Showdown form rows")
    mapping: dict[str, str] = {}
    authority: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise PublicFormContractError("invalid Showdown form row")
        source, target, source_authority = row.get("source"), row.get("target"), row.get("authority")
        if (
            not isinstance(source, str) or not source
            or not isinstance(target, str) or not target
            or source_authority not in {"battleOnly", "cosmeticFormes"}
            or norm(source) != source or norm(target) != target
            or source in mapping
        ):
            raise PublicFormContractError("invalid Showdown form mapping")
        mapping[source] = target
        authority[source] = source_authority
    if len(mapping) < 100:
        raise PublicFormContractError("implausibly small Showdown form mapping")
    return PublicFormContract(mapping=mapping, authority=authority)
