"""Canonical audit projection for Foul Play production sampling state."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from srcs.metagross.causal_reveal_ledger import MOVE_RECEIPT_ATTRIBUTE


class ProductionProjectionError(RuntimeError):
    pass


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _canonical(value: Any, active: set[int]) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"kind": "float", "value": "nan"}
        if math.isinf(value):
            return {"kind": "float", "value": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}

    identity = id(value)
    if identity in active:
        raise ProductionProjectionError("cycle in Foul Play mechanical state")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            entries = [
                [_canonical(key, active), _canonical(item, active)]
                for key, item in value.items()
            ]
            entries.sort(
                key=lambda row: json.dumps(
                    row[0], sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
            )
            return {"kind": "mapping", "type": _type_name(value), "entries": entries}
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return {
                "kind": "namedtuple",
                "type": _type_name(value),
                "fields": [
                    [name, _canonical(getattr(value, name), active)]
                    for name in value._fields
                ],
            }
        if isinstance(value, (list, tuple)):
            return {
                "kind": "sequence",
                "type": _type_name(value),
                "values": [_canonical(item, active) for item in value],
            }
        if isinstance(value, (set, frozenset)):
            values = [_canonical(item, active) for item in value]
            values.sort(
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
            )
            return {"kind": "set", "type": _type_name(value), "values": values}
        fields = getattr(value, "__dict__", None)
        if isinstance(fields, dict):
            projected = []
            for name in sorted(fields):
                if name == MOVE_RECEIPT_ATTRIBUTE:
                    continue
                projected.append([name, _canonical(fields[name], active)])
            return {"kind": "object", "type": _type_name(value), "fields": projected}
    finally:
        active.remove(identity)
    raise ProductionProjectionError(
        f"unsupported Foul Play mechanical value: {_type_name(value)}"
    )


def canonical_mechanical_projection(value: Any) -> bytes:
    payload = {
        "schema": "metagross-production-sampler-mechanical-projection/v1",
        "state": _canonical(value, set()),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def mechanical_projection_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_mechanical_projection(value)).hexdigest()
