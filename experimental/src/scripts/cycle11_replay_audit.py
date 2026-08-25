#!/usr/bin/env python3
"""Cycle 11's frozen, narrow replay-transport repairs over Cycle 9."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from experimental.src.scripts import cycle9_replay_audit as v9


ReplayAuditError = v9.ReplayAuditError
_BASE_CANONICAL_PUBLIC_LINES = v9.canonical_public_lines
SUPPORTED_COMMITS = frozenset({
    "07d1669294c8b75c3ee65ed3cc5e45b97241c2af",
    "755f4665a341e3a9533baf1e1a3802fd233b0d69",
    "7c8bd622d31ba37f0eaebe25916e2d7cf29ff33d",
    "a4b6781ad07ff854a5e25c73fc04b699caa3d96a",
    "e440c4a18385274f10c405d0b158b6a962ce6d94",
    "e59a52c8fd506249986e158cfa352cc2cb610022",
    "f8ac14003a5f27e1bdc8d8c59608a773c1cb96e5",
})
NAME_CHANGE_FORFEIT_COMMITS = frozenset({
    "755f4665a341e3a9533baf1e1a3802fd233b0d69",
    "e440c4a18385274f10c405d0b158b6a962ce6d94",
    "f8ac14003a5f27e1bdc8d8c59608a773c1cb96e5",
})
HP_RE = re.compile(r"(\d+)/(\d+)(.*)")
POKEMON_IDENT_RE = re.compile(r"p[12][a-z]?: .+")


def _public_hp(value: str) -> str:
    match = HP_RE.fullmatch(value)
    if not match or int(match.group(2)) <= 0:
        return value
    hp, maximum = int(match.group(1)), int(match.group(2))
    displayed = math.ceil(100 * hp / maximum)
    if displayed == 100 and hp < maximum:
        displayed = 99
    return f"{displayed}/100{match.group(3)}"


def canonical_public_lines(
    lines: Sequence[str], *, inputlog: str, showdown_commit: str,
) -> list[str]:
    if showdown_commit not in SUPPORTED_COMMITS:
        raise ReplayAuditError("worktree commit is outside frozen Cycle 11 support")
    base = _BASE_CANONICAL_PUBLIC_LINES(
        lines, inputlog=inputlog, showdown_commit=showdown_commit,
    )
    loser = v9._forced_loser_name(inputlog, lines)
    result: list[str] = []
    for line in base:
        if (
            loser
            and showdown_commit in NAME_CHANGE_FORFEIT_COMMITS
            and line == f"|-message|{loser} forfeited by changing their name."
        ):
            continue
        fields = line.split("|")
        if len(fields) > 3 and fields[1] == "-sethp":
            # Protocol grammar is one or more POKEMON, HP pairs followed by
            # optional effect/silent fields. Normalize HP only after a valid
            # public Pokemon ident; never scan arbitrary trailing fields.
            index = 2
            while index + 1 < len(fields) and POKEMON_IDENT_RE.fullmatch(fields[index]):
                fields[index + 1] = _public_hp(fields[index + 1])
                index += 2
            line = "|".join(fields)
        result.append(line)
    return result


def public_lines_from_capture(
    public_capture: Mapping[str, Any], *, inputlog: str, showdown_commit: str,
) -> list[str]:
    chunks = public_capture.get("public_chunks")
    if not isinstance(chunks, list):
        raise ReplayAuditError("public capture lacks chunks")
    lines: list[str] = []
    for row in chunks:
        if not isinstance(row, Mapping) or not isinstance(row.get("data"), str):
            raise ReplayAuditError("malformed public chunk")
        lines.extend(row["data"].splitlines())
    return canonical_public_lines(
        lines, inputlog=inputlog, showdown_commit=showdown_commit,
    )


def materialize_role(**kwargs: Any) -> dict[str, Any]:
    """Reuse Cycle 9 request semantics with Cycle 11 public canonicalization."""
    original = v9.canonical_public_lines
    v9.canonical_public_lines = canonical_public_lines
    try:
        return v9.materialize_role(**kwargs)
    finally:
        v9.canonical_public_lines = original
