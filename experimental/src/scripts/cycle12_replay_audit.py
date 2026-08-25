#!/usr/bin/env python3
"""Cycle 12's exact server/UI transport normalization over Cycle 11."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from experimental.src.scripts import cycle11_replay_audit as v11
from srcs.metagross.causal_reveal_ledger import norm


ReplayAuditError = v11.ReplayAuditError
_BASE_CANONICAL_PUBLIC_LINES = v11.canonical_public_lines
INVITE_RE = re.compile(r"^\|\|Invite sent to (.+)!$")
HIDELINES_UNLINK_RE = re.compile(r"^\|hidelines\|unlink\|([^|]+)$")
SIMPLE_FORFEIT_RE = re.compile(r"^\|-message\|(.+) forfeited\.$")
LOOKUP_ERROR_RE = re.compile(
    r"^\|error\|'[^|]+' doesn't match any Pokémon, item, move, ability or nature\. "
    r"\(Check your spelling\?\)$"
)
MODERATED_CHAT_LINE = (
    '|raw|<div class="broadcast-red"><strong>Moderated chat was set to +!</strong>'
    '<br />Only users of rank + and higher can talk.</div>'
)


def authenticated_player_names(lines: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for line in lines:
        fields = line.split("|")
        if (
            len(fields) >= 4 and fields[1] == "player"
            and fields[2] in {"p1", "p2"} and fields[3]
        ):
            result.add(fields[3])
    return result


def _server_transport(line: str, player_names: set[str]) -> bool:
    invite = INVITE_RE.fullmatch(line)
    if invite:
        return invite.group(1) in player_names
    hidden = HIDELINES_UNLINK_RE.fullmatch(line)
    if hidden:
        return norm(hidden.group(1)) in {norm(name) for name in player_names}
    forfeited = SIMPLE_FORFEIT_RE.fullmatch(line)
    if forfeited:
        return forfeited.group(1) in player_names
    return line == MODERATED_CHAT_LINE or LOOKUP_ERROR_RE.fullmatch(line) is not None


def canonical_public_lines(
    lines: Sequence[str], *, inputlog: str, showdown_commit: str,
) -> list[str]:
    names = authenticated_player_names(lines)
    base = _BASE_CANONICAL_PUBLIC_LINES(
        lines, inputlog=inputlog, showdown_commit=showdown_commit,
    )
    return [line for line in base if not _server_transport(line, names)]


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
    original = v11.canonical_public_lines
    v11.canonical_public_lines = canonical_public_lines
    try:
        return v11.materialize_role(**kwargs)
    finally:
        v11.canonical_public_lines = original

