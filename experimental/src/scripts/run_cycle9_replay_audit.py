#!/usr/bin/env python3
"""Use the frozen Cycle 8 runner with Cycle 9's repaired semantic contract."""

from experimental.src.scripts import run_cycle8_replay_audit as runner
from experimental.src.scripts import cycle9_replay_audit as cycle9


def main() -> None:
    runner.canonical_public_lines = cycle9.canonical_public_lines
    runner.public_lines_from_capture = cycle9.public_lines_from_capture
    runner.materialize_role = cycle9.materialize_role

    original_audit_positive = runner.audit_positive

    def audit_positive(**kwargs):
        row = kwargs["row"]
        # Cycle 8's audit function calls the three patched functions without a
        # commit argument. Bind the exact selected commit for this one call.
        commit = row["showdown_commit"]
        old_canonical = runner.canonical_public_lines
        old_capture = runner.public_lines_from_capture
        old_materialize = runner.materialize_role
        runner.canonical_public_lines = lambda lines, *, inputlog: cycle9.canonical_public_lines(
            lines, inputlog=inputlog, showdown_commit=commit,
        )
        runner.public_lines_from_capture = lambda capture, *, inputlog: cycle9.public_lines_from_capture(
            capture, inputlog=inputlog, showdown_commit=commit,
        )
        runner.materialize_role = lambda **call: cycle9.materialize_role(
            **call, showdown_commit=commit,
        )
        try:
            return original_audit_positive(**kwargs)
        finally:
            runner.canonical_public_lines = old_canonical
            runner.public_lines_from_capture = old_capture
            runner.materialize_role = old_materialize

    runner.audit_positive = audit_positive
    runner.main()


if __name__ == "__main__":
    main()
