ROOT_SYSTEM_PROMPT = """You are the AlphaPokemon recursive strategist.
The full battle log is not in this prompt. It is available inside the Python REPL
as BATTLE_LOG. Use programmatic filtering, sub_rlm calls, and REPL variables to
build POSTERIOR, STRATEGIC_PRIOR, and V_RLM. Never ask to see the whole log.
Write only Python code. Set FINAL = True when done.
"""

SUB_SYSTEM_PROMPT = """You are a focused Pokemon set-inference worker.
Return compact JSON only. Prefer conservative posteriors that keep all plausible
sets over overconfident eliminations.
"""


def root_iteration_prompt(metadata: dict, snapshot: dict, observations: list[str]) -> str:
    recent = "\n".join(observations[-3:]) if observations else "No REPL observations yet."
    return f"""{ROOT_SYSTEM_PROMPT}

Metadata:
{metadata}

Current REPL snapshot:
{snapshot}

Recent observations:
{recent}

Write the next Python code block. Keep stdout concise.
"""
