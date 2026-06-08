from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pathlib import Path

from model.state import normalize_name


def extract_block(text: str, start: int) -> tuple[str, int]:
    depth = 0
    block_start = start
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            if depth == 0:
                block_start = index + 1
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[block_start:index], index + 1
    return "", start


def parse_learnsets_text(text: str) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, set[str]]] = {}
    species_pattern = re.compile(r"\n\s*([a-z0-9]+)\s*:\s*\{", re.IGNORECASE)
    for match in species_pattern.finditer("\n" + text):
        species = normalize_name(match.group(1))
        block, _end = extract_block("\n" + text, match.end() - 1)
        learnset_match = re.search(r"learnset\s*:\s*\{", block)
        if not learnset_match:
            continue
        learnset_block, _ = extract_block(block, learnset_match.end() - 1)
        for move_match in re.finditer(r"([a-z0-9]+)\s*:\s*\[([^\]]*)\]", learnset_block, re.IGNORECASE | re.DOTALL):
            move = normalize_name(move_match.group(1))
            sources = move_match.group(2)
            gens = {int(gen) for gen in re.findall(r"['\"]?([1-9])", sources)}
            if not gens:
                gens = {9}
            for gen in gens:
                result.setdefault(species, {}).setdefault(f"gen{gen}", set()).add(move)
    return {species: {gen: sorted(moves) for gen, moves in gens.items()} for species, gens in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Pokemon Showdown learnsets JS/TS into JSON")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default="data/learnsets.json")
    args = parser.parse_args()
    candidates = []
    if args.input:
        candidates.append(Path(args.input))
    candidates.extend(
        [
            Path.home() / "ps-server-gen9/data/learnsets.js",
            Path.home() / "ps-server-gen9/data/learnsets.ts",
            Path.home() / "ps-server/data/learnsets.js",
            Path.home() / "ps-server/data/learnsets.ts",
        ]
    )
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise SystemExit("No learnsets.js/learnsets.ts found. Pass --input or install Pokemon Showdown data.")
    parsed = parse_learnsets_text(source.read_text(encoding="utf-8", errors="ignore"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(parsed, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"source": str(source), "species": len(parsed), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
