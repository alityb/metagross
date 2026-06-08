from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Ladder evaluation report helper")
    parser.add_argument("--gxe", type=float, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--deviation", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps({"gxe": args.gxe, "games": args.games, "deviation": args.deviation, "target_met": args.gxe > 90 and args.games >= 200 and args.deviation < 50}, indent=2))


if __name__ == "__main__":
    main()
