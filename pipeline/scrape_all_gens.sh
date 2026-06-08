#!/bin/bash
set -euo pipefail
cd "${HOME}/metagross"
FORMATS="gen1randombattle gen2randombattle gen3randombattle gen4randombattle gen5randombattle gen6randombattle gen7randombattle gen8randombattle"
for fmt in $FORMATS; do
    echo "=== Scraping $fmt ==="
    .venv/bin/python -m data_pipeline.scrape_replays \
        --output "data/replay_ids_${fmt}.jsonl" \
        --format "$fmt" \
        --target 5000 \
        --min-rating 1200
    .venv/bin/python -m data_pipeline.download_replays \
        --ids "data/replay_ids_${fmt}.jsonl" \
        --output "data/raw_replays_${fmt}" \
        --workers 8
    .venv/bin/python -m data_pipeline.parse_replays \
        --raw-dir "data/raw_replays_${fmt}" \
        --output "data/parsed_${fmt}" \
        --pool data/all_gen_pool.json \
        --format "$fmt" \
        --workers 4
    echo "Done $fmt"
done
