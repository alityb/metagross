# Provenance

## Accepted Result

The accepted r1 agent was run on the public Pokemon Showdown
`gen9randombattle` ladder as `metaexitr1`.

| Metric | Value |
|---|---:|
| Settled GXE | 92.4-92.7 |
| Rating deviation | 25 |
| Peak observed GXE | 93.6 |
| Previous project best | 91.4 GXE |

The result predates the later formal paired promotion protocol. Historical raw
logs were retained in `experimental/runs/exit_ladder_fresh`, but the repository
does not contain a single immutable final ladder API snapshot. The result is
therefore reported as a historical observed range, not a newly reproduced
benchmark.

## Frozen Components

| Component | Revision/artifact |
|---|---|
| Foul Play | `e1e2ca650598621e85c3b6ab751c66e625489934` plus `srcs/patches/foul-play-r1.patch` |
| Metamon | `0a00a759c9a4382a2877088d828302ec294a05a5` plus `srcs/patches/metamon-r1.patch` |
| poke-engine | 0.0.47 fork in `srcs/vendor/poke-engine`; stock delta in `srcs/patches/poke-engine-0.0.47-priors-v2.patch` |
| Policy | `randbats_exit_r1`, epoch 5 |

Artifact hashes are machine-readable in
`results/accepted-r1/artifacts.json`.

## Verified

- The checkpoint loaded through the real policy-server path.
- The player root prior was active in the deployed search configuration.
- The checkpoint matched 642 model keys and 142,832,563 parameters.
- The recovered patched engine passed a 100-game equivalence check against the
  deployment binary: 47/100, statistically indistinguishable from 50%.

## Caveat

Opponent priors were configured in the historical deployment, but historical
coverage telemetry was not retained. A later opponent-view adapter bug was
found during unrelated experiments. The production source preserves the final
corrected adapter, but this repository does not claim a separately measured
GXE contribution from opponent priors.
