# Resource-aware long-horizon expert: development result

Date: 2026-08-14

Decision: **the explicit resource-shadow treatment fails development; do not
touch holdout, run H2H, or distill its deviations.**

## Contract and calibration

The corrected local engine exposes Tera variants on 437/1,000 source roots
(6,992/16,000 sampled worlds). The old 50k action-Q oracle contains no Tera
variants and is not reusable for this question. The new Rust resource extractor
matches the Python reference over 1,000 sampled roots with maximum absolute
error `1.57e-7`.

The nonnegative resource model trained on 184,213 decisions / 4,998 battles.
Battle-disjoint test performance was 68.93% accuracy and 0.19702 Brier versus a
0.25000 constant reference. This is calibration evidence only. Artifact SHA-256:
`84e6176f72453db7eb718b49e5896b911c3d7556ae4624ce09b0f0c0b20345bf`;
engine model SHA-256:
`9ed2aa54ecc7d168f473d4db264f48b9850763ada048873c4f377a7b5f444dd5`.

The frozen development and untouched holdout panels each contain 50 unique
battles, 25 per Tera stratum, with zero overlap. Development panel SHA-256:
`55105e7b336b68a2658e6456322f0e34dbe5ef394c15076b14acb3046862ac63`;
holdout panel SHA-256:
`c622aa7d4a016c6d025c2f5562102add3404e2347440ee5f79a27d5b51636d5c`.

## Development gate

All metrics cover 100 fixed schedule units from 50 source battles. Lower regret
is better.

| Arm | Mean regret | Oracle top-1 | Tera rate | Regret >= 0.10 |
| --- | ---: | ---: | ---: | ---: |
| Archived historical 500 ms action | 0.041663 | 48% | 2% | 18 |
| Equal-depth hand search | **0.001690** | **88%** | 0% | 0 |
| Resource blend 0.05 | 0.001690 | 86% | 0% | 0 |
| Resource blend 0.10 | 0.002251 | 85% | 0% | 0 |
| Resource blend 0.15 | 0.002060 | 86% | 0% | 0 |
| Resource blend 0.25 | 0.003739 | 85% | 0% | 0 |

Resource-minus-hand mean regret improvement and source-battle bootstrap 95%
intervals were:

- weight 0.05: `-0.00000077`, `[-0.00000231, 0.00000000]`, 2/100 actions changed;
- weight 0.10: `-0.00056164`, `[-0.00230712, 0.00062219]`, 5/100 changed;
- weight 0.15: `-0.00037022`, `[-0.00230712, 0.00119724]`, 6/100 changed;
- weight 0.25: `-0.00204924`, `[-0.00617676, 0.00079824]`, 7/100 changed.

Every nonzero blend fails the equal-depth and positive-confidence gates. None
adds a catastrophic miss, but abstaining back to the hand evaluator is strictly
better on this development estimand.

## Interpretation

The shadow prices are predictive and directionally constrained, but that does
not make them causal action values. Adding them to search does not improve the
independent root metric. The result reproduces the central MegaGem warning in
the useful direction: offline value calibration is insufficient; the selector
must win independently before distillation.

The equal-depth hand search looks far better than the archived historical
action under a deeper hand-search oracle. This does **not** prove a current R1
gain: the historical comparator predates causal-history R1, and the oracle
shares the hand evaluator's inductive bias. It is a lead for a separately
preregistered corrected high-budget teacher, not authorization to reuse the
resource model or collect student targets.

No GPU, cloud service, new games, public ladder matches, or paid compute were
used. The frozen holdout was not evaluated.
