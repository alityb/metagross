# Causal-history R1 versus legacy-stateless R1

Date: 2026-08-14

Execution: local CPU only

Format: Gen 9 Random Battle

Search: identical production controller, 500 ms, parallelism 8, one search thread,
`c_puct=2.0`

## Treatment

Both arms used the same frozen R1 epoch-5 checkpoint, private-request legality,
exact selected-action receipts, belief/search implementation, randomization
scheme, and mirrored teams/seeds. The only treatment was the player-policy
trajectory input:

- **Causal history:** the repaired AMAGO sequence of real observations,
  reward-first previous action/reward inputs, and absolute time indices.
- **Legacy stateless:** the exact previous production input conditional on the
  same repaired current observation: one zero dummy timestep, current
  observation, all-zero RL2, and time indices `[0, 1]`.

The comparator does not reintroduce public-event action inference or any other
known bookkeeping bug. This isolates the information representation rather
than comparing two different harnesses.

## Live canary and parity

Before the A/B, a separate mirrored causal-history self-canary completed 1-1
with no voids. Across its two isolated servers, 138 live decisions in four
player trajectories were independently recomputed from the durable dumps with
maximum probability error `0.0`.

The A/B integration smoke then completed 2-0 for causal history. It was
exploratory and is not pooled with the primary screen.

All 735 causal-policy decisions from the smoke and primary screen were again
recomputed from the durable multi-battle dump. All 22 player trajectories and
all 735 probability vectors matched exactly, with maximum absolute difference
`0.0` at tolerance `1e-7`.

## Primary 20-game screen

| Metric | Result |
| --- | ---: |
| Causal-history wins | 11 |
| Legacy-stateless wins | 9 |
| Point estimate | 55.0% |
| Wilson 95% interval | 34.2%–74.2% |
| Complete mirrored pairs | 10 |
| Causal / split / stateless sweeps | 3 / 5 / 2 |
| Pair-bootstrap 95% interval | 35.0%–75.0% |
| Voids / unknowns | 0 / 0 |

The 55% point estimate corresponds to roughly **+35 head-to-head Elo** if it
were the true rate. The sample is far too small to claim that effect: both the
ordinary and pair-bootstrap intervals include 50% by a wide margin. The role
split was also volatile (causal 8-2 as challenger and 3-7 as acceptor), although
the mirrored pair score remains 55%.

## Mechanism

Counterfactual legacy-stateless inference was recomputed on the exact same 735
causal observations:

- top action changed on **32.4%** of decisions;
- mean total-variation distance was **0.282** (median 0.224, p90 0.626);
- mean Jensen-Shannon divergence was 0.0958 nats;
- on 713 decisions with a durable selected-action receipt, the causal prior's
  top action matched the eventual 500 ms search choice 59.7% of the time versus
  52.2% for stateless inference;
- causal mean probability on the selected search action was 0.514 versus 0.453.

The search-alignment comparison is descriptive and favors the treatment by
construction because the played search consumed the causal prior. Battle
outcomes, not agreement, remain the strength estimand.

## Decision

Keep and commit the causal-history repair as the correct production
architecture. It is not a cosmetic change: it materially changes policy and
has a positive live point estimate. But this screen does **not** establish that
the bot is stronger, and the observed effect is not the dramatic gain sought.

Do not extrapolate 11-9 into a GXE claim or immediately spend on a 500-game
promotion gate. A true +35 H2H Elo would be useful but is not enough by itself
to credibly move the accepted 92%+ GXE agent to 95% GXE.

The next performance branch remains a long-horizon, resource-aware expert:

1. Freeze battle-disjoint roots and common determinizations/seeds.
2. Require the expert to value future HP, Tera, PP, tempo, revealed
   information, and switch resources through continuation play.
3. Prove independent root/outcome advantage before collecting targets.
4. Distill only confident deviations, with pass-through R1 and explicit
   resource-preservation anchors.
5. Test the resulting weights at the same 500 ms budget.

## Primary artifacts

- `screen/result.json` — frozen H2H result
- `screen/paired-inference.json` — 100,000-resample pair bootstrap
- `causal-parity.json` — exact online/offline policy parity
- `history-mode-comparison.json` — same-state mechanism comparison
- `causal-prior-decisions.jsonl` — causal decision dump
- `stateless-prior-decisions.jsonl` — legacy-stateless decision dump

Primary SHA-256 values:

- screen result: `a4127245dfb15913053d7347130d063f737bb75fd1cddced9a1c854c000414c1`
- paired inference: `8ae19e3bfae5593a1136ac98a51c548c4cfc0413f521c7ea22fdc27af781df6d`
- causal parity: `9fc507927c19372e0bffff6bca3bc773dcb8f14d1c1b3a7f0b4e4dcaee0fc5df`
- history comparison: `eab3c45788fe149f270bb7fd0f199b92dd197affe63a4807faa83937c5ca6fed`
