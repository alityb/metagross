# R1 causal-history deployability gate

Date frozen: 2026-08-14

Execution: local CPU only

Checkpoint: frozen R1 epoch 5, SHA-256 `c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93`

## Question

Can production serve the transformer the same causal player-information
trajectory used by R1 training, without leaking hidden information, inferring
actions from outcomes, or silently reverting to a stateless policy?

This is a deployability gate, not a strength test. It authorizes later
resource-aware search/value work; it cannot establish a GXE improvement.

## Frozen transition contract

For discretionary decision `k`:

1. The private Showdown request `(battle, rqid, request_sha256)` defines the
   current observation and exact legal support.
2. `/priors` appends exactly one observation `O_k` and returns a monotone
   `decision_idx`.
3. After search/controller selection, but before the command may be returned
   to the Showdown sender, Foul Play posts the final canonical action to
   `/action` with the same request identity and `decision_idx`.
4. The server admits the action only if it was in the served support and maps
   to exactly one of R1's 13 action indices. An exact duplicate is idempotent;
   a conflict fails closed.
5. At the next discretionary request, the server computes
   `R_{k+1}=reward(S_k,S_{k+1})` and supplies reward-first RL2 input
   `[R_{k+1}, one_hot(A_k)]` beside `O_{k+1}`.

Public `|move|`, `|switch|`, failure, Tera, and reveal events update the
information state but are never action labels. Sole automatic Recharge or
Struggle requests do not add a policy observation. Missing acknowledgements,
reward failures, non-finite rewards, and any length mismatch fail the game;
there is no silent history reset.

## Required local checks

- Unit tests cover exact/idempotent/conflicting/stale acknowledgement behavior,
  forced requests, reward-first sequence construction, masks, truncation, and
  isolated per-episode observation spaces.
- Saved real Foul Play protocol artifacts must show 100% mapping of every
  correlated outbound learned choice to one private-request action and one R1
  index. Public action events must be counted only as ignored observations.
- A new live canary must subsequently show zero missing/conflicting action
  receipts and monotonically growing trajectory length through all ordinary
  decisions. This canary is required before any full-history model training.
- Online-versus-replay tensors (`text_tokens`, numeric features after
  `nan_to_num`, legality mask, RL2, and time indices) must be exact. Frozen R1
  probabilities must have maximum absolute difference `<=1e-7`.

## Admission decision

Full-history terminal-value or search-distillation training remains blocked
until every required check passes. Even on a pass, a learned action critic is
not automatically authorized: both Metagross's stateless action-Q experiment
and MegaGem's action-critic pilot found weak within-state action signal. The
next learning target must first be supported by an independently winning,
long-horizon search expert. Only confident expert deviations are distilled,
with pass-through policy rows and resource-preservation anchors.

## External methodological input

Alex Wa's MegaGem study is used as a design warning, not as Pokémon evidence:
its myopic offline selector lost live until liquidity's option value was
priced; the paced expert then improved live margin and only afterward was
distilled. Pokémon analogues of conserved option value include HP, Tera, PP,
tempo, revealed information, and remaining switch resources. See
[`Learning MegaGem, from self-play to price discovery`](https://github.com/djdumpling/djdumpling.github.io/blob/main/_posts/2026-08-09-megagem.md)
and the [implementation](https://github.com/djdumpling/MegagemBench).
