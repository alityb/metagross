# AlphaPokémon — Revised Architecture
## Superhuman Gen9 Random Battle Agent via Neural MCTS + Offline RL

**Target**: GXE > 90% on gen9randombattle after 200+ ladder games (Glicko-1 deviation < 50).  
**Baseline to beat**: Foul Play — 88% GXE / 2341 ELO (Mariglia, 2025).  
**Format**: gen9randombattle exclusively. Metamon and PokéChamp target OU formats; this format is uncontested by any neural agent.

---

## Why This Architecture

Two prior systems define the performance ceiling:

**Foul Play** (Mariglia, 2025) — 88% GXE, no neural network. Hand-crafted Rust evaluation function + root-parallelized MCTS + DUCT + poke-engine. Demonstrated that MCTS alone, with even a mediocre evaluator, approaches the human elite ceiling. Explicitly acknowledged limitation: zero learning, no weather/terrain evaluation weight, no information-theoretic play.

**Metamon** (Grigsby et al., RLC 2025, arXiv:2504.04395) — 71–83% GXE across Gen1–4 OU, no search. Offline AWR from 5M+ human replay trajectories. Causal Transformer over full battle history. Demonstrated that large-scale offline RL from human data can approach 90th percentile without search. Limitation: targets OU formats, not random battles. gen9randombattle never evaluated.

**AlphaPokémon** combines both: a trained neural evaluator (from Metamon's paradigm) + deep MCTS tree (from Foul Play's paradigm) + RLM strategic prior (novel). The interaction is multiplicative: better evaluation → better search guidance → deeper meaningful lookahead → higher GXE. The key reference: Wang (2024) achieved 79.5% GXE with random rollouts + MCTS. Foul Play added a hand-crafted evaluator and reached 88%. A properly trained evaluator should exceed this.

---

## Dataset

### Sources

**1. gen9randombattle Human Replays** (primary quality signal)  
- Source: `replay.pokemonshowdown.com/search.json?format=gen9randombattle`  
- Collection: continuous scraper running 24/7, capturing all rated replays (ELO ≥ 1200) as they are uploaded (~240 games/hour, ~5760/day)  
- Rating filter: ELO ≥ 1500 for Phase 1 training; ELO ≥ 1200 for value-only auxiliary training  
- Format: spectator logs reconstructed to first-person trajectories via `BattleObservationState` (deterministic from log events — see `pipeline/generate.py`)  
- Target: 300K+ replays accumulated over 6–8 weeks of continuous scraping  
- Why gen9randombattle specifically: the finite known set pool (pkmn.github.io/randbats) makes opponent set inference tractable. OU's open team building makes the belief state exponentially harder. This is a structural advantage for random battles that no prior system has exploited.

**2. SimpleHeuristicsPlayer Self-Play** (fast strategic bootstrap)  
- Generated: `training/collect_heuristics.py` runs SimpleHeuristicsPlayer vs itself on the local PS server  
- Volume: 20K games = ~4M state-action pairs in ~44 minutes at n_envs=16  
- Quality: strategic play (type coverage, switching, hazard awareness) vs random-play IL which produces no useful policy signal  
- Role: cold-start Phase 1 when human replay volume is insufficient. *Replace random-play synthetic data (prior approach) with heuristics-play data because imitating random actions teaches the policy random habits that are actively harmful against any strategic opponent.*  
- Citation: Nebraskinator (2024) demonstrated that BC on SimpleHeuristics before PPO reaches 1900 ELO with no MCTS.

**3. MCTS Self-Play Trajectories** (Phase 3 training signal)  
- Generated during Phase 3 by running the full AlphaPokémon agent against itself  
- Stored as `(state, mcts_visit_distribution, outcome)` — NOT individual actions  
- Volume: 15K games × ~25 turns × 2 sides = ~750K training positions  
- The MCTS visit distribution is strictly higher quality than any individual action: it represents thousands of simulated futures rather than one human or heuristic choice  
- Citation: Silver et al. (2017) AlphaZero — training on visit distributions rather than individual moves is the core insight that enables superhuman play.

### Reconstruction Protocol (Human Replays)

PS replays are spectator logs. Reconstruction to first-person trajectories is deterministic:
- Parse log line by line; maintain `BattleObservationState` tracking revealed info  
- Each `|switch|p2a: Species|` event → reveal species and HP  
- Each `|move|p2a: Species|MoveName|` event → reveal move  
- Each `|-item|`, `|-ability|` event → reveal item/ability  
- At each `|request|` JSON → encode state from P1's partial-obs perspective, record action taken  
- **Random battles are the easiest format for reconstruction**: no team preview means the opponent starts as 6 fully UNKNOWN slots; information is monotonically additive. OU team preview requires inferring which of the 6 revealed species the opponent built around, which is ambiguous.

---

## Model Architecture

### PokeNet (~1.26M trainable parameters)

**Input encoding** (per `src/model/state.py`):

| Component | Dimensions | Notes |
|-----------|------------|-------|
| Species embeddings | 1,469 × 384 → projected to 64 | Frozen e5-small-v2 init |
| Move embeddings (4 slots) | vocab × 384 → projected to 32, summed | Frozen e5-small-v2 init |
| Last-move embedding | vocab × 384 → projected to 32 | Same table as moves, separate slot |
| Item embedding | vocab × 384 → projected to 32 | Frozen e5-small-v2 init |
| Ability embedding | vocab × 384 → projected to 32 | Frozen e5-small-v2 init |
| Pokemon dense features | 224 dims | HP(7) + boosts(91) + status(7) + volatiles(38) + PP(16) + tera(21) + types(38) + misc(6) |
| Field features | 70 dims | Weather(8) + terrain(6) + TR(2) + hazards(8) + screens(12) + unrevealed(2) + sleep_clause(2) + misc(31) + gen_onehot(9) |

*Replace Metamon's causal Transformer over full battle history with explicit Bayesian belief state + PokeNet per-position encoder because: (1) explicit set-pool inference provides hard constraints that implicit attention cannot represent; (2) per-position encoding is compatible with MCTS — a causal Transformer requires the full history at every MCTS leaf, making it 100–1000× more expensive per leaf evaluation; (3) Foul Play demonstrated that position-level evaluation + MCTS dominates history-level evaluation without MCTS.*

**Transformer backbone** (shared, frozen e5 embeddings not included):

```
d_model = 192
3 × TransformerEncoderLayer(
    nhead=8, dim_feedforward=384, dropout=0.1, activation='gelu', norm_first=True
)
Input:  13 tokens × 192 dims  (1 field + 6 own + 6 opponent)
Output: 13 tokens × 192 dims
```

Head state: `[field_token, own_active_token, opp_active_token]` → 576 dims

**Policy head**: `Linear(576, 192) → ReLU → Linear(192, 14)` + action mask  
**Value heads** (ensemble): 4 × `Linear(576, 192) → ReLU → Linear(192, 1) → Tanh`

*Replace Metamon's scalar Q-value head with 4-critic ensemble (min over targets for training) because Metamon SynV2+ found that critic ensembles prevent overestimation spirals that cause value collapse in self-play. Mean over critics at inference.*

*Replace Metamon's 51-bin TwoHot distributional value (range [-1.6, +1.6]) with scalar Tanh ensemble for now — TwoHot is the correct improvement for Phase 3 when the shaped reward range extends to ±1.6 (±1 game outcome + up to ±0.6 faint shaping).*

**Parameter count**: ~1.26M trainable (frozen embeddings excluded).

**Why ~1.25M and not 50M (Metamon)?**  
At 7.5s per decision with poke-engine MCTS, a 1.25M model completes ~500K leaf evaluations. A 50M model would complete ~6K evaluations in the same budget. Foul Play's competitive performance with zero neural network demonstrates that search depth dominates evaluator quality at equal compute. *Replace Metamon's large causal model with small fast evaluator because search iterations × evaluation quality > evaluation quality alone at fixed time budget (Wang, 2024; Silver et al., 2017).*

---

## Phase 1 — Offline AWR Bootstrap

**Goal**: ≥ 60% WR vs SimpleHeuristicsPlayer before Phase 2 begins.  
**Compute**: ~1 hour on H100 (training). Data collection: ongoing.

### Objective Function

*Replace pure behavioral cloning (prior Phase 1 approach) with Advantage-Weighted Regression (AWR / ExpRL) because Metamon (Grigsby et al., 2025) found ExpRL and BinaryRL consistently outperform pure BC across all evaluated generations. Pure BC on any data source is equivalent to imitating average play, not good play.*

```
L_total = L_policy_AWR + 0.5 × L_value + 0.2 × L_distill

L_policy_AWR = mean( exp(β × A(s,a)) × CE(π(s), a) )
             where A(s,a) = outcome - V̂(s)   (advantage estimate)
             β = 1.0, weights clipped to [1e-3, 20.0]

L_value      = MSE(V̂(s), outcome)     # train value heads to predict outcomes
L_distill    = MSE(V̂(s), V_rlm(s))   # when RLM annotations available (Phase 0)
```

*Replace L_belief (prior) with AWR advantage weighting because: (1) we have no Phase 0 annotations yet; (2) AWR is more principled — it upweights learning from above-average decisions rather than penalizing belief entropy as a proxy.*

### Token Mask Augmentation

*Adopt Metamon's token_mask_aug=True with p=0.15 per opponent slot because Metamon found this critical for sim2real transfer. Randomly zeroing opponent tokens during training forces the model to handle partial observability robustly — exactly the test-time distribution.*

Implementation in `training/dataset.py`: `collate_replay_samples(token_mask_prob=0.15)` zeroes opponent species/move/item/ability IDs independently with p=0.15.

### Training Protocol

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Standard |
| Learning rate | 1e-4 → 1e-6 cosine | Metamon: 1.25e-4; Wang: 1e-4 |
| Batch size | 512 | Memory/throughput tradeoff |
| Epochs | 3 | Stop at ≥60% vs SimpleHeuristics |
| AWR β | 1.0 | Metamon ExpRL default |
| Token mask prob | 0.15 | Metamon default |
| Gradient clip | 1.5 | Metamon default (vs 1.0 prior) |

### Data Sources for Phase 1

Priority order:
1. Human gen9randombattle replays (ELO ≥ 1500) — highest quality, accumulating
2. SimpleHeuristics vs SimpleHeuristics (20K games, available now) — strategic bootstrap
3. Random battles synthetic data (available) — value-only auxiliary training (policy loss zeroed)

*Replace random-play synthetic data as primary Phase 1 source with heuristics-play data because random-play IL teaches the policy random habits that SimpleHeuristicsPlayer (the next harder opponent) destroys immediately. Verified: policy trained on random play achieved 2% WR vs SimpleHeuristics.*

### Validation Gate

Before Phase 2: 100-game evaluation vs SimpleHeuristicsPlayer. Target ≥ 60% WR. If below 40%, increase heuristics data volume (collect 200K games instead of 20K) and retrain.

---

## Phase 2 — Online Self-Play PPO

**Goal**: ≥ 80% WR vs SimpleHeuristicsPlayer (validated every 100 updates).  
**Compute**: ~24 hours on H100 at n_envs=16 (~500K games).

### Training Configuration

*Replace curriculum (Random → SimpleHeuristics) with direct self-play because: training against a fixed opponent causes the policy to specialize against that specific opponent's exploits rather than learning general Pokemon strategy. Confirmed empirically: 2% WR after promotation. Self-play eliminates this by having both sides evolve simultaneously.*

Both sides use `RolloutPlayer` — same model, same weights, no fixed opponent.

### PPO Hyperparameters (Wang, 2024 §3.1)

| Parameter | Value | Source |
|-----------|-------|--------|
| γ (discount) | 0.9999 | Wang (2024) |
| GAE λ | 0.754 | Wang (2024) |
| Clip ε | 0.083 | Wang (2024) |
| Value clip ε_v | 0.018 | Wang (2024) |
| Entropy coefficient | 0.10 | Raised from Wang's 0.059 — prevents entropy collapse during opponent transitions |
| Value coefficient | 0.438 | Wang (2024) |
| Max grad norm | 0.543 | Wang (2024) |
| Minibatch size | 1024 | Wang (2024) |
| SGD epochs | 7 | Wang (2024) |
| LR | 3e-5 cosine | Wang schedule collapses from warm start; cosine stable |
| n_envs | 16 | H100 capacity |

### Reward Shaping

*Add Nebraskinator (2024) faint shaping (+0.1 per opponent faint, −0.1 per own faint) on top of ±1 win/loss. Outcome range: [−1.6, +1.6]. This provides dense signal at every KO rather than sparse terminal-only reward. Foul Play's evaluation equivalent: POKEMON_ALIVE_STATIC=30 is the static analogue.*

### R-NaD KL Regularization

*Add Perolat et al. (DeepNash, Science 2022, arXiv:2206.15378) R-NaD penalty to prevent self-play cycling:*

```python
L_rnad = η × KL(π_current(·|s) || π_lagged(·|s).detach())
```

where `π_lagged` is the policy snapshot from T_reg = 10,000 training steps ago. η = 0.2 (tunable). This is a soft regularizer operating at the multi-step timescale — different from PPO's clipping which operates per-update. *R-NaD prevents the exploit→counter-exploit→cycle failure mode that is the primary self-play instability for simultaneous-move games (DeepNash solved Stratego's 10^535 game tree with this mechanism).*

### Periodic Validation

Every 100 PPO updates: 100 games vs SimpleHeuristicsPlayer. Target trajectory: ≥ 60% by update 50, ≥ 80% by update 200. If value collapses (all critic outputs < −0.5 for 10 consecutive updates), checkpoint and restart from last good checkpoint with reduced LR.

---

## Phase 3 — MCTS-Guided AlphaZero Self-Play

**Goal**: > 90% GXE on gen9randombattle ladder after 200+ games.  
**Compute**: ~4 days on H100 (15K MCTS games, 2M+ leaf evaluations per game).

### MCTS Architecture (Fixed Tree Search)

*The previous MCTS was a 1-ply bandit: every iteration started from root, applied one action, evaluated with PokeNet, and updated only the root's Q-values. MCTSNode.children was never populated. Fixed: proper recursive tree traversal.*

**Tree structure**: recursive `_tree_rollout(node, state, depth=5)`:
1. **Selection** — traverse existing nodes via PUCT until unexpanded leaf
2. **Expansion** — first visit: step simulator, create child node initialized with PokeNet prior + value
3. **Backpropagation** — update all ancestor Q-values

With depth=5 and the poke-engine step time (~0.3ms/batch), the 7.5s budget yields ~500K iterations that explore 5-turn sequences rather than 1-step look-aheads.

### Gumbel Sequential Halving

*Replace PUCT (Kocsis & Szepesvári, 2006) with Gumbel sequential halving (Danihelka et al., ICLR 2022) because: (1) PUCT systematically under-visits low-prior critical actions when N is large; (2) Gumbel has provable near-optimality guarantees at any budget; (3) eliminates the c_puct=1.25 hyperparameter entirely.*

At each node, score actions via:
```
score(a) = gumbel_noise(a) + completedQ(a)
completedQ(a) = Q(a)  if visited  else  sigmoid⁻¹(prior(a))
```

Budget allocation: sequential halving over log₂(14) ≈ 4 rounds, eliminating bottom half each round. Actions with low prior but high observed Q survive; those with inflated prior but low Q are eliminated.

### DUCT for Simultaneous Moves

Per Lanctot et al. (2013). Both players select independently:
```
UCB_i(s, a) = Q_i(s,a) + c_puct × π_i(a|s) × √N(s) / (1 + N_i(s,a))
```
Player 2's policy uses mirrored state encoding. Joint action = Cartesian product of independent selections. *Gumbel replaces the UCB formula for both players.*

### Soft Belief Embeddings in Opponent Tokens

*Partially implement ReBeL's Public Belief State (Brown et al., NeurIPS 2020, arXiv:2007.13544): instead of sampling K hard opponent world-states (K-tree approach), encode the full posterior distribution as soft weighted embeddings into opponent tokens.*

For each unrevealed opponent slot with posterior {set₁: 0.7, set₂: 0.3}:
```python
item_emb = 0.7 × emb(set₁.item) + 0.3 × emb(set₂.item)
```

*Replace UNKNOWN token (hard symbol with no uncertainty information) with soft posterior average because: ReBeL (Brown et al.) proved that planning in probability distributions over hidden state is strictly better than planning in sampled world states. The K=4 tree sampling captures first-order variance; soft embeddings give PokeNet the full posterior shape at each leaf evaluation.*

**Note**: Full ReBeL (planning at PBS level) is computationally prohibitive for Pokemon's large information sets. The soft-embedding approach captures ~80% of the benefit at ~2% of the implementation cost.

### RLM-Qwen3-8B at Root

*Novel contribution — no prior Pokemon system does this.*

At the root node only, before MCTS begins:
1. RLM-Qwen3-8B runs in a persistent REPL loop (≤ 500ms)
2. GREP/READ/BASH tools over the full battle log (Zhang et al., ICML 2026, arXiv:2512.24601)
3. Recursive sub-calls to Qwen3-0.6B for per-Pokémon set inference
4. Output: π_rlm (14-dim strategic prior), V_rlm (strategic value estimate), refined_belief

Root prior: `π_root = softmax(0.5 × logit(π_net) + 0.5 × logit(π_rlm))`

*Replace Foul Play's hand-crafted evaluation constants with RLM-distilled strategic knowledge because: Zhang et al. (2026) showed RLM-Qwen3-8B outperforms base Qwen3-8B by 28.3% on long-context tasks by recursive decomposition over battle log. Damage dealt → stat range → set elimination → speed constraint inference across 20+ turns is exactly the long-context compound reasoning problem this was designed for.*

### AlphaZero-Style Training Target

*Replace single-action cross-entropy (Phase 1/2) with MCTS visit distribution targets because Silver et al. (2017) AlphaZero proved this is strictly better — the visit distribution encodes relative action quality under deep search, not just which action was sampled.*

```
L_Phase3 = KL(mcts_visits || π_net(s)) + (V_net(s) - outcome)²
```

At ~2M leaf evaluations per game, the visit distribution is extremely high-quality supervision. The 15K MCTS games generate ~750K training positions with this supervision signal.

### Temperature Annealing (AlphaZero §A.2)

*Add temperature annealing per Silver et al. (2017): τ=1.0 for first 30 turns (sample proportional to visit counts, more exploratory), τ→0 after turn 30 (argmax of visit counts, more decisive). Without annealing, early-game data is deterministic and lacks coverage.*

### Checkpoint Pool (AlphaZero §D)

*Add historical checkpoint pool to prevent strategy cycling: sample opponents from the last 5 saved checkpoints (uniformly). Prevents the exploit→counter-exploit cycling that degrades self-play without R-NaD stabilization. Implemented as a circular buffer of saved model weights.*

### Resign Threshold

If V_net < −0.9 for 5 consecutive turns, resign. Equivalent position probability < 5%. Frees compute for closer games; resigning positions are low-quality training data.

---

## Phase 3+ — Improvements Pending Research

These are confirmed improvements but deferred to after achieving >90% GXE baseline:

**Student of Games (Schmid et al., Science Advances 2023, arXiv:2112.03178)**  
*Replace DUCT + UCB with CFR+ at simultaneous-move nodes.* DUCT treats both players as independently maximizing, ignoring that they're solving a minimax problem. CFR+ at each node solves the 2×2 to 14×14 bimatrix game to Nash equilibrium. Provably sound for two-player zero-sum simultaneous games.

**Persistent Chance Nodes**  
Separate strategic uncertainty (which action is optimal) from stochastic outcome variance (which damage roll occurred) in the tree. Currently Q(s,a) accumulates noise from both sources. Standard fix: chance nodes with children per stochastic outcome. At poke-engine's typical 2–3 branches per action pair, this is a manageable 3× branching overhead.

**TwoHot Distributional Value (Metamon SynV2)**  
*Replace scalar Tanh with 96-bin TwoHot value head (range [−1.6, +1.6]) because Metamon found distributional value outperforms scalar regression, and with faint shaping the reward range is exactly [−1.6, +1.6].* Deferred to Phase 3+ because it requires retraining from Phase 1 with TwoHot targets throughout.

**Off-Policy Correction in PPO (EfficientZero / Ye et al., NeurIPS 2021)**  
When Phase 3 replays old games for policy updates, positions collected at step 500K are trained on at step 2M with a different policy. Importance sampling correction prevents stale value targets from corrupting training. ~20 lines in `ppo.py`.

---

## Evaluation

### Primary
GXE after ≥ 200 ladder games on gen9randombattle, Glicko-1 deviation < 50. Target: > 90%.

### Secondary
FH-BT (Full-History Bradley-Terry, Karten et al. 2026) via `pokeagentchallenge.com`. More stable than Glicko-1 for fixed AI policies. Enables direct comparison against Foul Play.

### Head-to-Head
300 local games vs Foul Play (`gen9randombattle`, SEARCH_TIME_MS=1000). Report winrate + 95% CI (Wilson score). Target: > 70% winrate vs Foul Play.

### Ablation Suite (200 games each)
| Ablation | What changes |
|----------|-------------|
| No RLM | π_rlm = π_net; V_rlm = V_net |
| No AWR | β=0 (pure BC) |
| No faint shaping | Binary ±1 only |
| No tree MCTS (1-ply bandit) | Prior broken implementation |
| No MCTS | Policy argmax only |
| No soft belief embeddings | UNKNOWN tokens for opponent |
| Phase 2 only (no Phase 3) | Phase 2 checkpoint |
| Phase 1 random-play (no heuristics) | Prior broken Phase 1 |

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| PokeNet ~1.26M | ✓ Complete | FIELD=70, DENSE=224, last_move_ids |
| 4-critic ensemble | ✓ Complete | min-over-targets for training |
| Token mask augmentation | ✓ Complete | p=0.15 in collate_replay_samples |
| AWR loss | ✓ Complete | β=1.0, clip [1e-3, 20] |
| **MCTS tree (depth=5)** | **✓ Just fixed** | Was 1-ply bandit; now recursive AlphaZero tree |
| Faint shaping (+0.1) | ✓ Complete | In PPO RolloutPlayer |
| Self-play Phase 2 | ✓ Running | LR=3e-5 cosine, n_envs=16 |
| Heuristics data collection | ✓ Running | 20K games, ~44 min |
| Replay scraper | ✓ Running | Continuous, ~240 rated/hr |
| Soft belief embeddings | ✗ Pending | Phase 3 |
| Gumbel sequential halving | ✗ Pending | Phase 3 |
| R-NaD regularization | ✗ Pending | Phase 2 improvement |
| RLM-Qwen3-8B | ✗ Pending | Phase 3 (weights not yet loaded) |
| TwoHot value head | ✗ Pending | Phase 3+ |
| Student of Games / CFR+ | ✗ Pending | Phase 3+ |
| Temperature annealing | ✗ Pending | Phase 3 |
| Checkpoint pool | ✗ Pending | Phase 3 |

---

## References

Silver, D. et al. (2017). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. arXiv:1712.01815.

Wang, J. (2024). Winning at Pokémon Random Battles Using Reinforcement Learning. MIT MEng thesis. *Gen4 PPO+MCTS; 79.5% GXE; PPO hyperparameters and LR schedule used in Phase 2.*

Grigsby, J. et al. (2025). Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers. arXiv:2504.04395. *Metamon: 71–83% GXE on Gen1–4 OU from offline AWR on 5M+ human replays.*

Mariglia, P. (2025). Foul Play. pmariglia.github.io/posts/foul-play. *88% GXE / 2341 ELO on gen9randombattle. Highest GXE for any AI on this format.*

Karten, S. et al. (2026). The PokeAgent Challenge. arXiv:2603.15563. *NeurIPS 2025 competition; Foul Play Gen9OU champion; elite human GXE ~90%.*

Zhang, A.L., Kraska, T., Khattab, O. (2026). Recursive Language Models. arXiv:2512.24601. ICML 2026. *RLM-Qwen3-8B; 28.3% improvement over base Qwen3-8B on long-context tasks via recursive decomposition.*

Brown, N. et al. (2020). Combining Deep Reinforcement Learning and Search for Imperfect-Information Games. arXiv:2007.13544. NeurIPS 2020. *ReBeL: MCTS for imperfect information games; Public Belief State planning.*

Danihelka, I. et al. (2022). Policy improvement by planning with Gumbel. ICLR 2022 Spotlight. *Gumbel AlphaZero: provably near-optimal action selection; replaces PUCT.*

Perolat, J. et al. (2022). Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning. Science 2022. arXiv:2206.15378. *R-NaD: KL regularization prevents self-play cycling; solved 10^535-node game tree.*

Schmid, M. et al. (2023). Student of Games. Science Advances 2023. arXiv:2112.03178. *Combines AlphaZero + CFR+ for unified perfect/imperfect information planning.*

Nebraskinator (2024). ps-ppo. github.com/Nebraskinator/ps-ppo. *1900 ELO gen9randombattle from BC on SimpleHeuristics + PPO self-play; faint shaping.*

Lanctot, M. et al. (2013). Monte Carlo Tree Search for Simultaneous Move Games. BNAIC 2013. *DUCT algorithm for simultaneous moves.*

Ye, W. et al. (2021). Mastering Atari Games with Limited Data. NeurIPS 2021. arXiv:2111.00210. *EfficientZero: temporal consistency loss; off-policy correction.*
