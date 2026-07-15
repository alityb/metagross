# Gen9 MCTS Policy Distillation

`src/scripts/build_mcts_policy_sidecar.py` converts Foul Play decision logs into
optional Metamon policy targets. It maps normal moves, `-tera` moves, and
`switch <species>` using Metamon's canonical 13-action indexing and rejects any
unknown or illegal action. Visit masses are normalized only after all actions
validate.

The builder is deliberately fail-closed. Legacy decision records contain a
battle tag and username but no stable per-decision replay ID. It writes targets
only when exactly one learner POV exists and the full ordered log has the same
length as the parsed action sequence, with every logged selected action matching
the parsed action at its position. Any missing row, duplicate POV, length
mismatch, selected-action mismatch, or invalid visit action rejects the whole
trajectory. It does not infer labels from turn number, action names, or state
similarity. This means incomplete historical logs may yield zero labels; future
collection needs an explicit replay-action identity to relax this constraint.

New PFSP shards write schema-v2 decision records with `battle_tag`,
`learner_pov`, `mcts_decision_seq`, `canonical_selected_action_index`, and a
normalized `mcts_visit_target_13`. Build the paired parser identity artifact
before sidecar construction:

```bash
python src/scripts/build_mcts_trajectory_index.py \
  --parsed-root data/pfsp_learner_only \
  --output data/pfsp_trajectory_identity.jsonl
python src/scripts/build_mcts_policy_sidecar.py \
  --parsed-root data/pfsp_learner_only \
  --trajectory-index data/pfsp_trajectory_identity.jsonl \
  --decision-log data/pfsp/agent_a_decisions.jsonl \
  --output data/mcts_policy_targets.jsonl
```

Schema-v2 labels are still rejected individually when the canonical target or
selected action is illegal in the parsed state; stable identity never permits
an illegal target.

## Nested PFSP Finalization

For a completed high-budget schema-v2 PFSP collection, use the aggregate
finalizer rather than running the flat-directory scripts manually. It discovers
every nested shard containing `replays/*.json` and `agent_a_decisions.jsonl`,
validates each with the strict-shard protocol, mirrors shard paths below both
output roots, and builds one identity index and verified sidecar.

```bash
python src/scripts/finalize_schema_v2_pfsp.py \
  --raw-root data/pfsp_mcts_high_budget_raw \
  --parsed-root data/pfsp_mcts_high_budget_parsed \
  --learner-only-root data/pfsp_mcts_high_budget_learner_only \
  --trajectory-index data/pfsp_mcts_high_budget_trajectory_identity.jsonl \
  --sidecar data/pfsp_mcts_high_budget_targets.jsonl \
  --pool-path data/randbats_pools/gen9randombattle_pool_50000.json \
  --report data/pfsp_mcts_high_budget_finalization.json
```

The finalizer is post-collection only and parses sequentially in-process. It is
resumable: complete parsed outputs and matching learner links are reused. It
never overwrites partial parsed POV pairs; those, duplicate learner identities,
strict-shard failures, rejected/invalid sidecar rows, or zero verified targets
produce a nonzero exit after writing the JSON report. The parsed and
learner-only roots must be dedicated to this collection: stale trajectories are
reported as identity mismatches rather than silently included.

Enable the auxiliary loss explicitly:

```bash
python src/scripts/run_finetune_variant.py --variant base \
  --mcts-policy-sidecar data/mcts_policy_targets.jsonl \
  --mcts-policy-coeff 0.1 ...
```

Without both flags the existing offline advantage-weighted actor-critic and BC
losses are unchanged. This is an auxiliary masked cross-entropy distillation
objective over legal, non-padded action states, not a full AlphaZero
implementation.

## Modal H100 Pilot

`modal_train_mcts_distillation.py` is the dedicated, fixed-treatment launcher
for the high-budget schema-v2 pilot. It uses R1 epoch 5 with `A_rating`, one
epoch of 1,000 steps, 90% strict learner self-play and 10% human anchor data,
and MCTS policy coefficient 0.1. It does not enable a KL anchor.

Its local entrypoint packages the finalizer's nested learner-only root directly:
it preserves each nested trajectory under `gen9randombattle/` and rewrites only
the sidecar's relative trajectory keys to that archive layout. It rejects
missing, duplicate, or extra sidecar trajectory paths locally. The Modal
function then validates every target timestep and distribution against the
compressed trajectories before invoking the training subprocess.

The default R1 root is the local
`src/nets/checkpoints/randbats_full/randbats_exit_r1` directory. Its packaged
run name is `randbats_exit_r1`, using the nested
`randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt` checkpoint rather
than the duplicate compatibility copy at the root.

```bash
modal run src/scripts/modal_train_mcts_distillation.py \
  --learner-only-root data/pfsp_mcts_high_budget_learner_only \
  --sidecar data/pfsp_mcts_high_budget_targets.jsonl \
  --human-anchor-root data/parsed_replays
```

Pass `--r1-checkpoint-root <path>` only to override the default R1 root.

References: Silver et al., *Mastering Chess and Shogi by Self-Play with a
General Reinforcement Learning Algorithm* (2017),
https://arxiv.org/abs/1712.01815, motivates learning from search visit policies.
Kostrikov et al., *Offline Reinforcement Learning with Implicit Q-Learning*
(2021), https://arxiv.org/abs/2110.06169, motivates retaining a behavior
constraint when learning offline rather than replacing it with unconstrained
policy improvement.
