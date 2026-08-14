# Resource-aware long-horizon expert: frozen development protocol

Date: 2026-08-14

The experiment tests whether an explicit, leak-free shadow value for conserved
Pokemon resources improves long-horizon search before any target collection or
student training. It is a development-first gate. The holdout panel, live H2H,
and distillation are blocked unless the development treatment independently
beats both the archived 500 ms action and equal-depth unmodified search.

## Treatment

The leaf value is a root-centered blend of the stock hand evaluation and a
23-term linear resource logit. The 21 resource coefficients are constrained
nonnegative. Active terms cover own team/active/bench HP, remaining switch
depth, Tera availability, PP reserve and move availability, opposing HP/faints,
status and Tera use, boosts, screens, hazards, Substitute, and own held-item
reserve. Battle progress and Trick Room are context terms.

Four opponent-information slots are reserved but fixed to zero in deployable
search. Determinized leaves contain sampled hidden opponent sets and do not
retain the causal public reveal mask; reading those fields would leak the
sampled world. Information option value is therefore explicitly outside this
v1 estimand rather than silently approximated with private state.

The model is fit only for calibration from actual terminal wins/losses. Its
predictive accuracy is not a promotion metric. Search regret on fixed roots is
the development metric.

## Data and splits

- Calibration corpus: 184,213 decision states from 4,998 self-play battles.
- Split: deterministic SHA-256 70/15/15 by battle group, with inverse
  decision-count battle weights.
- Root source: the previously frozen 1,000-battle public action-Q panel.
- Development: 50 unique source battles, split evenly between roots with and
  without legal Tera variants.
- Holdout: 50 different source battles with the same stratum balance.
- Development/holdout overlap: zero battles.
- Each root has two fixed schedules and eight fixed worlds per schedule.

Panel selection and the shadow model were frozen before teacher execution.

## Search and gate

- Reference: stock hand-evaluated MCTS, 50,000 exact iterations per world, with
  a seed namespace disjoint from both treatment arms.
- Equal-depth comparator: stock hand-evaluated MCTS, 20,000 iterations/world.
- Treatment: identical 20,000-iteration MCTS with only the resource leaf blend
  changed.
- Frozen initial blend: 0.25.
- One allowed development-only bounded calibration: 0.05, 0.10, and 0.15.
- Archived historical comparator: the action actually selected by the source
  500 ms controller. It is descriptive because it predates causal-history R1.

Promotion requires all of:

1. lower mean oracle regret than the archived historical action;
2. lower mean regret than equal-depth unmodified search;
3. a positive 95% source-battle bootstrap lower endpoint for the improvement
   over equal-depth search;
4. no added regret events of at least 0.10.

Failure blocks holdout execution, H2H, target collection, distillation, and any
GXE claim.
