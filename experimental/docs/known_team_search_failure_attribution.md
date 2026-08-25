# Known-team search-failure attribution

## Research decision

The next experiment is research-backed, but an ordinary perfect-state learned
leaf evaluator is **not** authorized.  Metagross has already falsified that
variant on its deployment distribution.  The immediate experiment instead
attributes the 27 frozen Phase-1 headroom roots using a fixed information-set
particle panel, two independent seeds, a search-budget ladder, and two root
selectors.

This design follows five primary-source findings:

- [SAVE](https://arxiv.org/abs/1912.02807) shows that search-derived action
  values can be amortized and can help most at small budgets, but also reports
  that bad value estimates can make *more* search worse.  Therefore a budget
  ladder and an explicit no-model control come before training.
- [Expert Iteration](https://arxiv.org/abs/1705.08439) supports learning from a
  stronger search teacher, but does not make perfect-information labels valid
  in an imperfect-information deployment state.
- [ReBeL](https://arxiv.org/abs/2007.13544) places search and value learning in
  public belief states.  Therefore raw known-team states are diagnostic
  counterfactuals, not deployable model inputs.
- [Variance Reduction in MCTS](https://proceedings.neurips.cc/paper/2011/hash/d736bb10d83a904aefc1d6ce93dc54b8-Abstract.html)
  shows that finite-budget action selection can be variance-dominated.  Two
  independent seeds are required before calling a rescue real.
- [MAPLE](https://www.alphaxiv.org/abs/2605.24139) is newer preprint evidence
  that aggregating multiple possible states inside one search can mitigate
  strategy fusion.  It motivates the information-set aggregation constraint,
  but is not treated as settled evidence.

Exa was used to review 40 results across four search workstreams; alphaXiv was
used to inspect SAVE, Expert Iteration, ReBeL, and MAPLE at paper level.

## Frozen estimand

The cohort is exactly the 27 representative roots with stable Phase-1
known-team headroom of at least 0.02.  For each root:

1. Select 16 particles once by seeded systematic resampling from the Phase-2
   bank's alpha=0/current-belief weights.
2. Reuse the stored independent 20,000-iteration searches.
3. Run independent 5,000- and 80,000-iteration searches on the same particles.
4. Compare visit-count argmax with mean-Q-advantage argmax.
5. Score actions only with the frozen repeated known-team teacher.

## Attribution rules

- `particle_panel_or_current_search_resolved`: the fixed 16-particle panel at
  20,000 iterations already selects a teacher-beneficial action on both seeds.
  This is evidence that world-panel/aggregation details matter, not evidence for
  a larger search budget.
- `finite_search_budget`: both high-budget seeds agree on a teacher gain of at
  least 0.02 and the current-budget visit selector does not.
- `root_selector_or_visit_allocation`: both current- or high-budget
  mean-Q-advantage selectors agree on a gain of at least 0.02 where the
  corresponding visit selector does not qualify.  A current-budget selector
  rescue takes precedence over calling a later high-budget visit rescue a
  compute effect.
- `monte_carlo_seed_variance`: current-budget visit argmaxes disagree and neither
  rescue above qualifies.
- `unresolved_information_value_or_opponent_model`: none of the isolated
  mechanisms explains the miss.  This category is deliberately confounded; it
  does not prove a learned leaf is the cause.

No category authorizes public games.  A hidden-world learned leaf remains
forbidden.  If unresolved roots dominate, the next value model must consume the
player information state and posterior aggregate, or the project should prefer
the already-alive policy-prior/search-allocation track.
