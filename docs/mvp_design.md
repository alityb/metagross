# MVP Design

## MVP Name

**Streaming Speculative Search for Gen9 Foul Play**

## Goal

Determine whether opponent think-time and speculative caching can improve Foul Play’s effective search budget in `gen9randombattle` without changing the final decision distribution unsafely.

This is a measurement-first MVP. If idle-time/cache-hit measurements fail, do not implement the full search cache.

## Why This MVP

Supported facts:

- Stock Foul Play is the strongest implemented gen9 baseline in this repository.
- Learned eval, randbats belief variants, Tauros gates, standalone distillation, and value shield did not beat stock Foul Play.
- The value shield barely activated because final-selection vetoes are too weak as an intervention.

Engineering judgment:

- The best remaining mechanism is not another final selector. It is better compute allocation.
- Speculative search uses idle time while waiting for the opponent and can be rejected safely when the observed state does not match.

## Architecture

```text
Showdown foreground loop
  receive request
  check speculative cache
  if valid cache hit: use cached policy/value as prior or answer if still safe
  else run normal Foul Play search
  send /choose
  notify background worker of chosen action and public state

Background speculative worker
  receive chosen action + public state
  predict top opponent actions
  sample hidden worlds
  simulate likely successors or prepare likely determinization bundles
  run small search on candidate next states
  store cache entries keyed by canonical public state hash

Verifier
  on next request, canonicalize observed public state
  accept cache only on exact-enough match
  reject stale branches
```

## Required For MVP

- Timestamp instrumentation around `/choose` and next `|request|`.
- Measurement of opponent think-time distribution in local H2H and live ladder logs where available.
- Canonical public-state hash for Foul Play battle state.
- Background worker interface with cancellation on new observations.
- Safe cache acceptance rule.
- Fallback to stock Foul Play on cache miss.
- N=100 H2H gate versus stock Foul Play at equal wall-clock budget.

## Strongly Recommended

- Log cache hit rate, rejected cache rate, background compute time, and next-turn latency.
- Start with speculation over opponent top-k actions from cheap Foul Play draft search.
- Keep all final decisions stock-equivalent unless cache hit is exact and fresh.
- Measure local H2H first; ladder only after H2H gate passes.

## Nice To Have

- Tauros/Metamon as draft policy for Gen1 experiments only.
- Randbats generator posterior as draft hidden-world sampler.
- Adaptive speculation budget based on timer and state volatility.

## Future Research

- Full public-belief speculative search.
- CFR/regret re-solving on accepted speculative states.
- Learned opponent response model.
- Batched neural value/policy priors.

## Alternatives Considered

- More Tauros distillation: rejected for MVP because naive versions failed badly and require much more data/model work.
- More randbats belief sampling: rejected because N=100 static pool was neutral.
- Value veto shield: already tested and killed.
- More search budget only: likely helps but is not novel and may not transfer to public ladder timer constraints.

## Success Criteria

Measurement gate:

- Median usable opponent think-time is at least 500ms in the target environment, or background speculation still yields measurable cache hits under bot H2H.
- Cache hit rate is at least 20-30% on nontrivial turns.

H2H gate:

- `spec_search_fp` versus stock `foul_play`, N=100 paired, equal wall-clock budget.
- Continue only if point estimate is at least 55% or latency/iteration gains are large and explain a neutral win rate.
- Promote only after N>=1000 with CI lower bound above 50%.
