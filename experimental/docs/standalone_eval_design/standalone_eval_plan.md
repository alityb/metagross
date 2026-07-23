# Standalone Handwritten Eval Plan

## Goal

Build an original evaluation function for `gen9randombattle` that can be honestly presented as Metagross's core contribution:

> an original belief-aware handwritten eval and exact randbats posterior tracker, built on credited open-source Foul Play / poke-engine search infrastructure.

The first target is to match or exceed the current Step 1 / Step 4 patched FP results. The long-term target remains live ladder GXE above the true FP baseline.

## Design principles

- Preserve FP's fast search and state inference.
- Replace stock `genx/evaluate.rs` with a new feature-based eval.
- Every feature group is side-symmetric and separately traceable.
- All weights are tunable.
- Belief-aware terms are bounded; they bias, not veto.
- Final promotion requires paired H2H and ladder GXE.

## Feature groups

### 1. Material and HP

Replace flat HP/alive scoring with:

- level-scaled HP value;
- role-aware alive value;
- diminishing returns for low-impact low-HP Pokemon;
- last-Pokemon / no-switch endgame modifier.

### 2. Move pressure and damage race

For active vs active:

- best immediate damage;
- 2HKO / OHKO thresholds;
- revenge-kill vulnerability;
- priority move pressure;
- accuracy-adjusted expected damage;
- damage-roll threshold volatility.

### 3. Speed control

Evaluate effective speed, not only speed boosts:

- paralysis;
- Tailwind;
- Trick Room;
- Choice Scarf;
- Swift Swim / Chlorophyll / Surge Surfer / Unburden / Quick Feet;
- priority / anti-priority interactions.

### 4. Status and residual clocks

Improve status scoring:

- Toxic clock based on turn counter;
- burn as physical damage suppression and residual clock;
- sleep turns and Rest turns;
- Leech Seed, Salt Cure-like volatiles, trapping, Encore/Taunt/Disable if exposed.

### 5. Hazards and removal

Replace flat hazard constants with projected switch-in impact:

- actual Stealth Rock type damage;
- Spikes by grounded team members;
- Toxic Spikes by Poison/Steel/grounded status vulnerability;
- Sticky Web based on affected speed matchups;
- Heavy-Duty Boots and Magic Guard;
- hazard removal PP and availability.

### 6. PP and resource economy

Generalize the gen3 low-PP term:

- recovery PP;
- only-attacking-move PP;
- hazard removal PP;
- status move PP;
- Protect / Substitute / setup PP.

### 7. Boost and setup valuation

Boost value should depend on whether it can actually convert:

- Unaware / Mold Breaker-aware;
- damage race after boosts;
- speed after boosts;
- phazing / priority / status stop conditions;
- sweep path count.

### 8. Tera option value

Replace flat `USED_TERA = -75` with:

- offensive tera value;
- defensive tera value;
- value of preserving hidden tera;
- value of forcing opponent tera;
- tera type reveal penalty / benefit.

### 9. Belief-aware threat

Use exact generator-pool belief:

- possible unrevealed moves per opponent Pokemon;
- item and ability uncertainty;
- tera uncertainty;
- expected super-effective coverage threat;
- tail-risk penalty when ahead;
- upside allowance when behind.

### 10. Scouting and concealment

Add bounded terms for:

- moves that reveal item / ability / speed / tera / coverage;
- keeping our own tera/coverage hidden;
- not revealing choice-lock unnecessarily;
- forcing opponent to reveal last move / item.

### 11. Win-condition coverage

Team-level matrix:

- Which of our Pokemon can beat each revealed/plausible opponent Pokemon?
- Which opponent Pokemon can sweep us if unrevealed coverage exists?
- How many independent answers remain?
- Is one Pokemon mandatory to preserve?

This is the Athena-inspired feature group.

## Implementation phases

### Phase A — traceable eval skeleton

Create a new Rust module, e.g. `src/genx/metagross_eval.rs`, with feature groups and a debug trace option.

Gate: exact parity with stock for disabled weights.

### Phase B — port proven terms

Include the already-passing terms:

- PP penalty;
- Unaware-aware boost valuation;
- concealment value;
- belief threat/scout terms once fixed.

Gate: reproduce Step 1 / Step 4 results.

### Phase C — add new feature groups one at a time

Recommended order:

1. Active damage race.
2. Effective speed control.
3. Hazard projection.
4. PP resource roles.
5. Win-condition coverage.
6. Tera option value.

Each feature gates independently at N=200 first, N=400 if it shows 52-56%.

### Phase D — tuning

1. Collect positions from stock FP, eval-gaps FP, and live ladder games.
2. Fit initial weights with logistic/Texel loss to game outcome.
3. Run SPSA over search-coupled knobs.
4. Promote only by paired H2H.

### Phase E — ladder

Only ladder if H2H beats true-config FP, not just 100ms FP.

Report:

- ELO;
- GXE;
- RD;
- W/L;
- unique opponents;
- long-game split;
- feature traces from losses.

## Current status

Already implemented / in testing:

- PP penalty.
- Unaware-aware boost valuation.
- Concealment value.
- Live belief tracker.
- Belief threat/scout state fields.

Still needed for standalone eval:

- Replace the rest of FP stock eval rather than patching it.
- Add trace output.
- Add damage-race and speed-control terms.
- Tune weights systematically.

## Risk assessment

Strengths:

- Differentiates the project from FP.
- Targets verified omissions.
- Keeps the proven search infrastructure.
- Belief system is genuinely novel for randbats.

Risks:

- FP's flat eval is simple and hard to beat because MCTS covers tactics.
- More features can add noise if not tuned.
- Belief terms can over-penalize and make play timid.
- Offline action-match is not enough; all claims need H2H and ladder gates.
