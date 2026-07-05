# Foul Play / poke-engine Eval Audit

## Scope

Audited files:

- `external/source-dist/poke_engine-0.0.47/src/genx/evaluate.rs`
- `external/source-dist/poke_engine-0.0.47/src/gen3/evaluate.rs`
- `external/source-dist/poke_engine-0.0.47/src/genx/damage_calc.rs`
- `external/source-dist/poke_engine-0.0.47/src/state.rs`
- `external/foul-play/fp/search/main.py`
- `external/foul-play/fp/search/random_battles.py`
- `external/foul-play/data/pkmn_sets.py`
- `scripts/run_foul_play.py`

## Stock gen9 eval: what it scores

File: `external/source-dist/poke_engine-0.0.47/src/genx/evaluate.rs`.

Stock constants:

- Alive and HP: `POKEMON_ALIVE = 30.0`, `POKEMON_HP = 100.0`.
- Tera spent: `USED_TERA = -75.0`.
- Boost weights: attack `30`, defense `15`, special attack `30`, special defense `15`, speed `30`.
- Status penalties: freeze `-40`, sleep `-25`, paralysis `-25`, toxic `-30`, poison `-10`, burn `-25`.
- Active volatiles: Leech Seed `-30`, Substitute `+40`, Confusion `-20`.
- Screens / side effects: Reflect `+20`, Light Screen `+20`, Aurora Veil `+40`, Safeguard `+5`, Tailwind `+7`, Healing Wish `+30`.
- Hazards: Stealth Rock `-10`, Spikes `-7`, Toxic Spikes `-7`, Sticky Web `-25`.

Scored per alive Pokemon:

- HP percentage.
- Status with limited ability exceptions.
- Item present bonus.
- Alive bonus.
- Hazards, with Heavy-Duty Boots / Magic Guard checks.

Only active Pokemon get:

- Leech Seed, Substitute, Confusion.
- Attack / defense / special attack / special defense / speed boosts.

Tera handling:

- If any Pokemon on a side has terastallized, that side gets `USED_TERA = -75`.
- This is flat and does not value whether tera was actually good or preserved useful option value.

## Major omissions in stock gen9 eval

Stock `genx/evaluate.rs` does **not** score:

- PP / stall / resource depletion.
- Move quality, coverage, STAB, damage races, or KO thresholds.
- Effective speed relationships, Choice Scarf, Swift Swim / Chlorophyll / weather speed, Trick Room, priority.
- Weather or terrain value, except via simulated future turns.
- Future Sight, Wish timing, Protect loops, Encore / Taunt / Disable, recovery loops.
- Exact Stealth Rock type damage; it uses a flat constant.
- Unaware / Mold Breaker in boost valuation, even though damage calc handles Unaware.
- Hidden-information value: scouting, concealment, unrevealed coverage, unrevealed tera.
- Win-condition structure: setup sweep viability, last-mon matchups, PP endgames, residual locks.

## Useful ideas present in other gens

File: `external/source-dist/poke_engine-0.0.47/src/gen3/evaluate.rs`.

Gen3 has a low-PP penalty missing from gen9:

```rust
for mv in pokemon.moves.into_iter() {
    if mv.pp <= 10 {
        let penalty = (mv.pp * 3) as f32 - 30.0;
        score += penalty;
    }
}
```

Design implication: port the concept, but do not blindly penalize every low-PP move equally. Recovery, attacking coverage, hazard removal, status, and setup moves should have different PP value.

Gen3 also uses layer-specific Spikes constants instead of gen9's flat `-7 * layers`.

## Engine fields available for a stronger eval

Useful Rust state fields:

- `Pokemon` has ability, base ability, item, stats, status, sleep/rest turns, tera fields, moves: `external/source-dist/poke_engine-0.0.47/src/state.rs`.
- `Move` has id, disabled flag, PP, and full choice data.
- `SideConditions` include hazards, screens, Tailwind, ToxicCount.
- Weather, terrain, and Trick Room are present on `State`.
- Volatile statuses and durations are present.

PP is trustworthy enough to use:

- FP reads PP from request JSON for our active moves.
- Opponent PP is decremented from observed moves, including Pressure handling.
- `poke_engine` tracks `Move.pp`.

Unaware is handled in `genx/damage_calc.rs`, but stock eval's boost valuation ignores it. This is why boost-heavy positions into Unaware can be overvalued in eval even if simulated damage is correct.

## FP determinization and revealed moves

FP **does remember revealed moves**:

- `Pokemon.moves` persists revealed moves.
- `data/pkmn_sets.py` filters candidate sets so all revealed moves must be present.
- Randbats sampling uses remaining consistent sets.

What FP does **not** do:

- It does not evaluate uncertainty directly at leaf nodes.
- Each determinized world assumes a concrete full set, so future lines can suffer strategy fusion.
- It does not score scouting value or concealment value directly.

## Current Metagross modifications versus FP original

FP original:

- Battle client, parser, set inference, determinized MCTS orchestration.
- Stock `genx/evaluate.rs` scalar eval.
- Randbats sampler and pkmn data handling.

Metagross additions:

- Eval-gaps patch: PP penalty, Unaware-aware boost valuation, concealment value.
- Belief tracker: exact generator-pool belief over possible opponent sets.
- Belief-aware eval state fields: `s1_threat`, `s2_threat`, `scout_value`.
- Root-prior and opponent-prior plumbing.
- H2H and ladder evaluation harnesses.

## Keep / replace / add / avoid

Keep:

- FP's fast search and battle infrastructure.
- Revealed-move / item / ability / speed-range inference.
- Per-world search with sample aggregation.

Replace:

- The flat scalar eval constants with an original belief-aware eval.
- Flat boost valuation with matchup-aware boost valuation.
- Flat hazards with projected switch-in damage and status/speed effects.

Add:

- PP economy.
- Active-vs-active damage race and KO threshold terms.
- Effective speed and priority control.
- Win-condition and endgame terms.
- Belief-aware threat, scouting, and concealment terms.
- Debug trace output per feature.

Avoid:

- Mutating real battle hidden info outside deepcopied determinizations.
- Encoding belief by lying in mechanical fields like fake HP/item/ability.
- Replacing search decisions with policy overrides.
