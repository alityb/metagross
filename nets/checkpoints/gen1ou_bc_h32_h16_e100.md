# gen1ou_bc_h32_h16_e100 — Model Card

**Created:** 2026-06-26

## Architecture
- Input: 12 gen1-appropriate features
- Hidden: 32 → 16 (tanh activations)
- Output: win probability ∈ [0,1]
- Parameters: ~900 effective
- Format: `metagross_value_mlp_v1` (poke-engine Rust parser compatible)

## Training Data
- Source: `jakegrigsby/metamon-parsed-replays` gen1ou subset (HuggingFace, cc-by-nc-4.0)
- 199,024 human replay files, 10.67M state-label rows
- Label: 1 = player won, 0 = player lost (per Metamon filename)
- Split: 90/10 random, row-level (no by-game split — files are already separate)
- Epochs: 100 at lr=0.001, batch=8192 on EC2 m5.2xlarge

## Features (12)
| # | Name | Description |
|---|------|-------------|
| 0 | hp_frac_diff | Player team total HP fraction − opponent |
| 1 | alive_frac_diff | Player alive count − opponent (÷6) |
| 2 | active_hp_frac_diff | Active mon HP fraction difference |
| 3 | status_frac_diff | Opponent has active status − player |
| 4 | attack_boost_diff | Active atk boost diff ÷6 |
| 5 | defense_boost_diff | Active def boost diff ÷6 |
| 6 | special_attack_boost_diff | Active spa boost diff ÷6 |
| 7 | speed_boost_diff | Active spe boost diff ÷6 |
| 8 | sub_diff | Substitute present (always 0 in Metamon; active in poke-engine) |
| 9 | active_stat_total_diff | Active mon total base stats ÷1000 diff |
| 10 | team_stat_total_diff | Team total base stats ÷6000 diff |
| 11 | type_adv | log2(best attacking type mult, player) − log2(best, opponent) |

## Metrics
- Held-out accuracy: **65.43%** (base rate 49.3%)
- Held-out Brier: **0.2097**
- Train BCE ≈ Held BCE (no overfitting; underfitting; model can likely improve further)

## Known Limitations
- Feature 8 (sub_diff) is always 0 in training data (Metamon doesn't track subs)
  → poke-engine will set this correctly at inference; model assigns zero weight to it
- Feature 3 (status_frac_diff): in training, opponent status = active mon only;
  in poke-engine, all mons' statuses are known → slight train/inference distribution shift
- Feature 10 (team_stat_total_diff): opponent bench stats unknown in Metamon → bias

## Phase 1 Gate Status
- Pending A/B at N=100 vs stock Foul Play gen1ou
