use crate::choices::{Choices, MoveCategory};
use crate::engine::damage_calc::type_effectiveness_modifier;
use crate::engine::state::MoveChoice;
use crate::state::{Pokemon, PokemonStatus, Side, State};
use std::env;
use std::fs;
use std::sync::OnceLock;

// Legacy 14-feature schema. Keep this dimensionality while porting the
// learned evaluator to Gen9 so model loading remains explicit and stable.
//
//  0  hp_frac_diff              side total HP fraction (s1 - s2)
//  1  alive_frac_diff           alive count fraction (s1 - s2)
//  2  active_hp_frac_diff       active mon HP fraction (s1 - s2)
//  3  status_frac_diff          opponent status fraction - own (s2 - s1)
//  4  attack_boost_diff         (s1 - s2) / 6
//  5  defense_boost_diff        (s1 - s2) / 6
//  6  special_attack_boost_diff (s1 - s2) / 6  [= special boost in gen1]
//  7  speed_boost_diff          (s1 - s2) / 6
//  8  sub_diff                  s1_has_sub - s2_has_sub
//  9  active_stat_total_diff    active mon total stats normalized
// 10  team_stat_total_diff      team total stats normalized
// 11  damage_ratio_diff         (s1_best_dmg/s2_hp - s2_best_dmg/s1_hp) — KO proximity
// 12  speed_diff                (s1_eff_speed - s2_eff_speed) / 500 — who goes first
// 13  outspeeds                 +1 if s1 faster, -1 if s2 faster, 0 if tied
const FEATURE_COUNT: usize = 14;

#[derive(Debug)]
enum LearnedValueModel {
    Linear {
        bias: f32,
        weights: Vec<f32>,
    },
    Mlp {
        hidden1: usize,
        hidden2: usize,
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: Vec<f32>,
        w3: Vec<f32>,
        b3: f32,
    },
}

static MODEL: OnceLock<Option<LearnedValueModel>> = OnceLock::new();

pub fn learned_eval_enabled() -> bool {
    model().is_some()
}

pub fn learned_value(state: &State) -> Option<f32> {
    let model = model()?;
    let features = extract_features(state);
    let logit = model.predict(&features);
    Some(sigmoid(logit).clamp(0.0, 1.0))
}

/// Raw logit from the model (before sigmoid). Used for root-centered evaluation.
pub fn learned_logit(state: &State) -> Option<f32> {
    let model = model()?;
    let features = extract_features(state);
    Some(model.predict(&features))
}

/// Standard (unscaled) sigmoid for use with learned logits.
/// mcts.rs uses a *scaled* sigmoid (0.0125 factor) tuned for hand eval units
/// (~200 range). Learned model logits are in [-4, 4] so we need the real sigmoid.
pub fn standard_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Root-centered value for MCTS: standard_sigmoid(leaf_logit - root_logit).
/// Analogous to the hand eval's sigmoid(eval - root_eval) but in logit space.
pub fn learned_rollout_value(state: &State, root_logit: f32) -> Option<f32> {
    let logit = learned_logit(state)?;
    Some(standard_sigmoid(logit - root_logit).clamp(0.0, 1.0))
}

pub fn extract_features_vec(state: &State) -> Vec<f32> {
    extract_features(state).to_vec()
}

fn model() -> Option<&'static LearnedValueModel> {
    MODEL.get_or_init(load_model).as_ref()
}

fn load_model() -> Option<LearnedValueModel> {
    let path = match env::var("METAGROSS_VALUE_MODEL") {
        Ok(path) if !path.trim().is_empty() => path,
        _ => return None,
    };
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read METAGROSS_VALUE_MODEL {}: {}", path, err));
    parse_model(&contents)
        .unwrap_or_else(|err| panic!("invalid METAGROSS_VALUE_MODEL {}: {}", path, err))
}

impl LearnedValueModel {
    fn predict(&self, features: &[f32; FEATURE_COUNT]) -> f32 {
        match self {
            LearnedValueModel::Linear { bias, weights } => {
                let mut logit = *bias;
                for (w, f) in weights.iter().zip(features.iter()) {
                    logit += w * f;
                }
                logit
            }
            LearnedValueModel::Mlp {
                hidden1,
                hidden2,
                w1,
                b1,
                w2,
                b2,
                w3,
                b3,
            } => {
                let mut a1 = vec![0.0f32; *hidden1];
                for j in 0..*hidden1 {
                    let mut z = b1[j];
                    for i in 0..FEATURE_COUNT {
                        z += features[i] * w1[i * *hidden1 + j];
                    }
                    a1[j] = z.tanh();
                }
                let mut a2 = vec![0.0f32; *hidden2];
                for j in 0..*hidden2 {
                    let mut z = b2[j];
                    for i in 0..*hidden1 {
                        z += a1[i] * w2[i * *hidden2 + j];
                    }
                    a2[j] = z.tanh();
                }
                let mut logit = *b3;
                for i in 0..*hidden2 {
                    logit += a2[i] * w3[i];
                }
                logit
            }
        }
    }
}

fn parse_floats<'a, I: Iterator<Item = &'a str>>(
    parts: I,
    label: &str,
) -> Result<Vec<f32>, String> {
    parts
        .map(|p| {
            p.parse::<f32>()
                .map_err(|e| format!("invalid {} value: {}", label, e))
        })
        .collect()
}

fn parse_model(contents: &str) -> Result<Option<LearnedValueModel>, String> {
    if contents
        .lines()
        .any(|l| l.trim() == "metagross_value_mlp_v1")
    {
        return parse_mlp_model(contents).map(Some);
    }
    let mut bias = None;
    let mut weights = None;
    for raw in contents.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line == "metagross_value_net_v1" {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("bias") => {
                bias = Some(
                    parts
                        .next()
                        .ok_or("bias missing value")?
                        .parse::<f32>()
                        .map_err(|e| format!("invalid bias: {}", e))?,
                );
            }
            Some("weights") => {
                let parsed: Result<Vec<f32>, _> = parts.map(|p| p.parse::<f32>()).collect();
                let parsed = parsed.map_err(|e| format!("invalid weight: {}", e))?;
                if parsed.len() != FEATURE_COUNT {
                    return Err(format!(
                        "expected {} weights, found {}",
                        FEATURE_COUNT,
                        parsed.len()
                    ));
                }
                weights = Some(parsed);
            }
            Some(other) => return Err(format!("unknown model line: {}", other)),
            None => {}
        }
    }
    Ok(Some(LearnedValueModel::Linear {
        bias: bias.ok_or("missing bias")?,
        weights: weights.ok_or("missing weights")?,
    }))
}

fn parse_mlp_model(contents: &str) -> Result<LearnedValueModel, String> {
    let mut dims = None;
    let mut w1 = None;
    let mut b1 = None;
    let mut w2 = None;
    let mut b2 = None;
    let mut w3 = None;
    let mut b3 = None;
    for raw in contents.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line == "metagross_value_mlp_v1" {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("dims") => {
                let parsed: Result<Vec<usize>, _> = parts.map(|p| p.parse()).collect();
                let parsed = parsed.map_err(|e| format!("invalid dims: {}", e))?;
                if parsed.len() != 4 || parsed[0] != FEATURE_COUNT || parsed[3] != 1 {
                    return Err(format!("expected dims {} H1 H2 1", FEATURE_COUNT));
                }
                dims = Some((parsed[1], parsed[2]));
            }
            Some("w1") => w1 = Some(parse_floats(parts, "w1")?),
            Some("b1") => b1 = Some(parse_floats(parts, "b1")?),
            Some("w2") => w2 = Some(parse_floats(parts, "w2")?),
            Some("b2") => b2 = Some(parse_floats(parts, "b2")?),
            Some("w3") => w3 = Some(parse_floats(parts, "w3")?),
            Some("b3") => {
                b3 = Some(
                    parts
                        .next()
                        .ok_or("b3 missing value")?
                        .parse::<f32>()
                        .map_err(|e| format!("invalid b3: {}", e))?,
                );
            }
            Some(other) => return Err(format!("unknown model line: {}", other)),
            None => {}
        }
    }
    let (hidden1, hidden2) = dims.ok_or("missing dims")?;
    let w1 = w1.ok_or("missing w1")?;
    let b1 = b1.ok_or("missing b1")?;
    let w2 = w2.ok_or("missing w2")?;
    let b2 = b2.ok_or("missing b2")?;
    let w3 = w3.ok_or("missing w3")?;
    if w1.len() != FEATURE_COUNT * hidden1 || b1.len() != hidden1 {
        return Err("w1/b1 shape mismatch".to_string());
    }
    if w2.len() != hidden1 * hidden2 || b2.len() != hidden2 || w3.len() != hidden2 {
        return Err("w2/b2/w3 shape mismatch".to_string());
    }
    Ok(LearnedValueModel::Mlp {
        hidden1,
        hidden2,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3: b3.ok_or("missing b3")?,
    })
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Policy prior for PUCT ─────────────────────────────────────────────────────
//
// AlphaGo/AlphaZero key insight: PUCT = Q(s,a) + c_puct * P(a|s) * sqrt(N) / (1+n_a)
// The prior P(a|s) guides initial exploration toward promising moves.
// Without a prior (uniform P), all moves get equal initial exploration.
// With a strong prior, the search focuses on promising moves earlier.
//
// Phase 1 (now): Computed prior from domain knowledge (base_power, STAB, type matchup)
// Phase 2 (expert iteration): Replace with a trained policy network
//
// The `METAGROSS_POLICY_MODEL` env var will trigger the learned prior when set.
// Until then, the computed prior provides a meaningful non-uniform initialization.

/// Score for a single move choice — unnormalized prior logit.
fn prior_score_for_move(active: &Pokemon, choice: &MoveChoice) -> f32 {
    match choice {
        MoveChoice::Move(mv_idx) | MoveChoice::MoveTera(mv_idx) | MoveChoice::MoveMega(mv_idx) => {
            let mv = &active.moves[mv_idx];
            if mv.id == Choices::NONE || mv.pp <= 0 {
                return 0.0;
            }
            let bp = mv.choice.base_power;
            if mv.choice.category == MoveCategory::Status || bp <= 0.0 {
                // Status moves (Thunder Wave, Sleep Powder, etc.): meaningful but not high-power
                // Give them a fixed score so they compete with weak attacks
                15.0
            } else {
                // Offensive move: base_power * STAB
                let stab = if active.types.0 == mv.choice.move_type
                    || active.types.1 == mv.choice.move_type
                {
                    1.5
                } else {
                    1.0
                };
                bp * stab
            }
        }
        MoveChoice::Switch(sw_idx) => {
            // Switching is important; give a meaningful base score
            // Scale by HP fraction of the switched-in mon so we prefer healthy mons
            // (the pokemon array is on the Side, not passed here, so use flat score)
            40.0 // Will be adjusted in compute_move_priors which has the full Side
        }
        MoveChoice::None => 5.0,
    }
}

// ── Learned policy model (METAGROSS_POLICY_MODEL) ─────────────────────────────
//
// Format: metagross_policy_value_v1
//   policy_dims IN H1 H2 1
//   policy_w1 ...  policy_b1 ...
//   policy_w2 ...  policy_b2 ...
//   policy_w3 ...  policy_b3 SCALAR
//
// Input (IN=17): 12 state features || 5 move features
//   move features: [is_status, is_physical, is_special, bp/150, 0.0]
//
// Output: Q-score per move; policy = softmax over all legal moves.
// Higher Q → more promising move.

const POLICY_IN: usize = FEATURE_COUNT + 5; // 12 state + 5 move

#[derive(Debug)]
struct PolicyModel {
    hidden1: usize,
    hidden2: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: f32,
}

static POLICY_MODEL: OnceLock<Option<PolicyModel>> = OnceLock::new();

fn policy_model() -> Option<&'static PolicyModel> {
    POLICY_MODEL.get_or_init(load_policy_model).as_ref()
}

fn load_policy_model() -> Option<PolicyModel> {
    let path = match env::var("METAGROSS_POLICY_MODEL") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => return None,
    };
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read METAGROSS_POLICY_MODEL {}: {}", path, e));
    parse_policy_model(&contents)
        .unwrap_or_else(|e| panic!("invalid METAGROSS_POLICY_MODEL {}: {}", path, e))
}

fn parse_policy_model(contents: &str) -> Result<Option<PolicyModel>, String> {
    let mut dims = None;
    let mut w1 = None;
    let mut b1 = None;
    let mut w2 = None;
    let mut b2 = None;
    let mut w3 = None;
    let mut b3 = None;
    for raw in contents.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with("metagross_") || line.starts_with("value_") {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("policy_dims") => {
                let v: Vec<usize> = parts.map(|p| p.parse().unwrap_or(0)).collect();
                if v.len() != 4 || v[0] != POLICY_IN || v[3] != 1 {
                    return Err(format!(
                        "expected policy_dims {} H1 H2 1, got {:?}",
                        POLICY_IN, v
                    ));
                }
                dims = Some((v[1], v[2]));
            }
            Some("policy_w1") => w1 = Some(parse_floats(parts, "policy_w1")?),
            Some("policy_b1") => b1 = Some(parse_floats(parts, "policy_b1")?),
            Some("policy_w2") => w2 = Some(parse_floats(parts, "policy_w2")?),
            Some("policy_b2") => b2 = Some(parse_floats(parts, "policy_b2")?),
            Some("policy_w3") => w3 = Some(parse_floats(parts, "policy_w3")?),
            Some("policy_b3") => {
                b3 = Some(
                    parts
                        .next()
                        .ok_or("policy_b3 missing")?
                        .parse::<f32>()
                        .map_err(|e| e.to_string())?,
                );
            }
            _ => {}
        }
    }
    let (h1, h2) = match dims {
        Some(d) => d,
        None => return Ok(None),
    };
    let w1 = w1.ok_or("missing policy_w1")?;
    let b1 = b1.ok_or("missing policy_b1")?;
    let w2 = w2.ok_or("missing policy_w2")?;
    let b2 = b2.ok_or("missing policy_b2")?;
    let w3 = w3.ok_or("missing policy_w3")?;
    if w1.len() != POLICY_IN * h1 || b1.len() != h1 {
        return Err("policy w1/b1 shape".into());
    }
    if w2.len() != h1 * h2 || b2.len() != h2 || w3.len() != h2 {
        return Err("policy w2/b2/w3 shape".into());
    }
    Ok(Some(PolicyModel {
        hidden1: h1,
        hidden2: h2,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3: b3.unwrap_or(0.0),
    }))
}

impl PolicyModel {
    fn forward(&self, x: &[f32; POLICY_IN]) -> f32 {
        let h1 = self.hidden1;
        let h2 = self.hidden2;
        let mut a1 = vec![0.0f32; h1];
        for j in 0..h1 {
            let mut z = self.b1[j];
            for i in 0..POLICY_IN {
                z += x[i] * self.w1[i * h1 + j];
            }
            a1[j] = z.tanh();
        }
        let mut a2 = vec![0.0f32; h2];
        for j in 0..h2 {
            let mut z = self.b2[j];
            for i in 0..h1 {
                z += a1[i] * self.w2[i * h2 + j];
            }
            a2[j] = z.tanh();
        }
        let mut out = self.b3;
        for i in 0..h2 {
            out += a2[i] * self.w3[i];
        }
        out
    }
}

/// Per-move features matching the Python expert_iter.py training code:
///   [is_status, is_physical, is_special, bp/150, stab=0]
fn move_features_for_choice(active: &Pokemon, choice: &MoveChoice) -> [f32; 5] {
    match choice {
        MoveChoice::Move(mv_idx) | MoveChoice::MoveTera(mv_idx) | MoveChoice::MoveMega(mv_idx) => {
            let mv = &active.moves[mv_idx];
            if mv.id == Choices::NONE {
                return [0.0; 5];
            }
            let is_status = if mv.choice.category == MoveCategory::Status {
                1.0
            } else {
                0.0
            };
            let is_phys = if mv.choice.category == MoveCategory::Physical {
                1.0
            } else {
                0.0
            };
            let is_spec = if mv.choice.category == MoveCategory::Special {
                1.0
            } else {
                0.0
            };
            let bp_norm = (mv.choice.base_power / 150.0).clamp(0.0, 1.0);
            [is_status, is_phys, is_spec, bp_norm, 0.0]
        }
        _ => [0.0; 5],
    }
}

/// Compute prior probabilities over legal MoveChoices for PUCT.
/// Returns probabilities summing to 1.0.
///
/// When `METAGROSS_POLICY_MODEL` is set: uses the learned policy network
///   Q(state_feats || move_feats) → score; policy = softmax(scores).
/// Fallback: domain-knowledge sqrt-scale prior (base_power / hp).
pub fn compute_move_priors(
    side: &Side,
    choices: &[MoveChoice],
    state: &State,
    is_s2: bool,
) -> Vec<f32> {
    if choices.is_empty() {
        return vec![];
    }
    let active = &side.pokemon[side.active_index];

    if let Some(pm) = policy_model() {
        // Learned policy: score each move with the policy network
        let state_feats = extract_features_for_side(state, is_s2);
        let mut scores: Vec<f32> = choices
            .iter()
            .map(|c| {
                let mf = move_features_for_choice(active, c);
                let mut inp = [0.0f32; POLICY_IN];
                inp[..FEATURE_COUNT].copy_from_slice(&state_feats[..FEATURE_COUNT]);
                inp[FEATURE_COUNT..].copy_from_slice(&mf);
                pm.forward(&inp)
            })
            .collect();
        // Softmax over Q-scores (safe: subtract max for numerical stability)
        let max_q = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|q| (q - max_q).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        if sum_e > 0.0 {
            return exps.iter().map(|e| e / sum_e).collect();
        }
    }

    // Fallback: domain-knowledge sqrt-scale prior
    let mut scores: Vec<f32> = choices
        .iter()
        .map(|c| match c {
            MoveChoice::Switch(sw_idx) => {
                let target = &side.pokemon[*sw_idx];
                if target.hp == 0 {
                    0.001
                } else {
                    40.0 * (target.hp as f32 / target.maxhp.max(1) as f32)
                }
            }
            other => prior_score_for_move(active, other).max(0.001),
        })
        .collect();
    let sqrt_scores: Vec<f32> = scores.iter().map(|s| s.sqrt()).collect();
    let sum_e: f32 = sqrt_scores.iter().sum();
    if sum_e <= 0.0 {
        let u = 1.0 / choices.len() as f32;
        return vec![u; choices.len()];
    }
    sqrt_scores.iter().map(|e| e / sum_e).collect()
}

/// Extract state features for compute_move_priors.
/// For s2 (opponent side), negate the differential features so the model
/// sees the position from the acting player's perspective.
fn extract_features_for_side(state: &State, is_s2: bool) -> [f32; FEATURE_COUNT] {
    let mut f = extract_features(state);
    if is_s2 {
        // Flip sign: model always expects "my side - opponent side" framing
        for x in f.iter_mut() {
            *x = -*x;
        }
    }
    f
}

// ── Scalar helpers ─────────────────────────────────────────────────────────────
fn hp_fraction(p: &Pokemon) -> f32 {
    if p.maxhp <= 0 || p.hp <= 0 {
        0.0
    } else {
        (p.hp as f32 / p.maxhp as f32).clamp(0.0, 1.0)
    }
}
fn side_hp_fraction(side: &Side) -> f32 {
    side.pokemon.into_iter().map(hp_fraction).sum::<f32>() / 6.0
}
fn side_alive_fraction(side: &Side) -> f32 {
    side.pokemon.into_iter().filter(|p| p.hp > 0).count() as f32 / 6.0
}
fn side_status_fraction(side: &Side) -> f32 {
    side.pokemon
        .into_iter()
        .filter(|p| p.hp > 0 && p.status != PokemonStatus::NONE)
        .count() as f32
        / 6.0
}
fn active_stat_total(side: &Side) -> f32 {
    let a = &side.pokemon[side.active_index];
    (a.attack + a.defense + a.special_attack + a.special_defense + a.speed) as f32 / 1000.0
}
fn team_stat_total(side: &Side) -> f32 {
    side.pokemon
        .into_iter()
        .filter(|p| p.hp > 0)
        .map(|p| (p.attack + p.defense + p.special_attack + p.special_defense + p.speed) as f32)
        .sum::<f32>()
        / 6000.0
}

// ── Speed helpers ─────────────────────────────────────────────────────────────
fn speed_boost_mult(boost: i8) -> f32 {
    match boost {
        6 => 4.0,
        5 => 3.5,
        4 => 3.0,
        3 => 2.5,
        2 => 2.0,
        1 => 1.5,
        0 => 1.0,
        -1 => 2.0 / 3.0,
        -2 => 0.5,
        -3 => 0.4,
        -4 => 1.0 / 3.0,
        -5 => 2.0 / 7.0,
        -6 => 0.25,
        _ => 1.0,
    }
}

fn effective_speed(side: &Side) -> f32 {
    let active = &side.pokemon[side.active_index];
    active.speed as f32 * speed_boost_mult(side.speed_boost)
}

/// Best estimated damage fraction vs defender's current HP.
/// Uses base_power × type_mult as a proxy for actual damage.
/// Returns [0, 2.0] where >1.0 means likely OHKO.
fn best_damage_fraction(attacker: &Pokemon, defender: &Pokemon) -> f32 {
    if defender.hp == 0 {
        return 0.0;
    }
    let def_hp = defender.hp as f32;
    let mut best = 0.0f32;
    for mv in attacker.moves.into_iter() {
        if mv.id == Choices::NONE || mv.choice.category == MoveCategory::Status {
            continue;
        }
        if mv.choice.base_power <= 0.0 {
            continue;
        }
        let type_mult = type_effectiveness_modifier(&mv.choice.move_type, defender);
        if type_mult == 0.0 {
            continue;
        }
        // Rough damage estimate: base_power * type_mult * atk/def proxy
        let atk = if mv.choice.category == MoveCategory::Physical {
            attacker.attack as f32
        } else {
            attacker.special_attack as f32
        };
        let def_stat = if mv.choice.category == MoveCategory::Physical {
            defender.defense as f32
        } else {
            defender.special_defense as f32
        };
        let dmg = mv.choice.base_power * type_mult * (atk / def_stat.max(1.0)) / 5.0;
        if dmg > best {
            best = dmg;
        }
    }
    (best / def_hp.max(1.0)).clamp(0.0, 2.0)
}

// ── Main feature extractor: returns [f32; 14] ─────────────────────────────────
fn extract_features(state: &State) -> [f32; FEATURE_COUNT] {
    let s1 = &state.side_one;
    let s2 = &state.side_two;
    let a1 = &s1.pokemon[s1.active_index];
    let a2 = &s2.pokemon[s2.active_index];

    // Features 11-13: what the hand eval misses
    let s1_dmg = best_damage_fraction(a1, a2);
    let s2_dmg = best_damage_fraction(a2, a1);
    let damage_ratio_diff = (s1_dmg - s2_dmg).clamp(-2.0, 2.0);

    let spd1 = effective_speed(s1);
    let spd2 = effective_speed(s2);
    let speed_diff = ((spd1 - spd2) / 500.0).clamp(-1.0, 1.0);
    let outspeeds = if spd1 > spd2 * 1.001 {
        1.0f32
    } else if spd2 > spd1 * 1.001 {
        -1.0
    } else {
        0.0
    };

    [
        side_hp_fraction(s1) - side_hp_fraction(s2),         // 0
        side_alive_fraction(s1) - side_alive_fraction(s2),   // 1
        hp_fraction(a1) - hp_fraction(a2),                   // 2
        side_status_fraction(s2) - side_status_fraction(s1), // 3
        (s1.attack_boost - s2.attack_boost) as f32 / 6.0,    // 4
        (s1.defense_boost - s2.defense_boost) as f32 / 6.0,  // 5
        (s1.special_attack_boost - s2.special_attack_boost) as f32 / 6.0, // 6
        (s1.speed_boost - s2.speed_boost) as f32 / 6.0,      // 7
        if s1.substitute_health > 0 { 1.0 } else { 0.0 }
            - if s2.substitute_health > 0 { 1.0 } else { 0.0 }, // 8
        active_stat_total(s1) - active_stat_total(s2),       // 9
        team_stat_total(s1) - team_stat_total(s2),           // 10
        damage_ratio_diff,                                   // 11 NEW: KO proximity
        speed_diff,                                          // 12 NEW: speed advantage
        outspeeds,                                           // 13 NEW: turn order
    ]
}
