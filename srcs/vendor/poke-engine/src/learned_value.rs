use crate::engine::items::Items;
use crate::pokemon::PokemonName;
use crate::state::{Pokemon, PokemonStatus, Side, State};
use std::env;
use std::fs;
use std::sync::OnceLock;

const LEGACY_FEATURE_COUNT: usize = 16;
pub const PUBLIC_FEATURE_COUNT: usize = 18;
const MAX_HIDDEN1: usize = 64;
const MAX_HIDDEN2: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeatureContract {
    LegacyDeterminization,
    PublicInformationV1,
}

#[derive(Debug)]
enum LearnedValueModel {
    Linear {
        contract: FeatureContract,
        bias: f32,
        weights: Vec<f32>,
    },
    Mlp {
        contract: FeatureContract,
        input: usize,
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
    let features = match model.contract() {
        FeatureContract::LegacyDeterminization => extract_legacy_features(state).to_vec(),
        FeatureContract::PublicInformationV1 => extract_public_features(state).to_vec(),
    };
    Some(sigmoid(model.predict(&features)).clamp(0.0, 1.0))
}

pub fn extract_legacy_features_vec(state: &State) -> Vec<f32> {
    extract_legacy_features(state).to_vec()
}

/// Player-information features invariant to sampled hidden opponent sets.
///
/// Opponent inputs are restricted to public battle facts: active HP/status,
/// fainted count, tera use, boosts, screens, hazards, substitute, and trick room.
/// Opponent reserve HP, moves, items, abilities, EVs, and stats are never read.
pub fn extract_public_features_vec(state: &State) -> Vec<f32> {
    extract_public_features(state).to_vec()
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
    fn contract(&self) -> FeatureContract {
        match self {
            Self::Linear { contract, .. } | Self::Mlp { contract, .. } => *contract,
        }
    }

    fn predict(&self, features: &[f32]) -> f32 {
        match self {
            Self::Linear { bias, weights, .. } => {
                *bias
                    + weights
                        .iter()
                        .zip(features)
                        .map(|(weight, feature)| weight * feature)
                        .sum::<f32>()
            }
            Self::Mlp {
                input,
                hidden1,
                hidden2,
                w1,
                b1,
                w2,
                b2,
                w3,
                b3,
                ..
            } => {
                debug_assert_eq!(features.len(), *input);
                let mut a1 = [0.0; MAX_HIDDEN1];
                for output in 0..*hidden1 {
                    let mut value = b1[output];
                    for feature in 0..*input {
                        value += features[feature] * w1[feature * *hidden1 + output];
                    }
                    a1[output] = value.tanh();
                }
                let mut a2 = [0.0; MAX_HIDDEN2];
                for output in 0..*hidden2 {
                    let mut value = b2[output];
                    for feature in 0..*hidden1 {
                        value += a1[feature] * w2[feature * *hidden2 + output];
                    }
                    a2[output] = value.tanh();
                }
                *b3 + a2[..*hidden2]
                    .iter()
                    .zip(w3)
                    .map(|(value, weight)| value * weight)
                    .sum::<f32>()
            }
        }
    }
}

fn parse_values<'a>(parts: impl Iterator<Item = &'a str>, label: &str) -> Result<Vec<f32>, String> {
    parts
        .map(|part| {
            part.parse::<f32>()
                .map_err(|err| format!("invalid {}: {}", label, err))
        })
        .collect()
}

fn parse_model(contents: &str) -> Result<Option<LearnedValueModel>, String> {
    let public = contents
        .lines()
        .any(|line| line.trim() == "metagross_public_value_mlp_v1");
    if public {
        return parse_public_mlp(contents).map(Some);
    }
    if !contents
        .lines()
        .any(|line| line.trim() == "metagross_value_net_v1")
    {
        return Err("missing recognized value-model schema".to_string());
    }
    let mut bias = None;
    let mut weights = None;
    for raw_line in contents.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line == "metagross_value_net_v1" {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("bias") => {
                bias = Some(
                    parts
                        .next()
                        .ok_or_else(|| "bias line missing value".to_string())?
                        .parse::<f32>()
                        .map_err(|err| format!("invalid bias: {}", err))?,
                );
            }
            Some("weights") => {
                let parsed = parse_values(parts, "weight")?;
                if parsed.len() != LEGACY_FEATURE_COUNT {
                    return Err(format!(
                        "expected {} weights, found {}",
                        LEGACY_FEATURE_COUNT,
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
        contract: FeatureContract::LegacyDeterminization,
        bias: bias.ok_or_else(|| "missing bias".to_string())?,
        weights: weights.ok_or_else(|| "missing weights".to_string())?,
    }))
}

fn parse_public_mlp(contents: &str) -> Result<LearnedValueModel, String> {
    let mut dims = None;
    let mut w1 = None;
    let mut b1 = None;
    let mut w2 = None;
    let mut b2 = None;
    let mut w3 = None;
    let mut b3 = None;
    for raw_line in contents.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line == "metagross_public_value_mlp_v1" {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("dims") => {
                let parsed: Result<Vec<usize>, _> = parts.map(str::parse).collect();
                let parsed = parsed.map_err(|err| format!("invalid dims: {}", err))?;
                if parsed.len() != 4 || parsed[0] != PUBLIC_FEATURE_COUNT || parsed[3] != 1 {
                    return Err(format!("expected dims {} H1 H2 1", PUBLIC_FEATURE_COUNT));
                }
                if parsed[1] > MAX_HIDDEN1 || parsed[2] > MAX_HIDDEN2 {
                    return Err(format!(
                        "public MLP hidden dims exceed {}x{}",
                        MAX_HIDDEN1, MAX_HIDDEN2
                    ));
                }
                dims = Some((parsed[0], parsed[1], parsed[2]));
            }
            Some("w1") => w1 = Some(parse_values(parts, "w1")?),
            Some("b1") => b1 = Some(parse_values(parts, "b1")?),
            Some("w2") => w2 = Some(parse_values(parts, "w2")?),
            Some("b2") => b2 = Some(parse_values(parts, "b2")?),
            Some("w3") => w3 = Some(parse_values(parts, "w3")?),
            Some("b3") => {
                b3 = Some(
                    parts
                        .next()
                        .ok_or_else(|| "b3 missing value".to_string())?
                        .parse::<f32>()
                        .map_err(|err| format!("invalid b3: {}", err))?,
                )
            }
            Some(other) => return Err(format!("unknown model line: {}", other)),
            None => {}
        }
    }
    let (input, hidden1, hidden2) = dims.ok_or_else(|| "missing dims".to_string())?;
    let w1 = w1.ok_or_else(|| "missing w1".to_string())?;
    let b1 = b1.ok_or_else(|| "missing b1".to_string())?;
    let w2 = w2.ok_or_else(|| "missing w2".to_string())?;
    let b2 = b2.ok_or_else(|| "missing b2".to_string())?;
    let w3 = w3.ok_or_else(|| "missing w3".to_string())?;
    if w1.len() != input * hidden1 || b1.len() != hidden1 {
        return Err("w1/b1 shape mismatch".to_string());
    }
    if w2.len() != hidden1 * hidden2 || b2.len() != hidden2 || w3.len() != hidden2 {
        return Err("w2/b2/w3 shape mismatch".to_string());
    }
    Ok(LearnedValueModel::Mlp {
        contract: FeatureContract::PublicInformationV1,
        input,
        hidden1,
        hidden2,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3: b3.ok_or_else(|| "missing b3".to_string())?,
    })
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn is_known(pokemon: &Pokemon) -> bool {
    pokemon.id != PokemonName::NONE
}

fn hp_fraction(pokemon: &Pokemon) -> f32 {
    if !is_known(pokemon) || pokemon.maxhp <= 0 || pokemon.hp <= 0 {
        0.0
    } else {
        (pokemon.hp as f32 / pokemon.maxhp as f32).clamp(0.0, 1.0)
    }
}

fn side_hp_fraction(side: &Side) -> f32 {
    side.pokemon.into_iter().map(hp_fraction).sum::<f32>() / 6.0
}

fn side_alive_fraction(side: &Side) -> f32 {
    side.pokemon
        .into_iter()
        .filter(|pokemon| is_known(pokemon) && pokemon.hp > 0)
        .count() as f32
        / 6.0
}

fn side_fainted_fraction(side: &Side) -> f32 {
    side.pokemon
        .into_iter()
        .filter(|pokemon| is_known(pokemon) && pokemon.hp <= 0)
        .count() as f32
        / 6.0
}

fn side_status_fraction(side: &Side) -> f32 {
    side.pokemon
        .into_iter()
        .filter(|pokemon| {
            is_known(pokemon) && pokemon.hp > 0 && pokemon.status != PokemonStatus::NONE
        })
        .count() as f32
        / 6.0
}

fn side_item_fraction(side: &Side) -> f32 {
    side.pokemon
        .into_iter()
        .filter(|pokemon| is_known(pokemon) && pokemon.hp > 0 && pokemon.item != Items::NONE)
        .count() as f32
        / 6.0
}

fn side_used_tera(side: &Side) -> f32 {
    side.pokemon
        .into_iter()
        .any(|pokemon| is_known(pokemon) && pokemon.terastallized) as u8 as f32
}

fn active_stat_total(side: &Side) -> f32 {
    let active = &side.pokemon[side.active_index];
    (active.attack + active.defense + active.special_attack + active.special_defense + active.speed)
        as f32
        / 1000.0
}

fn team_stat_total(side: &Side) -> f32 {
    side.pokemon
        .into_iter()
        .filter(|pokemon| is_known(pokemon) && pokemon.hp > 0)
        .map(|pokemon| {
            (pokemon.attack
                + pokemon.defense
                + pokemon.special_attack
                + pokemon.special_defense
                + pokemon.speed) as f32
        })
        .sum::<f32>()
        / 6000.0
}

fn screen_score(side: &Side) -> f32 {
    let conditions = &side.side_conditions;
    (conditions.reflect + conditions.light_screen + conditions.aurora_veil * 2) as f32 / 8.0
}

fn hazard_score(side: &Side) -> f32 {
    let conditions = &side.side_conditions;
    (conditions.stealth_rock
        + conditions.spikes
        + conditions.toxic_spikes
        + conditions.sticky_web * 2) as f32
        / 8.0
}

fn boost_features(side: &Side) -> [f32; 5] {
    [
        side.attack_boost as f32 / 6.0,
        side.defense_boost as f32 / 6.0,
        side.special_attack_boost as f32 / 6.0,
        side.special_defense_boost as f32 / 6.0,
        side.speed_boost as f32 / 6.0,
    ]
}

fn extract_legacy_features(state: &State) -> [f32; LEGACY_FEATURE_COUNT] {
    let side_one = &state.side_one;
    let side_two = &state.side_two;
    let side_one_boosts = boost_features(side_one);
    let side_two_boosts = boost_features(side_two);
    [
        side_hp_fraction(side_one) - side_hp_fraction(side_two),
        side_alive_fraction(side_one) - side_alive_fraction(side_two),
        hp_fraction(&side_one.pokemon[side_one.active_index])
            - hp_fraction(&side_two.pokemon[side_two.active_index]),
        side_status_fraction(side_two) - side_status_fraction(side_one),
        side_item_fraction(side_one) - side_item_fraction(side_two),
        side_used_tera(side_two) - side_used_tera(side_one),
        side_one_boosts[0] - side_two_boosts[0],
        side_one_boosts[1] - side_two_boosts[1],
        side_one_boosts[2] - side_two_boosts[2],
        side_one_boosts[3] - side_two_boosts[3],
        side_one_boosts[4] - side_two_boosts[4],
        screen_score(side_one) - screen_score(side_two),
        hazard_score(side_two) - hazard_score(side_one),
        active_stat_total(side_one) - active_stat_total(side_two),
        team_stat_total(side_one) - team_stat_total(side_two),
        (side_one.substitute_health > 0) as u8 as f32
            - (side_two.substitute_health > 0) as u8 as f32,
    ]
}

fn extract_public_features(state: &State) -> [f32; PUBLIC_FEATURE_COUNT] {
    let own = &state.side_one;
    let opponent = &state.side_two;
    let own_active = &own.pokemon[own.active_index];
    let opponent_active = &opponent.pokemon[opponent.active_index];
    let own_boosts = boost_features(own);
    let opponent_boosts = boost_features(opponent);
    [
        side_hp_fraction(own),
        side_alive_fraction(own),
        hp_fraction(own_active),
        hp_fraction(opponent_active),
        side_fainted_fraction(opponent),
        side_status_fraction(own),
        (opponent_active.status != PokemonStatus::NONE) as u8 as f32,
        side_item_fraction(own),
        side_used_tera(opponent) - side_used_tera(own),
        own_boosts[0] - opponent_boosts[0],
        own_boosts[1] - opponent_boosts[1],
        own_boosts[2] - opponent_boosts[2],
        own_boosts[3] - opponent_boosts[3],
        own_boosts[4] - opponent_boosts[4],
        screen_score(own) - screen_score(opponent),
        hazard_score(opponent) - hazard_score(own),
        (own.substitute_health > 0) as u8 as f32 - (opponent.substitute_health > 0) as u8 as f32,
        state.trick_room.active as u8 as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choices::Choices;
    use crate::state::{PokemonIndex, PokemonMoveIndex};

    #[test]
    fn public_features_ignore_sampled_private_opponent_sets() {
        let mut first = State::default();
        first.side_one.pokemon[PokemonIndex::P0].id = PokemonName::PIKACHU;
        first.side_two.pokemon[PokemonIndex::P0].id = PokemonName::CHARIZARD;
        first.side_two.pokemon[PokemonIndex::P0].hp = 50;
        first.side_two.pokemon[PokemonIndex::P0].maxhp = 100;
        let mut second = first.clone();
        for index in [
            PokemonIndex::P1,
            PokemonIndex::P2,
            PokemonIndex::P3,
            PokemonIndex::P4,
            PokemonIndex::P5,
        ] {
            let pokemon = &mut second.side_two.pokemon[index];
            pokemon.id = PokemonName::BULBASAUR;
            pokemon.hp = 317;
            pokemon.maxhp = 401;
            pokemon.attack = 399;
            pokemon.defense = 211;
            pokemon.speed = 303;
            pokemon.item = Items::LEFTOVERS;
            pokemon.replace_move(PokemonMoveIndex::M0, Choices::THUNDERBOLT);
        }
        let active = &mut second.side_two.pokemon[PokemonIndex::P0];
        active.attack = 401;
        active.defense = 73;
        active.speed = 499;
        active.item = Items::LEFTOVERS;
        active.replace_move(PokemonMoveIndex::M0, Choices::THUNDERBOLT);

        assert_eq!(
            extract_public_features(&first),
            extract_public_features(&second)
        );
        assert_ne!(
            extract_legacy_features(&first),
            extract_legacy_features(&second)
        );
    }

    #[test]
    fn public_features_respond_to_public_changes() {
        let mut state = State::default();
        state.side_one.pokemon[PokemonIndex::P0].id = PokemonName::PIKACHU;
        state.side_two.pokemon[PokemonIndex::P0].id = PokemonName::CHARIZARD;
        let before = extract_public_features(&state);
        state.side_two.pokemon[PokemonIndex::P0].hp /= 2;
        state.side_two.attack_boost = 2;
        let after = extract_public_features(&state);

        assert_ne!(before, after);
    }
}
