use crate::engine::state::MoveChoice;
use crate::mcts::perform_mcts_seeded;
use crate::state::{Side, State};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

pub const SHARED_ROOT_SOLVER_CONTRACT: &str = "weighted-shared-rm-plus-v1";
pub const MAX_PARTICLES: usize = 64;
pub const MAX_SHARED_ACTIONS: usize = 16;
pub const MAX_OPPONENT_ACTIONS: usize = 16;
pub const MAX_PAYOFF_CELLS: u64 = 16_384;
pub const MAX_RM_ITERATIONS: u32 = 1_000_000;
pub const MAX_CONTINUATION_ITERATIONS: u32 = 1_000_000;
pub const MAX_TOTAL_CONTINUATION_ITERATIONS: u64 = 100_000_000;
const WEIGHT_TOLERANCE: f64 = 1e-6;
const PROBABILITY_TOLERANCE: f64 = 1e-9;
const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

#[derive(Clone, Debug, PartialEq)]
pub struct SharedRootConfig {
    pub iterations: u32,
    pub continuation_iterations: u32,
    pub seed: u64,
    pub prior_strength: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedRootAction {
    pub action: String,
    pub probability: f64,
    pub counterfactual_value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedRootSourceParticle {
    pub input_index: u32,
    pub input_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedRootContinuation {
    pub seed: u64,
    pub requested_iterations: u32,
    pub executed_iterations: u32,
    pub visits: u32,
    pub total_score: f32,
    pub total_score_bits: u32,
    pub payoff: f64,
    pub payoff_bits: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedRootReplayParticle {
    pub canonical_index: u32,
    pub state: String,
    pub normalized_weight: f64,
    pub source_particles: Vec<SharedRootSourceParticle>,
    pub opponent_action_support: Vec<String>,
    pub normalized_opponent_prior: Option<Vec<f64>>,
    pub payoff_matrix: Vec<Vec<f64>>,
    pub continuations: Vec<Vec<SharedRootContinuation>>,
    pub opponent_policy: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedRootReplayCapture {
    pub schema_version: u32,
    pub solver_contract: String,
    pub configuration: SharedRootConfig,
    pub own_action_support: Vec<String>,
    pub normalized_player_prior: Option<Vec<f64>>,
    pub canonical_particles: Vec<SharedRootReplayParticle>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedRootDiagnostics {
    pub solver_contract: String,
    pub iterations: u32,
    pub continuation_iterations: u32,
    pub seed: u64,
    pub prior_strength: f64,
    pub expected_value: f64,
    pub player_best_response_value: f64,
    pub opponent_best_response_value: f64,
    pub player_best_response_gain: f64,
    pub opponent_best_response_gain: f64,
    pub nash_conv: f64,
    pub exploitability: f64,
    pub player_regret_bound: f64,
    pub opponent_regret_bound: f64,
    pub total_regret_bound: f64,
    pub payoff_cells: u64,
    pub total_forced_continuation_iterations: u64,
    pub input_particle_count: u32,
    pub positive_particle_count: u32,
    pub canonical_particle_count: u32,
    pub normalized_weight_sum: f64,
    pub action_support_digest: String,
    pub particle_digest: String,
    pub payoff_digest: String,
    pub player_prior_digest: String,
    pub opponent_prior_digest: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedRootResult {
    pub policy: Vec<SharedRootAction>,
    pub opponent_policies: Vec<Vec<(String, f64)>>,
    pub diagnostics: SharedRootDiagnostics,
    pub replay_capture: SharedRootReplayCapture,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatrixSolveResult {
    pub player_policy: Vec<f64>,
    pub opponent_policies: Vec<Vec<f64>>,
    pub counterfactual_values: Vec<f64>,
    pub expected_value: f64,
    pub player_best_response_value: f64,
    pub opponent_best_response_value: f64,
    pub player_best_response_gain: f64,
    pub opponent_best_response_gain: f64,
    pub nash_conv: f64,
    pub exploitability: f64,
    pub player_regret_bound: f64,
    pub opponent_regret_bound: f64,
    pub total_regret_bound: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SharedRootError {
    InvalidBounds(&'static str),
    InvalidWeights,
    InvalidPayoffMatrix,
    InvalidPrior {
        role: &'static str,
        world: Option<usize>,
    },
    DuplicatePriorAction {
        role: &'static str,
        action: String,
    },
    UnknownPriorAction {
        role: &'static str,
        action: String,
    },
    EmptyActionSupport {
        role: &'static str,
        world: usize,
    },
    AmbiguousAction {
        role: &'static str,
        world: usize,
        action: String,
    },
    SharedSupportMismatch {
        world: usize,
    },
    NonFinitePayoff,
    ContinuationFailed {
        world: usize,
        opponent_action: String,
    },
}

impl fmt::Display for SharedRootError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBounds(message) => write!(formatter, "invalid shared-root bounds: {message}"),
            Self::InvalidWeights => write!(formatter, "particle weights must be finite, nonnegative, normalized, and have positive mass"),
            Self::InvalidPayoffMatrix => write!(formatter, "invalid shared-root payoff matrix"),
            Self::InvalidPrior { role, world } => match world {
                Some(world) => write!(formatter, "invalid {role} prior for world {world}"),
                None => write!(formatter, "invalid {role} prior"),
            },
            Self::DuplicatePriorAction { role, action } => write!(formatter, "duplicate {role} prior action: {action}"),
            Self::UnknownPriorAction { role, action } => write!(formatter, "unknown {role} prior action: {action}"),
            Self::EmptyActionSupport { role, world } => write!(formatter, "world {world} has empty {role} action support"),
            Self::AmbiguousAction { role, world, action } => write!(formatter, "world {world} has ambiguous {role} action: {action}"),
            Self::SharedSupportMismatch { world } => write!(formatter, "positive world {world} has different side-one action support"),
            Self::NonFinitePayoff => write!(formatter, "shared-root payoff oracle returned a nonfinite value"),
            Self::ContinuationFailed { world, opponent_action } => write!(formatter, "forced continuation failed for world {world} and opponent action {opponent_action}"),
        }
    }
}

impl Error for SharedRootError {}

#[derive(Clone)]
struct PreparedParticle {
    state: State,
    state_string: String,
    weight: f64,
    own_by_name: HashMap<String, MoveChoice>,
    opponent_names: Vec<String>,
    opponent_options: Vec<MoveChoice>,
    opponent_prior: Option<Vec<f64>>,
    source_particles: Vec<SharedRootSourceParticle>,
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn stable_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn digest(parts: impl IntoIterator<Item = impl AsRef<[u8]>>) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for part in parts {
        for byte in part.as_ref() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn row_seed(seed: u64, state_string: &str, opponent_action: &str) -> u64 {
    splitmix64(
        seed ^ stable_hash_bytes(state_string.as_bytes())
            ^ stable_hash_bytes(opponent_action.as_bytes()).rotate_left(23),
    )
}

fn canonical_action(side: &Side, choice: &MoveChoice) -> String {
    match choice {
        MoveChoice::Switch(_) => format!("switch {}", choice.to_string(side)),
        _ => choice.to_string(side),
    }
    .to_lowercase()
}

fn canonical_options(
    side: &Side,
    options: Vec<MoveChoice>,
    role: &'static str,
    world: usize,
) -> Result<(Vec<String>, HashMap<String, MoveChoice>), SharedRootError> {
    if options.is_empty() {
        return Err(SharedRootError::EmptyActionSupport { role, world });
    }
    let mut by_name = HashMap::with_capacity(options.len());
    for option in options {
        let name = canonical_action(side, &option);
        if by_name.insert(name.clone(), option).is_some() {
            return Err(SharedRootError::AmbiguousAction {
                role,
                world,
                action: name,
            });
        }
    }
    let mut names: Vec<_> = by_name.keys().cloned().collect();
    names.sort();
    Ok((names, by_name))
}

fn normalize_prior(
    supplied: Option<&[(String, f64)]>,
    support: &[String],
    role: &'static str,
    world: Option<usize>,
    ignore_unknown: bool,
) -> Result<Option<Vec<f64>>, SharedRootError> {
    let Some(supplied) = supplied else {
        return Ok(None);
    };
    let positions: HashMap<&str, usize> = support
        .iter()
        .enumerate()
        .map(|(index, action)| (action.as_str(), index))
        .collect();
    let mut values = vec![0.0; support.len()];
    let mut seen = HashSet::with_capacity(supplied.len());
    for (action, probability) in supplied {
        let action = action.to_lowercase();
        if !seen.insert(action.clone()) {
            return Err(SharedRootError::DuplicatePriorAction { role, action });
        }
        if !probability.is_finite() || *probability < 0.0 {
            return Err(SharedRootError::InvalidPrior { role, world });
        }
        let Some(index) = positions.get(action.as_str()) else {
            if ignore_unknown {
                continue;
            }
            return Err(SharedRootError::UnknownPriorAction { role, action });
        };
        values[*index] = *probability;
    }
    let total: f64 = values.iter().sum();
    if !total.is_finite() {
        return Err(SharedRootError::InvalidPrior { role, world });
    }
    if total <= 0.0 && ignore_unknown {
        return Ok(None);
    }
    if total <= 0.0 {
        return Err(SharedRootError::InvalidPrior { role, world });
    }
    for value in &mut values {
        *value /= total;
    }
    Ok(Some(values))
}

fn validate_config(config: &SharedRootConfig) -> Result<(), SharedRootError> {
    if config.iterations == 0 || config.iterations > MAX_RM_ITERATIONS {
        return Err(SharedRootError::InvalidBounds(
            "iterations must be between 1 and 1000000",
        ));
    }
    if config.continuation_iterations == 0
        || config.continuation_iterations > MAX_CONTINUATION_ITERATIONS
    {
        return Err(SharedRootError::InvalidBounds(
            "continuation_iterations must be between 1 and 1000000",
        ));
    }
    if !config.prior_strength.is_finite() || !(0.0..=1_000.0).contains(&config.prior_strength) {
        return Err(SharedRootError::InvalidBounds(
            "prior_strength must be finite and in [0, 1000]",
        ));
    }
    Ok(())
}

fn validate_weights(weights: &[f64]) -> Result<(), SharedRootError> {
    if weights.is_empty()
        || weights.len() > MAX_PARTICLES
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(SharedRootError::InvalidWeights);
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 || (total - 1.0).abs() > WEIGHT_TOLERANCE {
        return Err(SharedRootError::InvalidWeights);
    }
    Ok(())
}

fn normalized_or_uniform(regrets: &[f64]) -> Vec<f64> {
    let total: f64 = regrets.iter().sum();
    if total > 0.0 && total.is_finite() {
        regrets.iter().map(|regret| regret / total).collect()
    } else {
        vec![1.0 / regrets.len() as f64; regrets.len()]
    }
}

fn warm_regrets(prior: Option<&[f64]>, length: usize, strength: f64) -> Vec<f64> {
    prior
        .map(|prior| prior.iter().map(|value| value * strength).collect())
        .unwrap_or_else(|| vec![0.0; length])
}

fn average_strategy(sum: Vec<f64>, iterations: u32) -> Vec<f64> {
    let denominator = iterations as f64;
    let mut result: Vec<_> = sum.into_iter().map(|value| value / denominator).collect();
    let total: f64 = result.iter().sum();
    for value in &mut result {
        *value /= total;
        if value.abs() < PROBABILITY_TOLERANCE {
            *value = 0.0;
        }
    }
    let total: f64 = result.iter().sum();
    for value in &mut result {
        *value /= total;
    }
    result
}

/// Solve a weighted Bayesian zero-sum root game with deterministic full-batch RM+.
///
/// `payoffs[world][own_action][opponent_action]` is side one's payoff. Side one
/// has one policy shared across worlds; side two has one policy per world.
pub fn solve_weighted_matrix_rm_plus(
    weights: &[f64],
    payoffs: &[Vec<Vec<f64>>],
    iterations: u32,
    player_prior: Option<&[f64]>,
    opponent_priors: &[Option<Vec<f64>>],
    prior_strength: f64,
) -> Result<MatrixSolveResult, SharedRootError> {
    validate_weights(weights)?;
    if iterations == 0
        || iterations > MAX_RM_ITERATIONS
        || payoffs.len() != weights.len()
        || opponent_priors.len() != weights.len()
        || !prior_strength.is_finite()
        || !(0.0..=1_000.0).contains(&prior_strength)
    {
        return Err(SharedRootError::InvalidPayoffMatrix);
    }
    let action_count = payoffs.first().map(Vec::len).unwrap_or(0);
    if action_count == 0 || action_count > MAX_SHARED_ACTIONS {
        return Err(SharedRootError::InvalidPayoffMatrix);
    }
    if let Some(prior) = player_prior {
        if prior.len() != action_count
            || prior.iter().any(|value| !value.is_finite() || *value < 0.0)
            || (prior.iter().sum::<f64>() - 1.0).abs() > WEIGHT_TOLERANCE
        {
            return Err(SharedRootError::InvalidPrior {
                role: "side-one",
                world: None,
            });
        }
    }
    let mut opponent_counts = Vec::with_capacity(payoffs.len());
    for (world, matrix) in payoffs.iter().enumerate() {
        if matrix.len() != action_count {
            return Err(SharedRootError::InvalidPayoffMatrix);
        }
        let opponent_count = matrix.first().map(Vec::len).unwrap_or(0);
        if opponent_count == 0
            || opponent_count > MAX_OPPONENT_ACTIONS
            || matrix.iter().any(|row| {
                row.len() != opponent_count || row.iter().any(|value| !value.is_finite())
            })
        {
            return Err(SharedRootError::InvalidPayoffMatrix);
        }
        if let Some(prior) = &opponent_priors[world] {
            if prior.len() != opponent_count
                || prior.iter().any(|value| !value.is_finite() || *value < 0.0)
                || (prior.iter().sum::<f64>() - 1.0).abs() > WEIGHT_TOLERANCE
            {
                return Err(SharedRootError::InvalidPrior {
                    role: "side-two",
                    world: Some(world),
                });
            }
        }
        opponent_counts.push(opponent_count);
    }

    let mut player_regrets = warm_regrets(player_prior, action_count, prior_strength);
    let mut opponent_regrets: Vec<Vec<f64>> = opponent_counts
        .iter()
        .enumerate()
        .map(|(world, count)| {
            warm_regrets(opponent_priors[world].as_deref(), *count, prior_strength)
        })
        .collect();
    let mut player_sum = vec![0.0; action_count];
    let mut opponent_sums: Vec<Vec<f64>> = opponent_counts
        .iter()
        .map(|count| vec![0.0; *count])
        .collect();

    for _ in 0..iterations {
        let player = normalized_or_uniform(&player_regrets);
        let opponents: Vec<Vec<f64>> = opponent_regrets
            .iter()
            .map(|regrets| normalized_or_uniform(regrets))
            .collect();
        for (sum, probability) in player_sum.iter_mut().zip(&player) {
            *sum += probability;
        }
        for (sums, policy) in opponent_sums.iter_mut().zip(&opponents) {
            for (sum, probability) in sums.iter_mut().zip(policy) {
                *sum += probability;
            }
        }

        let mut own_values = vec![0.0; action_count];
        for (world, weight) in weights.iter().copied().enumerate() {
            if weight == 0.0 {
                continue;
            }
            for (own_action, value) in own_values.iter_mut().enumerate() {
                *value += weight
                    * payoffs[world][own_action]
                        .iter()
                        .zip(&opponents[world])
                        .map(|(payoff, probability)| payoff * probability)
                        .sum::<f64>();
            }
        }
        let expected: f64 = player
            .iter()
            .zip(&own_values)
            .map(|(probability, value)| probability * value)
            .sum();
        for (regret, value) in player_regrets.iter_mut().zip(own_values) {
            *regret = (*regret + value - expected).max(0.0);
        }

        for world in 0..weights.len() {
            if weights[world] == 0.0 {
                continue;
            }
            let opponent_values: Vec<f64> = (0..opponent_counts[world])
                .map(|opponent_action| {
                    player
                        .iter()
                        .enumerate()
                        .map(|(own_action, probability)| {
                            probability * payoffs[world][own_action][opponent_action]
                        })
                        .sum()
                })
                .collect();
            let local_expected: f64 = opponents[world]
                .iter()
                .zip(&opponent_values)
                .map(|(probability, value)| probability * value)
                .sum();
            for (regret, value) in opponent_regrets[world].iter_mut().zip(opponent_values) {
                *regret = (*regret + local_expected - value).max(0.0);
            }
        }
    }

    let player_policy = average_strategy(player_sum, iterations);
    let opponent_policies: Vec<Vec<f64>> = opponent_sums
        .into_iter()
        .map(|sum| average_strategy(sum, iterations))
        .collect();
    let mut counterfactual_values = vec![0.0; action_count];
    for (world, weight) in weights.iter().copied().enumerate() {
        if weight == 0.0 {
            continue;
        }
        for (own_action, value) in counterfactual_values.iter_mut().enumerate() {
            *value += weight
                * payoffs[world][own_action]
                    .iter()
                    .zip(&opponent_policies[world])
                    .map(|(payoff, probability)| payoff * probability)
                    .sum::<f64>();
        }
    }
    let expected_value: f64 = player_policy
        .iter()
        .zip(&counterfactual_values)
        .map(|(probability, value)| probability * value)
        .sum();
    let player_best_response_value = counterfactual_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let opponent_best_response_value: f64 = weights
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, weight)| *weight > 0.0)
        .map(|(world, weight)| {
            let minimum = (0..opponent_counts[world])
                .map(|opponent_action| {
                    player_policy
                        .iter()
                        .enumerate()
                        .map(|(own_action, probability)| {
                            probability * payoffs[world][own_action][opponent_action]
                        })
                        .sum::<f64>()
                })
                .fold(f64::INFINITY, f64::min);
            weight * minimum
        })
        .sum();
    let player_best_response_gain = (player_best_response_value - expected_value).max(0.0);
    let opponent_best_response_gain = (expected_value - opponent_best_response_value).max(0.0);
    let nash_conv = player_best_response_gain + opponent_best_response_gain;
    let player_regret_bound =
        player_regrets.iter().copied().fold(0.0, f64::max) / iterations as f64;
    let opponent_regret_bound: f64 = weights
        .iter()
        .copied()
        .enumerate()
        .map(|(world, weight)| {
            weight * opponent_regrets[world].iter().copied().fold(0.0, f64::max) / iterations as f64
        })
        .sum();

    Ok(MatrixSolveResult {
        player_policy,
        opponent_policies,
        counterfactual_values,
        expected_value,
        player_best_response_value,
        opponent_best_response_value,
        player_best_response_gain,
        opponent_best_response_gain,
        nash_conv,
        exploitability: nash_conv / 2.0,
        player_regret_bound,
        opponent_regret_bound,
        total_regret_bound: player_regret_bound + opponent_regret_bound,
    })
}

fn compare_optional_priors(left: &Option<Vec<f64>>, right: &Option<Vec<f64>>) -> Ordering {
    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left), Some(right)) => left
            .iter()
            .zip(right)
            .find_map(|(left, right)| {
                let ordering = left.total_cmp(right);
                (ordering != Ordering::Equal).then_some(ordering)
            })
            .unwrap_or_else(|| left.len().cmp(&right.len())),
    }
}

fn prepare_particles(
    states: &[State],
    weights: &[f64],
    opponent_priors: Option<&[Option<Vec<(String, f64)>>]>,
) -> Result<(Vec<PreparedParticle>, Vec<String>, u32), SharedRootError> {
    validate_weights(weights)?;
    if states.len() != weights.len()
        || states.len() > MAX_PARTICLES
        || opponent_priors.is_some_and(|priors| priors.len() != states.len())
    {
        return Err(SharedRootError::InvalidBounds(
            "states, weights, and opponent priors must have matching bounded lengths",
        ));
    }
    let positive_count = weights.iter().filter(|weight| **weight > 0.0).count() as u32;
    let mut particles = Vec::with_capacity(positive_count as usize);
    let mut shared_names: Option<Vec<String>> = None;
    for (world, (state, weight)) in states.iter().zip(weights).enumerate() {
        if *weight == 0.0 {
            continue;
        }
        let root = state.clone();
        let (own_options, opponent_options) = root.root_get_all_options();
        let (own_names, own_by_name) =
            canonical_options(&state.side_one, own_options, "side-one", world)?;
        let (opponent_names, opponent_by_name) =
            canonical_options(&state.side_two, opponent_options, "side-two", world)?;
        if own_names.len() > MAX_SHARED_ACTIONS || opponent_names.len() > MAX_OPPONENT_ACTIONS {
            return Err(SharedRootError::InvalidBounds(
                "root action support exceeds 16 actions",
            ));
        }
        if let Some(expected) = &shared_names {
            if expected != &own_names {
                return Err(SharedRootError::SharedSupportMismatch { world });
            }
        } else {
            shared_names = Some(own_names);
        }
        let supplied_prior = opponent_priors.and_then(|priors| priors[world].as_deref());
        let opponent_prior = normalize_prior(
            supplied_prior,
            &opponent_names,
            "side-two",
            Some(world),
            true,
        )?;
        let ordered_opponent_options = opponent_names
            .iter()
            .map(|name| opponent_by_name[name].clone())
            .collect();
        particles.push(PreparedParticle {
            state: state.clone(),
            state_string: state.serialize(),
            weight: *weight,
            own_by_name,
            opponent_names,
            opponent_options: ordered_opponent_options,
            opponent_prior,
            source_particles: vec![SharedRootSourceParticle {
                input_index: world as u32,
                input_weight: *weight,
            }],
        });
    }
    particles.sort_by(|left, right| {
        left.state_string
            .cmp(&right.state_string)
            .then_with(|| compare_optional_priors(&left.opponent_prior, &right.opponent_prior))
    });
    let mut canonical: Vec<PreparedParticle> = Vec::with_capacity(particles.len());
    for particle in particles {
        if let Some(previous) = canonical.last_mut() {
            if previous.state_string == particle.state_string
                && previous.opponent_prior == particle.opponent_prior
            {
                previous.source_particles.extend(particle.source_particles);
                continue;
            }
        }
        canonical.push(particle);
    }
    for particle in &mut canonical {
        particle
            .source_particles
            .sort_by_key(|source| source.input_index);
        let mut component_weights: Vec<_> = particle
            .source_particles
            .iter()
            .map(|source| source.input_weight)
            .collect();
        component_weights.sort_by(f64::total_cmp);
        particle.weight = component_weights.into_iter().sum();
    }
    let total: f64 = canonical.iter().map(|particle| particle.weight).sum();
    for particle in &mut canonical {
        particle.weight /= total;
    }
    Ok((
        canonical,
        shared_names.expect("positive weights guarantee support"),
        positive_count,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn shared_information_set_root_search(
    states: &[State],
    weights: &[f64],
    config: &SharedRootConfig,
    player_prior: Option<&[(String, f64)]>,
    opponent_priors: Option<&[Option<Vec<(String, f64)>>]>,
) -> Result<SharedRootResult, SharedRootError> {
    validate_config(config)?;
    let input_particle_count = states.len() as u32;
    let (particles, shared_names, positive_particle_count) =
        prepare_particles(states, weights, opponent_priors)?;
    let normalized_player_prior =
        normalize_prior(player_prior, &shared_names, "side-one", None, true)?;
    let payoff_cells: u64 = particles
        .iter()
        .map(|particle| shared_names.len() as u64 * particle.opponent_names.len() as u64)
        .sum();
    if payoff_cells == 0 || payoff_cells > MAX_PAYOFF_CELLS {
        return Err(SharedRootError::InvalidBounds(
            "payoff matrix exceeds 16384 cells",
        ));
    }
    let total_forced_continuation_iterations =
        payoff_cells.saturating_mul(config.continuation_iterations as u64);
    if total_forced_continuation_iterations > MAX_TOTAL_CONTINUATION_ITERATIONS {
        return Err(SharedRootError::InvalidBounds(
            "total forced continuation iterations exceed 100000000",
        ));
    }

    let mut payoffs = Vec::with_capacity(particles.len());
    let mut continuation_captures = Vec::with_capacity(particles.len());
    for (world, particle) in particles.iter().enumerate() {
        let mut matrix = vec![vec![0.0; particle.opponent_options.len()]; shared_names.len()];
        let mut continuations =
            vec![vec![None; particle.opponent_options.len()]; shared_names.len()];
        for (opponent_index, opponent) in particle.opponent_options.iter().enumerate() {
            let seed = row_seed(
                config.seed,
                &particle.state_string,
                &particle.opponent_names[opponent_index],
            );
            for (own_index, own_name) in shared_names.iter().enumerate() {
                let own = particle
                    .own_by_name
                    .get(own_name)
                    .expect("shared support was validated")
                    .clone();
                let mut continuation = particle.state.clone();
                let result = perform_mcts_seeded(
                    &mut continuation,
                    vec![own],
                    vec![opponent.clone()],
                    config.continuation_iterations,
                    seed,
                );
                let Some(root) = result.s1.first() else {
                    return Err(SharedRootError::ContinuationFailed {
                        world,
                        opponent_action: particle.opponent_names[opponent_index].clone(),
                    });
                };
                if root.visits == 0 {
                    return Err(SharedRootError::ContinuationFailed {
                        world,
                        opponent_action: particle.opponent_names[opponent_index].clone(),
                    });
                }
                let value = root.total_score as f64 / root.visits as f64;
                if !value.is_finite() {
                    return Err(SharedRootError::NonFinitePayoff);
                }
                matrix[own_index][opponent_index] = value;
                continuations[own_index][opponent_index] = Some(SharedRootContinuation {
                    seed,
                    requested_iterations: config.continuation_iterations,
                    executed_iterations: result.iteration_count,
                    visits: root.visits,
                    total_score: root.total_score,
                    total_score_bits: root.total_score.to_bits(),
                    payoff: value,
                    payoff_bits: value.to_bits(),
                });
            }
        }
        payoffs.push(matrix);
        continuation_captures.push(
            continuations
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|cell| cell.expect("every payoff cell was evaluated"))
                        .collect()
                })
                .collect(),
        );
    }
    let canonical_weights: Vec<_> = particles.iter().map(|particle| particle.weight).collect();
    let canonical_opponent_priors: Vec<_> = particles
        .iter()
        .map(|particle| particle.opponent_prior.clone())
        .collect();
    let solved = solve_weighted_matrix_rm_plus(
        &canonical_weights,
        &payoffs,
        config.iterations,
        normalized_player_prior.as_deref(),
        &canonical_opponent_priors,
        config.prior_strength,
    )?;
    let action_support_digest = digest(shared_names.iter().map(String::as_bytes));
    let particle_digest_parts: Vec<Vec<u8>> = particles
        .iter()
        .flat_map(|particle| {
            [
                particle.state_string.as_bytes().to_vec(),
                particle.weight.to_le_bytes().to_vec(),
            ]
        })
        .collect();
    let particle_digest = digest(&particle_digest_parts);
    let payoff_digest = digest(
        payoffs
            .iter()
            .flat_map(|matrix| matrix.iter())
            .flat_map(|row| row.iter())
            .map(|value| value.to_le_bytes()),
    );
    let player_prior_digest_parts: Vec<Vec<u8>> = normalized_player_prior
        .as_ref()
        .map(|prior| {
            prior
                .iter()
                .map(|value| value.to_le_bytes().to_vec())
                .collect()
        })
        .unwrap_or_else(|| vec![b"none".to_vec()]);
    let player_prior_digest = digest(&player_prior_digest_parts);
    let opponent_prior_digest_parts: Vec<Vec<u8>> = particles
        .iter()
        .flat_map(|particle| {
            particle
                .opponent_prior
                .as_ref()
                .map(|prior| {
                    prior
                        .iter()
                        .map(|value| value.to_le_bytes().to_vec())
                        .collect()
                })
                .unwrap_or_else(|| vec![b"none".to_vec()])
        })
        .collect();
    let opponent_prior_digest = digest(&opponent_prior_digest_parts);
    let normalized_weight_sum: f64 = canonical_weights.iter().sum();
    let opponent_policies: Vec<Vec<(String, f64)>> = particles
        .iter()
        .zip(&solved.opponent_policies)
        .map(|(particle, policy)| {
            particle
                .opponent_names
                .iter()
                .cloned()
                .zip(policy.iter().copied())
                .collect()
        })
        .collect();
    let replay_capture = SharedRootReplayCapture {
        schema_version: 1,
        solver_contract: SHARED_ROOT_SOLVER_CONTRACT.to_owned(),
        configuration: config.clone(),
        own_action_support: shared_names.clone(),
        normalized_player_prior: normalized_player_prior.clone(),
        canonical_particles: particles
            .iter()
            .zip(payoffs)
            .zip(continuation_captures)
            .zip(&solved.opponent_policies)
            .enumerate()
            .map(
                |(
                    canonical_index,
                    (((particle, payoff_matrix), continuations), opponent_policy),
                )| {
                    SharedRootReplayParticle {
                        canonical_index: canonical_index as u32,
                        state: particle.state_string.clone(),
                        normalized_weight: particle.weight,
                        source_particles: particle.source_particles.clone(),
                        opponent_action_support: particle.opponent_names.clone(),
                        normalized_opponent_prior: particle.opponent_prior.clone(),
                        payoff_matrix,
                        continuations,
                        opponent_policy: opponent_policy.clone(),
                    }
                },
            )
            .collect(),
    };
    let policy = shared_names
        .into_iter()
        .zip(solved.player_policy.iter().copied())
        .zip(solved.counterfactual_values.iter().copied())
        .map(
            |((action, probability), counterfactual_value)| SharedRootAction {
                action,
                probability,
                counterfactual_value,
            },
        )
        .collect();

    Ok(SharedRootResult {
        policy,
        opponent_policies,
        replay_capture,
        diagnostics: SharedRootDiagnostics {
            solver_contract: SHARED_ROOT_SOLVER_CONTRACT.to_owned(),
            iterations: config.iterations,
            continuation_iterations: config.continuation_iterations,
            seed: config.seed,
            prior_strength: config.prior_strength,
            expected_value: solved.expected_value,
            player_best_response_value: solved.player_best_response_value,
            opponent_best_response_value: solved.opponent_best_response_value,
            player_best_response_gain: solved.player_best_response_gain,
            opponent_best_response_gain: solved.opponent_best_response_gain,
            nash_conv: solved.nash_conv,
            exploitability: solved.exploitability,
            player_regret_bound: solved.player_regret_bound,
            opponent_regret_bound: solved.opponent_regret_bound,
            total_regret_bound: solved.total_regret_bound,
            payoff_cells,
            total_forced_continuation_iterations,
            input_particle_count,
            positive_particle_count,
            canonical_particle_count: particles.len() as u32,
            normalized_weight_sum,
            action_support_digest,
            particle_digest,
            payoff_digest,
            player_prior_digest,
            opponent_prior_digest,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choices::Choices;
    use crate::state::{PokemonIndex, PokemonMoveIndex};

    fn solve(payoffs: Vec<Vec<Vec<f64>>>, iterations: u32) -> MatrixSolveResult {
        solve_weighted_matrix_rm_plus(&[1.0], &payoffs, iterations, None, &[None], 0.0).unwrap()
    }

    fn assert_policy_close(policy: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(policy.len(), expected.len());
        for (actual, expected) in policy.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{:?} != {:?}",
                policy,
                expected
            );
        }
    }

    #[test]
    fn matching_pennies_converges_and_reports_low_exploitability() {
        let result = solve(vec![vec![vec![1.0, -1.0], vec![-1.0, 1.0]]], 50_000);
        assert_policy_close(&result.player_policy, &[0.5, 0.5], 0.02);
        assert_policy_close(&result.opponent_policies[0], &[0.5, 0.5], 0.02);
        assert!(result.exploitability < 0.02, "{:?}", result);
    }

    #[test]
    fn rock_paper_scissors_converges_to_uniform() {
        let result = solve(
            vec![vec![
                vec![0.0, -1.0, 1.0],
                vec![1.0, 0.0, -1.0],
                vec![-1.0, 1.0, 0.0],
            ]],
            100_000,
        );
        assert_policy_close(&result.player_policy, &[1.0 / 3.0; 3], 0.02);
        assert!(result.exploitability < 0.02);
    }

    #[test]
    fn dominated_action_loses_mass() {
        let result = solve(
            vec![vec![
                vec![0.0, -1.0, 1.0],
                vec![1.0, 0.0, -1.0],
                vec![-1.0, 1.0, 0.0],
                vec![-2.0; 3],
            ]],
            100_000,
        );
        assert!(result.player_policy[3] < 1e-3, "{:?}", result);
    }

    #[test]
    fn pure_saddle_converges_to_the_correct_pair() {
        let result = solve(vec![vec![vec![3.0, 1.0], vec![4.0, 2.0]]], 10_000);
        assert!(result.player_policy[1] > 0.999);
        assert!(result.opponent_policies[0][1] > 0.999);
        assert!(result.exploitability < 1e-3);
    }

    #[test]
    fn exploitability_and_regret_decline_with_iterations() {
        let game = vec![vec![vec![3.0, 1.0], vec![4.0, 2.0]]];
        let short = solve(game.clone(), 100);
        let long = solve(game, 100_000);
        assert!(
            long.exploitability < short.exploitability,
            "short={:?} long={:?}",
            short,
            long
        );
        assert!(long.total_regret_bound < short.total_regret_bound);
    }

    #[test]
    fn counterfactual_and_best_response_values_are_exact() {
        let result = solve_weighted_matrix_rm_plus(
            &[0.25, 0.75],
            &[vec![vec![1.0], vec![0.0]], vec![vec![-1.0], vec![2.0]]],
            1_000,
            None,
            &[None, None],
            0.0,
        )
        .unwrap();
        assert!((result.counterfactual_values[0] + 0.5).abs() < 1e-12);
        assert!((result.counterfactual_values[1] - 1.5).abs() < 1e-12);
        assert!(result.player_policy[1] > 0.999);
    }

    #[test]
    fn particle_permutation_and_split_weights_are_invariant() {
        let first = vec![vec![1.0], vec![-1.0]];
        let second = vec![vec![-1.0], vec![1.0]];
        let base = solve_weighted_matrix_rm_plus(
            &[0.7, 0.3],
            &[first.clone(), second.clone()],
            10_000,
            None,
            &[None, None],
            0.0,
        )
        .unwrap();
        let permuted = solve_weighted_matrix_rm_plus(
            &[0.3, 0.7],
            &[second.clone(), first.clone()],
            10_000,
            None,
            &[None, None],
            0.0,
        )
        .unwrap();
        let split = solve_weighted_matrix_rm_plus(
            &[0.35, 0.35, 0.3],
            &[first.clone(), first, second],
            10_000,
            None,
            &[None, None, None],
            0.0,
        )
        .unwrap();
        assert_policy_close(&base.player_policy, &permuted.player_policy, 1e-12);
        assert_policy_close(&base.player_policy, &split.player_policy, 1e-12);
        assert!((base.exploitability - split.exploitability).abs() < 1e-12);
    }

    #[test]
    fn zero_weight_particle_has_no_effect() {
        let base = solve_weighted_matrix_rm_plus(
            &[1.0],
            &[vec![vec![1.0], vec![0.0]]],
            1_000,
            None,
            &[None],
            0.0,
        )
        .unwrap();
        let extra = solve_weighted_matrix_rm_plus(
            &[1.0, 0.0],
            &[vec![vec![1.0], vec![0.0]], vec![vec![-100.0], vec![100.0]]],
            1_000,
            None,
            &[None, None],
            0.0,
        )
        .unwrap();
        assert_eq!(base.player_policy, extra.player_policy);
        assert_eq!(base.counterfactual_values, extra.counterfactual_values);
    }

    #[test]
    fn prior_is_only_a_finite_iteration_warm_start() {
        let game = vec![vec![vec![1.0], vec![0.0]]];
        let cold =
            solve_weighted_matrix_rm_plus(&[1.0], &game, 100_000, None, &[None], 0.0).unwrap();
        let warm =
            solve_weighted_matrix_rm_plus(&[1.0], &game, 100_000, Some(&[0.0, 1.0]), &[None], 10.0)
                .unwrap();
        assert!(cold.player_policy[0] > 0.999);
        assert!(warm.player_policy[0] > 0.999);
    }

    fn state_with_moves() -> State {
        let mut state = State::default();
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::TACKLE);
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M1, Choices::SPLASH);
        state
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::TACKLE);
        state
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M1, Choices::SPLASH);
        for side in [&mut state.side_one, &mut state.side_two] {
            side.get_active().moves.m2.disabled = true;
            side.get_active().moves.m3.disabled = true;
            for index in [
                PokemonIndex::P1,
                PokemonIndex::P2,
                PokemonIndex::P3,
                PokemonIndex::P4,
                PokemonIndex::P5,
            ] {
                side.pokemon[index].hp = 0;
            }
        }
        state.s1_can_tera = false;
        state.s2_can_tera = false;
        state
    }

    #[test]
    fn pokemon_search_is_reproducible_nonmutating_and_normalized() {
        let state = state_with_moves();
        let before = state.serialize();
        let config = SharedRootConfig {
            iterations: 2_000,
            continuation_iterations: 32,
            seed: 19,
            prior_strength: 0.0,
        };
        let first =
            shared_information_set_root_search(&[state.clone()], &[1.0], &config, None, None)
                .unwrap();
        let second =
            shared_information_set_root_search(&[state.clone()], &[1.0], &config, None, None)
                .unwrap();
        assert_eq!(first, second);
        assert_eq!(state.serialize(), before);
        assert!(
            (first
                .policy
                .iter()
                .map(|entry| entry.probability)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-12
        );
        assert!(first
            .policy
            .iter()
            .all(|entry| entry.probability.is_finite() && entry.probability >= 0.0));
        assert!(first
            .policy
            .iter()
            .all(|entry| entry.action == "tackle" || entry.action == "splash"));
    }

    #[test]
    fn priors_are_conditioned_on_engine_support() {
        let state = state_with_moves();
        let config = SharedRootConfig {
            iterations: 1_000,
            continuation_iterations: 8,
            seed: 21,
            prior_strength: 1.0,
        };
        let unanchored =
            shared_information_set_root_search(&[state.clone()], &[1.0], &config, None, None)
                .unwrap();
        let unmatched = shared_information_set_root_search(
            &[state],
            &[1.0],
            &config,
            None,
            Some(&[Some(vec![("missing".to_owned(), 1.0)])]),
        )
        .unwrap();
        assert_eq!(unanchored, unmatched);
        let unmatched_player = shared_information_set_root_search(
            &[state_with_moves()],
            &[1.0],
            &config,
            Some(&[("missing".to_owned(), 1.0)]),
            None,
        )
        .unwrap();
        assert_eq!(unanchored, unmatched_player);
    }

    #[test]
    fn pokemon_particles_merge_ignore_zero_weight_and_preserve_policy() {
        let state = state_with_moves();
        let mut irrelevant = State::default();
        irrelevant
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::EMBER);
        irrelevant
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::EMBER);
        let config = SharedRootConfig {
            iterations: 1_000,
            continuation_iterations: 16,
            seed: 23,
            prior_strength: 0.0,
        };
        let single =
            shared_information_set_root_search(&[state.clone()], &[1.0], &config, None, None)
                .unwrap();
        let split = shared_information_set_root_search(
            &[state.clone(), state.clone()],
            &[0.4, 0.6],
            &config,
            None,
            None,
        )
        .unwrap();
        let zero = shared_information_set_root_search(
            &[state, irrelevant],
            &[1.0, 0.0],
            &config,
            None,
            None,
        )
        .unwrap();
        assert_eq!(single.policy, split.policy);
        assert_eq!(single.policy, zero.policy);
        assert_eq!(split.diagnostics.canonical_particle_count, 1);
        assert_eq!(zero.diagnostics.positive_particle_count, 1);
        let capture = &split.replay_capture;
        assert_eq!(capture.schema_version, 1);
        assert_eq!(capture.configuration, config);
        assert_eq!(capture.canonical_particles.len(), 1);
        let particle = &capture.canonical_particles[0];
        assert_eq!(particle.canonical_index, 0);
        assert_eq!(particle.normalized_weight, 1.0);
        assert_eq!(
            particle.source_particles,
            vec![
                SharedRootSourceParticle {
                    input_index: 0,
                    input_weight: 0.4,
                },
                SharedRootSourceParticle {
                    input_index: 1,
                    input_weight: 0.6,
                },
            ]
        );
        assert_eq!(
            particle.payoff_matrix.len(),
            capture.own_action_support.len()
        );
        assert_eq!(
            particle.continuations.len(),
            capture.own_action_support.len()
        );
        assert_eq!(
            particle.opponent_policy.len(),
            particle.opponent_action_support.len()
        );
        for (payoff_row, continuation_row) in
            particle.payoff_matrix.iter().zip(&particle.continuations)
        {
            assert_eq!(payoff_row.len(), particle.opponent_action_support.len());
            assert_eq!(
                continuation_row.len(),
                particle.opponent_action_support.len()
            );
            for (payoff, continuation) in payoff_row.iter().zip(continuation_row) {
                assert_eq!(*payoff, continuation.payoff);
                assert_eq!(payoff.to_bits(), continuation.payoff_bits);
                assert_eq!(
                    continuation.total_score.to_bits(),
                    continuation.total_score_bits
                );
                assert_eq!(
                    continuation.requested_iterations,
                    config.continuation_iterations
                );
                assert_eq!(
                    continuation.executed_iterations,
                    config.continuation_iterations
                );
                assert_eq!(continuation.visits, config.continuation_iterations);
            }
        }
        assert_eq!(
            zero.replay_capture.canonical_particles[0].source_particles,
            vec![SharedRootSourceParticle {
                input_index: 0,
                input_weight: 1.0,
            }]
        );
    }

    #[test]
    fn pokemon_particle_permutation_is_exactly_invariant() {
        let first = state_with_moves();
        let mut second = first.clone();
        second.side_two.get_active().hp -= 1;
        let config = SharedRootConfig {
            iterations: 1_000,
            continuation_iterations: 8,
            seed: 29,
            prior_strength: 0.0,
        };
        let forward = shared_information_set_root_search(
            &[first.clone(), second.clone()],
            &[0.4, 0.6],
            &config,
            None,
            None,
        )
        .unwrap();
        let reverse =
            shared_information_set_root_search(&[second, first], &[0.6, 0.4], &config, None, None)
                .unwrap();

        assert_eq!(forward.policy, reverse.policy);
        assert_eq!(forward.opponent_policies, reverse.opponent_policies);
        assert_eq!(forward.diagnostics, reverse.diagnostics);
        assert_eq!(
            forward.replay_capture.own_action_support,
            reverse.replay_capture.own_action_support
        );
        for (mut forward_particle, mut reverse_particle) in forward
            .replay_capture
            .canonical_particles
            .into_iter()
            .zip(reverse.replay_capture.canonical_particles)
        {
            let mut forward_sources = std::mem::take(&mut forward_particle.source_particles);
            let mut reverse_sources = std::mem::take(&mut reverse_particle.source_particles);
            forward_sources.sort_by(|left, right| left.input_weight.total_cmp(&right.input_weight));
            reverse_sources.sort_by(|left, right| left.input_weight.total_cmp(&right.input_weight));
            assert_eq!(forward_particle, reverse_particle);
            assert_eq!(
                forward_sources
                    .iter()
                    .map(|source| source.input_weight)
                    .collect::<Vec<_>>(),
                reverse_sources
                    .iter()
                    .map(|source| source.input_weight)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn forced_switch_roots_return_only_legal_normalized_actions() {
        let mut forced = state_with_moves();
        forced.side_one.force_switch = true;
        forced.side_one.pokemon[PokemonIndex::P1].hp = 100;
        let config = SharedRootConfig {
            iterations: 100,
            continuation_iterations: 4,
            seed: 31,
            prior_strength: 0.0,
        };
        let forced_result =
            shared_information_set_root_search(&[forced], &[1.0], &config, None, None).unwrap();
        assert_eq!(forced_result.policy.len(), 1);
        assert!(forced_result.policy[0].action.starts_with("switch "));
        assert_eq!(forced_result.policy[0].probability, 1.0);
    }

    #[cfg(feature = "terastallization")]
    #[test]
    fn tera_roots_return_only_legal_normalized_actions() {
        let config = SharedRootConfig {
            iterations: 100,
            continuation_iterations: 4,
            seed: 31,
            prior_strength: 0.0,
        };
        let mut tera = state_with_moves();
        tera.s1_can_tera = true;
        let tera_result =
            shared_information_set_root_search(&[tera], &[1.0], &config, None, None).unwrap();
        assert!(tera_result
            .policy
            .iter()
            .any(|entry| entry.action.ends_with("-tera")));
        assert!(
            (tera_result
                .policy
                .iter()
                .map(|entry| entry.probability)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn support_mismatch_and_invalid_inputs_fail_closed() {
        let first = state_with_moves();
        let mut second = state_with_moves();
        second
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M1, Choices::EMBER);
        let config = SharedRootConfig {
            iterations: 10,
            continuation_iterations: 1,
            seed: 0,
            prior_strength: 0.0,
        };
        assert!(matches!(
            shared_information_set_root_search(&[first, second], &[0.5, 0.5], &config, None, None),
            Err(SharedRootError::SharedSupportMismatch { .. })
        ));
        assert!(
            solve_weighted_matrix_rm_plus(&[0.9], &[vec![vec![0.0]]], 10, None, &[None], 0.0)
                .is_err()
        );
    }
}
