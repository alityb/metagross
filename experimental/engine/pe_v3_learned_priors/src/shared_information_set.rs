use crate::engine::state::MoveChoice;
use crate::mcts::perform_mcts;
use crate::state::{Side, State};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct SharedRootAction {
    pub action: String,
    pub probability: f32,
    pub pulls: u32,
    pub estimated_value: f32,
}

#[derive(Debug, Clone)]
pub struct SharedRootDiagnostics {
    pub rounds: u32,
    pub continuation_iterations: u32,
    pub unique_payoff_cells_evaluated: u64,
    pub cache_hits: u64,
    pub total_forced_continuation_iterations: u64,
    pub world_pulls: Vec<u32>,
    pub elapsed_ms: u64,
    pub shared_policy_entropy: f32,
    pub shared_policy_max_probability: f32,
    pub human_prior_mix: f32,
    pub player_prior_mix: f32,
    pub player_prior_available: bool,
    pub player_prior_coverage: f32,
    pub baseline_action: Option<String>,
    pub baseline_advantage_available: bool,
    pub baseline_advantage_mean: Option<f32>,
    pub baseline_advantage_standard_error: Option<f32>,
    pub baseline_advantage_lcb: Option<f32>,
    pub baseline_advantage_world_count: u32,
    pub baseline_advantage_effective_world_count: f32,
    pub lcb_z: f32,
    pub paired_evaluation_iterations: u32,
}

#[derive(Debug, Clone)]
pub struct SharedRootResult {
    pub policy: Vec<SharedRootAction>,
    pub diagnostics: SharedRootDiagnostics,
}

#[derive(Debug)]
pub struct OracleResult {
    pub probabilities: Vec<f32>,
    pub opponent_probabilities: Vec<Vec<f32>>,
    pub estimated_values: Vec<f32>,
    pub rounds: u32,
    pub world_pulls: Vec<u32>,
    pub unique_payoff_cells_evaluated: u64,
    pub cache_hits: u64,
    pub baseline_advantage_available: bool,
    pub baseline_advantage_mean: Option<f32>,
    pub baseline_advantage_standard_error: Option<f32>,
    pub baseline_advantage_lcb: Option<f32>,
    pub baseline_advantage_world_count: u32,
    pub baseline_advantage_effective_world_count: f32,
}

#[derive(Debug, Default)]
struct PairedAdvantage {
    available: bool,
    mean: Option<f64>,
    standard_error: Option<f64>,
    lcb: Option<f64>,
    world_count: u32,
    effective_world_count: f64,
}

/// Computes a paired across-world diagnostic from one empirical payoff table.
///
/// These cells are biased cached MCTS estimates in production. The resulting
/// LCB is an empirical selection heuristic, not a confidence guarantee.
fn paired_advantage_from_cache(
    action_count: usize,
    world_weights: &[f64],
    opponent_probabilities: &[Vec<f32>],
    player_probabilities: &[f64],
    baseline_action: usize,
    lcb_z: f64,
    payoff_cache: &HashMap<(usize, usize, usize), f64>,
) -> Option<PairedAdvantage> {
    let weight_total: f64 = world_weights.iter().sum();
    let normalized_weights: Vec<f64> = world_weights
        .iter()
        .map(|weight| weight / weight_total)
        .collect();
    let mut differences = Vec::new();

    for (world, weight) in normalized_weights.iter().copied().enumerate() {
        if weight == 0.0 {
            continue;
        }
        let mut shared_value = 0.0;
        let mut baseline_value = 0.0;
        for (opponent_action, probability) in opponent_probabilities[world]
            .iter()
            .map(|probability| *probability as f64)
            .enumerate()
        {
            if probability == 0.0 {
                continue;
            }
            baseline_value +=
                probability * payoff_cache.get(&(world, opponent_action, baseline_action))?;
            for (action, player_probability) in player_probabilities
                .iter()
                .copied()
                .enumerate()
                .take(action_count)
            {
                if player_probability > 0.0 {
                    shared_value += probability
                        * player_probability
                        * payoff_cache.get(&(world, opponent_action, action))?;
                }
            }
        }
        differences.push((weight, shared_value - baseline_value));
    }

    let mean: f64 = differences
        .iter()
        .map(|(weight, difference)| weight * difference)
        .sum();
    let squared_weight_sum: f64 = differences.iter().map(|(weight, _)| weight * weight).sum();
    let effective_world_count = 1.0 / squared_weight_sum;
    let standard_error = if differences.len() <= 1 || 1.0 - squared_weight_sum <= f64::EPSILON {
        0.0
    } else {
        let weighted_squared_deviation: f64 = differences
            .iter()
            .map(|(weight, difference)| weight * (difference - mean).powi(2))
            .sum();
        let unbiased_variance = weighted_squared_deviation / (1.0 - squared_weight_sum);
        (unbiased_variance / effective_world_count).sqrt()
    };

    Some(PairedAdvantage {
        available: true,
        mean: Some(mean),
        standard_error: Some(standard_error),
        lcb: Some(mean - lcb_z * standard_error),
        world_count: differences.len() as u32,
        effective_world_count,
    })
}

fn strategy(regrets: &[f64]) -> Vec<f64> {
    let total: f64 = regrets.iter().sum();
    if total > 0.0 && total.is_finite() {
        regrets.iter().map(|regret| regret / total).collect()
    } else {
        vec![1.0 / regrets.len() as f64; regrets.len()]
    }
}

fn sample_strategy(policy: &[f64], rng: &mut StdRng) -> Result<usize, String> {
    WeightedIndex::new(policy)
        .map_err(|error| format!("invalid RM+ strategy: {}", error))
        .map(|distribution| distribution.sample(rng))
}

fn prune_policy(mut policy: Vec<f64>, minimum: f64) -> Vec<f64> {
    let mut top = 0;
    for index in 1..policy.len() {
        if policy[index] > policy[top] {
            top = index;
        }
    }
    for probability in &mut policy {
        if *probability < minimum {
            *probability = 0.0;
        }
    }
    if policy.iter().sum::<f64>() == 0.0 {
        policy[top] = 1.0;
        return policy;
    }
    let total = policy.iter().sum::<f64>();
    policy
        .iter_mut()
        .for_each(|probability| *probability /= total);
    policy
}

/// Generic shared-root RM+ loop. The oracle receives world, opponent-action,
/// and shared-action indices, making the regret logic independently testable.
pub fn solve_shared_root_with_oracle<F>(
    action_count: usize,
    world_weights: &[f64],
    opponent_action_counts: &[usize],
    opponent_priors: &[Option<Vec<f64>>],
    human_prior_mix: f64,
    player_prior: Option<&[f64]>,
    player_prior_mix: f64,
    min_policy_probability: f64,
    duration: Duration,
    round_limit: u32,
    seed: u64,
    baseline_action: Option<usize>,
    lcb_z: f64,
    continuation_iterations: u32,
    paired_evaluation_iterations: u32,
    mut oracle: F,
) -> Result<OracleResult, String>
where
    F: FnMut(usize, usize, usize, u32) -> Result<f32, String>,
{
    if action_count == 0 {
        return Err("shared action support must not be empty".to_string());
    }
    if world_weights.is_empty()
        || world_weights.len() != opponent_priors.len()
        || world_weights.len() != opponent_action_counts.len()
    {
        return Err("states, world_weights, and s2_priors must have matching lengths".to_string());
    }
    if round_limit == 0 && duration.is_zero() {
        return Err("duration_ms or rounds must be greater than zero".to_string());
    }
    if !human_prior_mix.is_finite() || !(0.0..=1.0).contains(&human_prior_mix) {
        return Err("human_prior_mix must be finite and in [0, 1]".to_string());
    }
    if !player_prior_mix.is_finite() || !(0.0..=1.0).contains(&player_prior_mix) {
        return Err("player_prior_mix must be finite and in [0, 1]".to_string());
    }
    if !min_policy_probability.is_finite() || !(0.0..=1.0).contains(&min_policy_probability) {
        return Err("min_policy_probability must be finite and in [0, 1]".to_string());
    }
    if baseline_action.is_some_and(|action| action >= action_count) {
        return Err("baseline action must be in shared action support".to_string());
    }
    if !lcb_z.is_finite() || lcb_z < 0.0 {
        return Err("lcb_z must be finite and nonnegative".to_string());
    }
    if continuation_iterations == 0 {
        return Err("continuation_iterations must be greater than zero".to_string());
    }
    if paired_evaluation_iterations == 0 {
        return Err("paired_evaluation_iterations must be greater than zero".to_string());
    }
    if world_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err("world weights must be finite and nonnegative".to_string());
    }
    if world_weights.iter().sum::<f64>() <= 0.0 {
        return Err("at least one world weight must be positive".to_string());
    }
    if opponent_action_counts.iter().any(|count| *count == 0) {
        return Err("every world must have at least one side-two action".to_string());
    }
    for (world, prior) in opponent_priors.iter().enumerate() {
        let Some(prior) = prior else { continue };
        if prior.len() != opponent_action_counts[world] {
            return Err("side-two prior length must match its world's action count".to_string());
        }
        if prior.is_empty()
            || prior
                .iter()
                .any(|probability| !probability.is_finite() || *probability < 0.0)
            || prior.iter().sum::<f64>() <= 0.0
        {
            return Err(
                "side-two priors must be finite, nonnegative, and have positive mass".to_string(),
            );
        }
    }
    if let Some(prior) = player_prior {
        if prior.len() != action_count
            || prior
                .iter()
                .any(|probability| !probability.is_finite() || *probability < 0.0)
            || prior.iter().sum::<f64>() <= 0.0
        {
            return Err(
                "side-one prior must match the shared action count and contain finite, nonnegative values with positive mass"
                    .to_string(),
            );
        }
    }

    let world_distribution = WeightedIndex::new(world_weights)
        .map_err(|error| format!("invalid world weights: {}", error))?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut player_regrets = vec![0.0_f64; action_count];
    let mut opponent_regrets: Vec<Vec<f64>> = opponent_action_counts
        .iter()
        .map(|count| vec![0.0; *count])
        .collect();
    let mut strategy_sum = vec![0.0_f64; action_count];
    let mut opponent_strategy_sums: Vec<Vec<f64>> = opponent_action_counts
        .iter()
        .map(|count| vec![0.0; *count])
        .collect();
    let mut value_sum = vec![0.0_f64; action_count];
    let mut world_pulls = vec![0_u32; world_weights.len()];
    let mut payoff_cache = HashMap::new();
    let mut cache_hits = 0_u64;
    let mut rounds = 0_u32;
    let start = Instant::now();

    let mut payoff = |world, opponent_action, action| -> Result<f64, String> {
        let key = (world, opponent_action, action);
        if let Some(value) = payoff_cache.get(&key) {
            cache_hits += 1;
            return Ok(*value);
        }
        let value = oracle(world, opponent_action, action, continuation_iterations)?;
        if !value.is_finite() {
            return Err("payoff oracle returned a nonfinite value".to_string());
        }
        let value = value as f64;
        payoff_cache.insert(key, value);
        Ok(value)
    };

    while (round_limit > 0 && rounds < round_limit)
        || (round_limit == 0 && (rounds == 0 || start.elapsed() < duration))
    {
        let policy = strategy(&player_regrets);
        let behavior = match player_prior {
            Some(prior) => policy
                .iter()
                .zip(prior)
                .map(|(rm, human)| (1.0 - player_prior_mix) * rm + player_prior_mix * human)
                .collect::<Vec<_>>(),
            None => policy.clone(),
        };
        for (sum, probability) in strategy_sum.iter_mut().zip(&behavior) {
            *sum += probability;
        }

        let world = world_distribution.sample(&mut rng);
        world_pulls[world] += 1;
        let opponent_policy = strategy(&opponent_regrets[world]);
        let opponent_behavior = match &opponent_priors[world] {
            Some(prior) => opponent_policy
                .iter()
                .zip(prior)
                .map(|(rm, human)| (1.0 - human_prior_mix) * rm + human_prior_mix * human)
                .collect::<Vec<_>>(),
            None => opponent_policy.clone(),
        };
        for (sum, probability) in opponent_strategy_sums[world]
            .iter_mut()
            .zip(&opponent_behavior)
        {
            *sum += probability;
        }
        let opponent_action = sample_strategy(&opponent_behavior, &mut rng)?;
        let mut values = Vec::with_capacity(action_count);
        for action in 0..action_count {
            let value = payoff(world, opponent_action, action)?;
            values.push(value);
            value_sum[action] += value;
        }
        let expected: f64 = behavior
            .iter()
            .zip(&values)
            .map(|(probability, value)| probability * value)
            .sum();
        for (regret, value) in player_regrets.iter_mut().zip(&values) {
            *regret = (*regret + value - expected).max(0.0);
        }

        let player_action = sample_strategy(&behavior, &mut rng)?;
        let mut opponent_values = Vec::with_capacity(opponent_action_counts[world]);
        for action in 0..opponent_action_counts[world] {
            let value = if action == opponent_action {
                values[player_action]
            } else {
                payoff(world, action, player_action)?
            };
            opponent_values.push(value);
        }
        let opponent_expected: f64 = opponent_policy
            .iter()
            .zip(&opponent_values)
            .map(|(probability, value)| probability * value)
            .sum();
        for (regret, value) in opponent_regrets[world].iter_mut().zip(opponent_values) {
            *regret = (*regret + opponent_expected - value).max(0.0);
        }
        rounds += 1;
    }
    drop(payoff);

    let denominator = rounds as f64;
    let probabilities = prune_policy(
        strategy_sum
            .into_iter()
            .map(|sum| sum / denominator)
            .collect(),
        min_policy_probability,
    );
    let opponent_probabilities: Vec<Vec<f32>> = opponent_strategy_sums
        .into_iter()
        .zip(&world_pulls)
        .map(|(sums, pulls)| {
            let denominator = (*pulls).max(1) as f64;
            sums.into_iter()
                .map(|sum| (sum / denominator) as f32)
                .collect()
        })
        .collect();
    let mut paired_advantage = PairedAdvantage::default();
    if let Some(baseline_action) = baseline_action {
        let positive_world_count = world_weights.iter().filter(|weight| **weight > 0.0).count();
        let weight_total: f64 = world_weights.iter().sum();
        let squared_weight_sum: f64 = world_weights
            .iter()
            .filter(|weight| **weight > 0.0)
            .map(|weight| (weight / weight_total).powi(2))
            .sum();
        paired_advantage.world_count = positive_world_count as u32;
        paired_advantage.effective_world_count = 1.0 / squared_weight_sum;

        let averages_exist = world_weights
            .iter()
            .zip(&world_pulls)
            .all(|(weight, pulls)| *weight == 0.0 || *pulls > 0);
        let mut complete = averages_exist;
        let mut paired_payoff_cache: HashMap<(usize, usize, usize), f64> = HashMap::new();
        'worlds: for (world, weight) in world_weights.iter().enumerate() {
            if *weight == 0.0 || !complete {
                continue;
            }
            for (opponent_action, opponent_probability) in
                opponent_probabilities[world].iter().enumerate()
            {
                if *opponent_probability == 0.0 {
                    continue;
                }
                for action in 0..action_count {
                    if action != baseline_action && probabilities[action] == 0.0 {
                        continue;
                    }
                    let key = (world, opponent_action, action);
                    if paired_payoff_cache.contains_key(&key) {
                        continue;
                    }
                    if round_limit == 0 && start.elapsed() >= duration {
                        complete = false;
                        break 'worlds;
                    }
                    let value =
                        oracle(world, opponent_action, action, paired_evaluation_iterations)?;
                    if !value.is_finite() {
                        return Err("payoff oracle returned a nonfinite value".to_string());
                    }
                    paired_payoff_cache.insert(key, value as f64);
                }
            }
        }
        if complete {
            if let Some(result) = paired_advantage_from_cache(
                action_count,
                world_weights,
                &opponent_probabilities,
                &probabilities,
                baseline_action,
                lcb_z,
                &paired_payoff_cache,
            ) {
                paired_advantage = result;
            }
        } else {
            // Deeper paired evaluation could not complete within the deadline.
            // Fall back to the cached RM+ optimization cells so diagnostics
            // remain available, just with lower-quality estimates.
            let fallback_cache: HashMap<(usize, usize, usize), f64> = payoff_cache
                .iter()
                .map(|(key, value)| (*key, *value as f64))
                .collect();
            if let Some(result) = paired_advantage_from_cache(
                action_count,
                world_weights,
                &opponent_probabilities,
                &probabilities,
                baseline_action,
                lcb_z,
                &fallback_cache,
            ) {
                paired_advantage = result;
            }
        }
    }

    Ok(OracleResult {
        probabilities: probabilities
            .into_iter()
            .map(|probability| probability as f32)
            .collect(),
        opponent_probabilities,
        estimated_values: value_sum
            .into_iter()
            .map(|sum| (sum / denominator) as f32)
            .collect(),
        rounds,
        world_pulls,
        unique_payoff_cells_evaluated: payoff_cache.len() as u64,
        cache_hits,
        baseline_advantage_available: paired_advantage.available,
        baseline_advantage_mean: paired_advantage.mean.map(|value| value as f32),
        baseline_advantage_standard_error: paired_advantage
            .standard_error
            .map(|value| value as f32),
        baseline_advantage_lcb: paired_advantage.lcb.map(|value| value as f32),
        baseline_advantage_world_count: paired_advantage.world_count,
        baseline_advantage_effective_world_count: paired_advantage.effective_world_count as f32,
    })
}

fn canonical_action(side: &Side, choice: &MoveChoice) -> String {
    match choice {
        MoveChoice::Switch(_) => format!("switch {}", choice.to_string(side)),
        _ => choice.to_string(side),
    }
    .to_lowercase()
}

fn unique_options(
    options: Vec<MoveChoice>,
    side: &Side,
    world: usize,
    side_name: &str,
) -> Result<(Vec<String>, HashMap<String, MoveChoice>), String> {
    let mut names = Vec::with_capacity(options.len());
    let mut by_name = HashMap::with_capacity(options.len());
    for option in options {
        let name = canonical_action(side, &option);
        if by_name.insert(name.clone(), option).is_some() {
            return Err(format!(
                "world {} has ambiguous duplicate {} action '{}'",
                world, side_name, name
            ));
        }
        names.push(name);
    }
    Ok((names, by_name))
}

#[allow(clippy::too_many_arguments)]
pub fn shared_information_set_root_search(
    states: &[State],
    world_weights: &[f64],
    duration: Duration,
    rounds: u32,
    continuation_iterations: u32,
    s1_prior: Option<&[(String, f64)]>,
    player_prior_mix: f64,
    s2_priors: Option<&[Option<Vec<(String, f64)>>]>,
    human_prior_mix: f64,
    min_policy_probability: f64,
    seed: u64,
    baseline_action: Option<&str>,
    lcb_z: f64,
    paired_evaluation_iterations: u32,
) -> Result<SharedRootResult, String> {
    if states.len() != world_weights.len() || states.is_empty() {
        return Err(
            "states and world_weights must be nonempty and have matching lengths".to_string(),
        );
    }
    if world_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err("world weights must be finite and nonnegative".to_string());
    }
    if world_weights.iter().sum::<f64>() <= 0.0 {
        return Err("at least one world weight must be positive".to_string());
    }
    if continuation_iterations == 0 {
        return Err("continuation_iterations must be greater than zero".to_string());
    }
    if paired_evaluation_iterations == 0 {
        return Err("paired_evaluation_iterations must be greater than zero".to_string());
    }
    if !player_prior_mix.is_finite() || !(0.0..=1.0).contains(&player_prior_mix) {
        return Err("player_prior_mix must be finite and in [0, 1]".to_string());
    }
    if !lcb_z.is_finite() || lcb_z < 0.0 {
        return Err("lcb_z must be finite and nonnegative".to_string());
    }
    if let Some(prior) = s1_prior {
        if prior
            .iter()
            .any(|(_, probability)| !probability.is_finite() || *probability < 0.0)
        {
            return Err("side-one prior values must be finite and nonnegative".to_string());
        }
    }
    if let Some(priors) = s2_priors {
        if priors.len() != states.len() {
            return Err("s2_priors must contain one entry per world".to_string());
        }
    }

    let mut s1_maps = Vec::with_capacity(states.len());
    let mut s2_options_by_world = Vec::with_capacity(states.len());
    let mut normalized_s2_priors = Vec::with_capacity(states.len());
    let mut shared_names: Option<Vec<String>> = None;
    let mut shared_support: Option<HashSet<String>> = None;

    for (world, state) in states.iter().enumerate() {
        let option_state = state.clone();
        let (s1_options, s2_options) = option_state.root_get_all_options();
        let (s1_names, s1_map) = unique_options(s1_options, &state.side_one, world, "side-one")?;
        let (s2_names, s2_map) = unique_options(s2_options, &state.side_two, world, "side-two")?;
        if world_weights[world] > 0.0 {
            let support: HashSet<String> = s1_names.iter().cloned().collect();
            if let Some(expected) = &shared_support {
                if expected != &support {
                    return Err(format!(
                        "positive world {} has different side-one action support",
                        world
                    ));
                }
            } else {
                shared_names = Some(s1_names);
                shared_support = Some(support);
            }
        }

        let supplied_prior = s2_priors.and_then(|all| all[world].as_ref());
        let normalized = match supplied_prior {
            None => None,
            Some(pairs) => {
                let mut seen = HashSet::new();
                let mut values = vec![0.0; s2_names.len()];
                for (name, probability) in pairs {
                    if !probability.is_finite() || *probability < 0.0 {
                        return Err(format!(
                            "world {} side-two prior values must be finite and nonnegative",
                            world
                        ));
                    }
                    let name = name.to_lowercase();
                    if !seen.insert(name.clone()) {
                        return Err(format!(
                            "world {} has duplicate prior action '{}'",
                            world, name
                        ));
                    }
                    if let Some(index) = s2_names.iter().position(|candidate| candidate == &name) {
                        values[index] = *probability;
                    }
                }
                let total: f64 = values.iter().sum();
                if !total.is_finite() {
                    return Err(format!("world {} side-two prior mass is nonfinite", world));
                }
                if total <= 0.0 {
                    None
                } else {
                    values.iter_mut().for_each(|value| *value /= total);
                    Some(values)
                }
            }
        };
        s1_maps.push(s1_map);
        s2_options_by_world.push(
            s2_names
                .iter()
                .map(|name| s2_map.get(name).expect("option map is complete").clone())
                .collect::<Vec<_>>(),
        );
        normalized_s2_priors.push(normalized);
    }

    let shared_names =
        shared_names.ok_or_else(|| "at least one world weight must be positive".to_string())?;
    let baseline_action = baseline_action
        .map(str::to_lowercase)
        .map(|name| {
            shared_names
                .iter()
                .position(|candidate| candidate == &name)
                .map(|index| (name, index))
                .ok_or_else(|| "baseline action must be in shared action support".to_string())
        })
        .transpose()?;
    let (normalized_s1_prior, player_prior_coverage) = match s1_prior {
        None => (None, 0.0),
        Some(pairs) => {
            let mut seen = HashSet::new();
            let mut values = vec![0.0; shared_names.len()];
            let mut positive_legal = 0_usize;
            for (name, probability) in pairs {
                let name = name.to_lowercase();
                if !seen.insert(name.clone()) {
                    return Err(format!("duplicate side-one prior action '{}'", name));
                }
                if let Some(index) = shared_names.iter().position(|candidate| candidate == &name) {
                    values[index] = *probability;
                    if *probability > 0.0 {
                        positive_legal += 1;
                    }
                }
            }
            let total: f64 = values.iter().sum();
            if total > 0.0 {
                values.iter_mut().for_each(|value| *value /= total);
                (
                    Some(values),
                    positive_legal as f64 / shared_names.len() as f64,
                )
            } else {
                (None, 0.0)
            }
        }
    };
    let start = Instant::now();
    let oracle_result = solve_shared_root_with_oracle(
        shared_names.len(),
        world_weights,
        &s2_options_by_world.iter().map(Vec::len).collect::<Vec<_>>(),
        &normalized_s2_priors,
        human_prior_mix,
        normalized_s1_prior.as_deref(),
        player_prior_mix,
        min_policy_probability,
        duration,
        rounds,
        seed,
        baseline_action.as_ref().map(|(_, index)| *index),
        lcb_z,
        continuation_iterations,
        paired_evaluation_iterations,
        |world, opponent_index, action_index, iterations| {
            let action = s1_maps[world]
                .get(&shared_names[action_index])
                .ok_or_else(|| "sampled world is missing a shared side-one action".to_string())?
                .clone();
            let opponent = s2_options_by_world[world][opponent_index].clone();
            let mut continuation_state = states[world].clone();
            let result = perform_mcts(
                &mut continuation_state,
                vec![action],
                vec![opponent],
                Duration::ZERO,
                iterations,
                None,
                None,
                2.0,
            );
            let root = &result.s1[0];
            if root.visits == 0 {
                return Err("forced-root continuation produced no visits".to_string());
            }
            Ok(root.total_score / root.visits as f32)
        },
    )?;

    let completed_rounds = oracle_result.rounds;
    let entropy = -oracle_result
        .probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| probability * probability.ln())
        .sum::<f32>();
    let max_probability = oracle_result
        .probabilities
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    Ok(SharedRootResult {
        policy: shared_names
            .into_iter()
            .zip(oracle_result.probabilities)
            .zip(oracle_result.estimated_values)
            .map(
                |((action, probability), estimated_value)| SharedRootAction {
                    action,
                    probability,
                    pulls: completed_rounds,
                    estimated_value,
                },
            )
            .collect(),
        diagnostics: SharedRootDiagnostics {
            rounds: completed_rounds,
            continuation_iterations,
            unique_payoff_cells_evaluated: oracle_result.unique_payoff_cells_evaluated,
            cache_hits: oracle_result.cache_hits,
            total_forced_continuation_iterations: oracle_result
                .unique_payoff_cells_evaluated
                .saturating_mul(continuation_iterations as u64),
            world_pulls: oracle_result.world_pulls,
            elapsed_ms: start.elapsed().as_millis().min(u64::MAX as u128) as u64,
            shared_policy_entropy: entropy,
            shared_policy_max_probability: max_probability,
            human_prior_mix: human_prior_mix as f32,
            player_prior_mix: player_prior_mix as f32,
            player_prior_available: normalized_s1_prior.is_some(),
            player_prior_coverage: player_prior_coverage as f32,
            baseline_action: baseline_action.map(|(name, _)| name),
            baseline_advantage_available: oracle_result.baseline_advantage_available,
            baseline_advantage_mean: oracle_result.baseline_advantage_mean,
            baseline_advantage_standard_error: oracle_result.baseline_advantage_standard_error,
            baseline_advantage_lcb: oracle_result.baseline_advantage_lcb,
            baseline_advantage_world_count: oracle_result.baseline_advantage_world_count,
            baseline_advantage_effective_world_count: oracle_result
                .baseline_advantage_effective_world_count,
            lcb_z: lcb_z as f32,
            paired_evaluation_iterations,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve(
        weights: &[f64],
        payoffs: &[Vec<Vec<f32>>],
        priors: &[Option<Vec<f64>>],
        human_prior_mix: f64,
        min_policy_probability: f64,
        rounds: u32,
    ) -> OracleResult {
        let action_count = payoffs[0].len();
        solve_shared_root_with_oracle(
            action_count,
            weights,
            &payoffs
                .iter()
                .map(|world| world[0].len())
                .collect::<Vec<_>>(),
            priors,
            human_prior_mix,
            None,
            0.0,
            min_policy_probability,
            Duration::ZERO,
            rounds,
            7,
            None,
            1.645,
            128,
            512,
            |world, opponent, action, _iterations| Ok(payoffs[world][action][opponent]),
        )
        .unwrap()
    }

    #[test]
    fn matching_pennies_converges_for_both_players() {
        let payoffs = vec![vec![vec![1.0, -1.0], vec![-1.0, 1.0]]];
        let result = solve(&[1.0], &payoffs, &[None], 0.0, 0.0, 100_000);

        for probability in result
            .probabilities
            .iter()
            .chain(&result.opponent_probabilities[0])
        {
            assert!((*probability - 0.5).abs() < 0.03, "{:?}", result);
        }
    }

    #[test]
    fn payoff_oracle_is_called_at_most_once_per_cell() {
        let mut calls = vec![vec![vec![0_u32; 2]; 2]; 2];
        let result = solve_shared_root_with_oracle(
            2,
            &[1.0, 1.0],
            &[2, 2],
            &[None, None],
            0.0,
            None,
            0.0,
            0.0,
            Duration::ZERO,
            100,
            7,
            None,
            1.645,
            128,
            512,
            |world, opponent, action, _iterations| {
                calls[world][opponent][action] += 1;
                Ok(calls[world][opponent][action] as f32)
            },
        )
        .unwrap();

        assert!(calls.iter().flatten().flatten().all(|count| *count <= 1));
        assert_eq!(
            result.unique_payoff_cells_evaluated,
            calls
                .iter()
                .flatten()
                .flatten()
                .map(|count| *count as u64)
                .sum()
        );
        assert_eq!(
            result.unique_payoff_cells_evaluated + result.cache_hits,
            300
        );
        assert!(result.estimated_values.iter().all(|value| *value == 1.0));
    }

    #[test]
    fn rock_paper_scissors_converges_to_thirds() {
        let payoffs = vec![vec![
            vec![0.0, -1.0, 1.0],
            vec![1.0, 0.0, -1.0],
            vec![-1.0, 1.0, 0.0],
        ]];
        let result = solve(&[1.0], &payoffs, &[None], 0.0, 0.0, 200_000);

        for probability in result.probabilities {
            assert!((probability - 1.0 / 3.0).abs() < 0.03);
        }
    }

    #[test]
    fn dominated_action_loses_mass() {
        let payoffs = vec![vec![
            vec![0.0, -1.0, 1.0],
            vec![1.0, 0.0, -1.0],
            vec![-1.0, 1.0, 0.0],
            vec![-2.0, -2.0, -2.0],
        ]];
        let result = solve(&[1.0], &payoffs, &[None], 0.0, 0.0, 100_000);
        assert!(result.probabilities[3] < 0.001);
    }

    #[test]
    fn belief_weights_control_one_global_policy() {
        let payoffs = vec![vec![vec![1.0], vec![-1.0]], vec![vec![-1.0], vec![1.0]]];
        let equal = solve(&[1.0, 1.0], &payoffs, &[None, None], 0.0, 0.0, 100_000);
        let first = solve(&[9.0, 1.0], &payoffs, &[None, None], 0.0, 0.0, 50_000);
        let second = solve(&[1.0, 9.0], &payoffs, &[None, None], 0.0, 0.0, 50_000);

        assert!(equal.probabilities[0] > 0.2, "{:?}", equal);
        assert!(equal.probabilities[1] > 0.2, "{:?}", equal);
        assert!(first.probabilities[0] > 0.95);
        assert!(second.probabilities[1] > 0.95);
    }

    #[test]
    fn human_prior_mix_shifts_policy_but_zero_mix_is_equilibrium() {
        let payoffs = vec![vec![vec![1.0, -1.0], vec![-1.0, 1.0]]];
        let equilibrium = solve(&[1.0], &payoffs, &[Some(vec![1.0, 0.0])], 0.0, 0.0, 100_000);
        let anchored = solve(
            &[1.0],
            &payoffs,
            &[Some(vec![1.0, 0.0])],
            0.75,
            0.0,
            100_000,
        );

        assert!((equilibrium.probabilities[0] - 0.5).abs() < 0.03);
        assert!(anchored.probabilities[0] > 0.95, "{:?}", anchored);
    }

    fn solve_with_player_prior(prior: &[f64], mix: f64, rounds: u32) -> OracleResult {
        solve_shared_root_with_oracle(
            2,
            &[1.0],
            &[1],
            &[None],
            0.0,
            Some(prior),
            mix,
            0.0,
            Duration::ZERO,
            rounds,
            7,
            None,
            1.645,
            128,
            512,
            |_, _, action, _iterations| Ok(if action == 0 { 1.0 } else { 0.0 }),
        )
        .unwrap()
    }

    #[test]
    fn zero_player_prior_mix_reproduces_rm_and_prior_shifts_behavior() {
        let equilibrium = solve_with_player_prior(&[0.0, 1.0], 0.0, 10_000);
        let anchored = solve_with_player_prior(&[0.0, 1.0], 0.75, 10_000);

        assert!(equilibrium.probabilities[0] > 0.99);
        assert!(anchored.probabilities[1] > 0.74);
    }

    #[test]
    fn regrets_correct_more_of_a_bad_prior_as_mix_decreases() {
        let high_mix = solve_with_player_prior(&[0.0, 1.0], 0.75, 10_000);
        let low_mix = solve_with_player_prior(&[0.0, 1.0], 0.10, 10_000);

        assert!(low_mix.probabilities[0] > high_mix.probabilities[0] + 0.6);
        assert!(low_mix.probabilities[0] > 0.89);
    }

    #[test]
    fn output_floor_removes_residue_and_renormalizes() {
        let pruned = prune_policy(vec![0.50, 0.49, 0.01], 0.02);
        assert_eq!(pruned[2], 0.0);
        assert!((pruned.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((pruned[0] - 0.50 / 0.99).abs() < 1e-12);

        assert_eq!(prune_policy(vec![0.5, 0.5], 1.0), vec![1.0, 0.0]);
    }

    fn exact_paired_advantage(
        weights: &[f64],
        player: &[f64],
        baseline: usize,
        payoffs: &[Vec<Vec<f64>>],
    ) -> PairedAdvantage {
        let opponent = vec![vec![0.25_f32, 0.75_f32]; weights.len()];
        let cache = payoffs
            .iter()
            .enumerate()
            .flat_map(|(world, actions)| {
                actions
                    .iter()
                    .enumerate()
                    .flat_map(move |(action, values)| {
                        values
                            .iter()
                            .copied()
                            .enumerate()
                            .map(move |(opponent, value)| ((world, opponent, action), value))
                    })
            })
            .collect();
        paired_advantage_from_cache(
            player.len(),
            weights,
            &opponent,
            player,
            baseline,
            1.645,
            &cache,
        )
        .unwrap()
    }

    #[test]
    fn paired_advantage_exact_positive_negative_and_zero() {
        // Both opponent columns have the same paired action difference. Large
        // common payoff levels cancel rather than inflating the uncertainty.
        let payoffs = vec![vec![vec![10.0, 100.0], vec![14.0, 104.0]]];
        let positive = exact_paired_advantage(&[1.0], &[0.25, 0.75], 0, &payoffs);
        let negative = exact_paired_advantage(&[1.0], &[0.25, 0.75], 1, &payoffs);
        let zero = exact_paired_advantage(&[1.0], &[1.0, 0.0], 0, &payoffs);

        assert_eq!(positive.mean, Some(3.0));
        assert_eq!(positive.standard_error, Some(0.0));
        assert_eq!(positive.lcb, Some(3.0));
        assert_eq!(negative.mean, Some(-1.0));
        assert_eq!(zero.mean, Some(0.0));
    }

    #[test]
    fn paired_advantage_uses_normalized_weights_and_effective_world_count() {
        let payoffs = vec![
            vec![vec![10.0, 100.0], vec![14.0, 104.0]],
            vec![vec![-20.0, 80.0], vec![-21.0, 79.0]],
        ];
        let result = exact_paired_advantage(&[1.0, 3.0], &[0.25, 0.75], 0, &payoffs);
        let expected_se = (7.03125_f64 / 1.6).sqrt();

        assert_eq!(result.world_count, 2);
        assert!((result.effective_world_count - 1.6).abs() < 1e-12);
        assert!((result.mean.unwrap() - 0.1875).abs() < 1e-12);
        assert!((result.standard_error.unwrap() - expected_se).abs() < 1e-12);
        assert!((result.lcb.unwrap() - (0.1875 - 1.645 * expected_se)).abs() < 1e-12);
    }

    #[test]
    fn opponent_average_accumulates_behavior_including_human_mixture() {
        let result = solve_shared_root_with_oracle(
            1,
            &[1.0],
            &[2],
            &[Some(vec![1.0, 0.0])],
            1.0,
            None,
            0.0,
            0.0,
            Duration::ZERO,
            1,
            0,
            None,
            1.645,
            128,
            512,
            |_, opponent, _, _iterations| Ok(opponent as f32),
        )
        .unwrap();

        assert_eq!(result.opponent_probabilities, vec![vec![1.0, 0.0]]);
    }

    #[test]
    fn timed_diagnostic_is_unavailable_when_required_cells_are_incomplete() {
        let result = solve_shared_root_with_oracle(
            2,
            &[1.0],
            &[2],
            &[None],
            0.0,
            None,
            0.0,
            0.0,
            Duration::from_millis(1),
            0,
            0,
            Some(0),
            1.645,
            128,
            512,
            |_, _, _, _iterations| {
                std::thread::sleep(Duration::from_millis(2));
                Ok(0.0)
            },
        )
        .unwrap();

        assert!(!result.baseline_advantage_available);
        assert_eq!(result.baseline_advantage_mean, None);
    }

    #[test]
    fn invalid_baseline_index_and_lcb_z_are_rejected() {
        let run = |baseline, z| {
            solve_shared_root_with_oracle(
                1,
                &[1.0],
                &[1],
                &[None],
                0.0,
                None,
                0.0,
                0.0,
                Duration::ZERO,
                1,
                0,
                baseline,
                z,
                128,
                512,
                |_, _, _, _iterations| Ok(0.0),
            )
        };

        assert!(run(Some(1), 1.645).is_err());
        assert!(run(None, -0.1).is_err());
        assert!(run(None, f64::NAN).is_err());
        assert!(run(None, f64::INFINITY).is_err());
    }

    #[test]
    fn invalid_mix_and_floor_are_rejected() {
        let run = |mix, floor| {
            solve_shared_root_with_oracle(
                1,
                &[1.0],
                &[1],
                &[None],
                mix,
                None,
                0.0,
                floor,
                Duration::ZERO,
                1,
                0,
                None,
                1.645,
                128,
                512,
                |_, _, _, _iterations| Ok(0.0),
            )
        };
        assert!(run(-0.1, 0.0).is_err());
        assert!(run(1.1, 0.0).is_err());
        assert!(run(0.0, -0.1).is_err());
        assert!(run(0.0, 1.1).is_err());

        let invalid_player_mix = |mix| {
            solve_shared_root_with_oracle(
                1,
                &[1.0],
                &[1],
                &[None],
                0.0,
                Some(&[1.0]),
                mix,
                0.0,
                Duration::ZERO,
                1,
                0,
                None,
                1.645,
                128,
                512,
                |_, _, _, _iterations| Ok(0.0),
            )
        };
        assert!(invalid_player_mix(-0.1).is_err());
        assert!(invalid_player_mix(1.1).is_err());
        assert!(invalid_player_mix(f64::NAN).is_err());

        let invalid_values = solve_shared_root_with_oracle(
            1,
            &[1.0],
            &[1],
            &[None],
            0.0,
            Some(&[-1.0]),
            0.25,
            0.0,
            Duration::ZERO,
            1,
            0,
            None,
            1.645,
            128,
            512,
            |_, _, _, _iterations| Ok(0.0),
        );
        assert!(invalid_values.is_err());
    }

    #[test]
    fn paired_evaluation_uses_different_iteration_count_than_optimization() {
        let mut optimization_calls = 0u32;
        let mut paired_calls = 0u32;

        let result = solve_shared_root_with_oracle(
            2,
            &[1.0],
            &[2],
            &[None],
            0.0,
            None,
            0.0,
            0.0,
            Duration::ZERO,
            10,
            7,
            Some(0),
            0.0,
            128,
            512,
            |_world, _opponent, action, iterations| {
                if iterations == 128 {
                    optimization_calls += 1;
                } else if iterations == 512 {
                    paired_calls += 1;
                }
                Ok(if action == 0 { 1.0 } else { 0.0 })
            },
        )
        .unwrap();

        assert!(
            optimization_calls > 0,
            "optimization cells should be evaluated with continuation_iterations"
        );
        assert!(
            paired_calls > 0,
            "paired diagnostic cells should be evaluated with paired_evaluation_iterations"
        );
        assert!(
            result.baseline_advantage_available,
            "paired advantage should be available when baseline_action is supplied"
        );
    }
}
