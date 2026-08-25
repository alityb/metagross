use crate::engine::evaluate::evaluate;
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::StateInstructions;
use crate::mcts::{perform_mcts_seeded, MctsResult, MctsSideResult};
use crate::state::{Side, State};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

pub const MAX_PAIRED_ROLLOUTS: u32 = 100_000;
pub const MAX_CONTINUATION_ITERATIONS: u32 = 10_000_000;
pub const MAX_CONTINUATION_STEPS: u32 = 100;
pub const MAX_TOTAL_CONTINUATION_ITERATIONS: u64 = 100_000_000;

const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const TAPE_ROLLOUT_STRIDE: u64 = 2 + 2 * MAX_CONTINUATION_STEPS as u64;
const ROOT_OPPONENT_CHANNEL: u64 = 0;
const ROOT_BRANCH_CHANNEL: u64 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct PairedRootPolicyEvaluation {
    pub pairs: u32,
    pub baseline_sum: f64,
    pub candidate_sum: f64,
    pub delta_sum: f64,
    pub delta_squared_sum: f64,
    pub catastrophic_count: u32,
    pub candidate_catastrophic_count: u32,
    pub baseline_catastrophic_count: u32,
    pub candidate_catastrophic_severity_sum: f64,
    pub baseline_catastrophic_severity_sum: f64,
    pub candidate_better_count: u32,
    pub baseline_better_count: u32,
    pub equal_count: u32,
    pub baseline_terminal_count: u32,
    pub candidate_terminal_count: u32,
    pub baseline_nonterminal_evaluation_delta_sum: f64,
    pub candidate_nonterminal_evaluation_delta_sum: f64,
    pub baseline_nonterminal_count: u32,
    pub candidate_nonterminal_count: u32,
    pub continuation_iterations_executed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PairedRootError {
    InvalidBounds(&'static str),
    IllegalAction { role: &'static str, action: String },
    InvalidOpponentPriors,
    InvalidBranchProbabilities,
    NoLegalContinuation,
}

impl fmt::Display for PairedRootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBounds(message) => write!(f, "invalid evaluator bounds: {message}"),
            Self::IllegalAction { role, action } => {
                write!(f, "illegal {role} action: {action}")
            }
            Self::InvalidOpponentPriors => {
                write!(f, "opponent priors contain invalid or duplicate entries")
            }
            Self::InvalidBranchProbabilities => {
                write!(f, "engine returned invalid branch probabilities")
            }
            Self::NoLegalContinuation => write!(f, "nonterminal state has no legal continuation"),
        }
    }
}

impl Error for PairedRootError {}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Return one word from the common random tape.
///
/// The exact scheme is `SplitMix64(seed + GAMMA * counter)`, where `counter =
/// rollout * 202 + channel`. Channels 0 and 1 are the root opponent and root
/// branch. Continuation step `s` uses channels `2 + 2*s` for both MCTS seeds
/// and `3 + 2*s` for both transition branches. Action names, action identity,
/// and arm identity are deliberately absent.
fn tape_word(seed: u64, rollout: u32, channel: u64) -> u64 {
    let counter = (rollout as u64)
        .wrapping_mul(TAPE_ROLLOUT_STRIDE)
        .wrapping_add(channel);
    splitmix64(seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(counter)))
}

fn unit_uniform(word: u64) -> f64 {
    const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
    ((word >> 11) as f64) * SCALE
}

fn uniform_index(word: u64, length: usize) -> usize {
    (((word as u128) * (length as u128)) >> 64) as usize
}

const OPPONENT_PRIOR_UNIFORM_MIX: f32 = 0.25;

fn opponent_weights(
    side: &Side,
    options: &[MoveChoice],
    priors: Option<&[(String, f32)]>,
) -> Result<Option<Vec<f32>>, PairedRootError> {
    let Some(priors) = priors else {
        return Ok(None);
    };
    let mut supplied = HashMap::with_capacity(priors.len());
    let mut supplied_total = 0.0_f32;
    for (action, probability) in priors {
        if action.is_empty()
            || !probability.is_finite()
            || *probability < 0.0
            || *probability > 1.0
            || supplied.insert(action.as_str(), *probability).is_some()
        {
            return Err(PairedRootError::InvalidOpponentPriors);
        }
        supplied_total += probability;
    }
    if !supplied_total.is_finite() || supplied_total <= 0.0 {
        return Err(PairedRootError::InvalidOpponentPriors);
    }

    let mut weights: Vec<f32> = options
        .iter()
        .map(|choice| {
            supplied
                .get(canonical_action_name(side, choice).as_str())
                .copied()
                .filter(|probability| *probability > 0.0)
                .unwrap_or(0.0)
        })
        .collect();
    let matched_total: f32 = weights.iter().sum();
    if matched_total <= 0.0 {
        return Ok(None);
    }
    let uniform_weight = OPPONENT_PRIOR_UNIFORM_MIX / weights.len() as f32;
    for weight in &mut weights {
        *weight = (1.0 - OPPONENT_PRIOR_UNIFORM_MIX) * (*weight / matched_total) + uniform_weight;
    }
    Ok(Some(weights))
}

fn opponent_index(word: u64, length: usize, weights: Option<&[f32]>) -> usize {
    let Some(weights) = weights else {
        return uniform_index(word, length);
    };
    let target = unit_uniform(word);
    let mut cumulative = 0.0_f64;
    let mut last_positive = 0;
    for (index, weight) in weights.iter().enumerate() {
        if *weight > 0.0 {
            cumulative += *weight as f64;
            last_positive = index;
            if target < cumulative {
                return index;
            }
        }
    }
    last_positive
}

pub fn canonical_action_name(side: &Side, choice: &MoveChoice) -> String {
    match choice {
        MoveChoice::Switch(_) => format!("switch {}", choice.to_string(side)),
        _ => choice.to_string(side),
    }
}

fn sort_options(side: &Side, options: &mut [MoveChoice]) {
    options.sort_by_cached_key(|choice| canonical_action_name(side, choice));
}

fn resolve_action(
    side: &Side,
    options: &[MoveChoice],
    action: &str,
    role: &'static str,
) -> Result<MoveChoice, PairedRootError> {
    options
        .iter()
        .find(|choice| canonical_action_name(side, choice) == action)
        .cloned()
        .ok_or_else(|| PairedRootError::IllegalAction {
            role,
            action: action.to_owned(),
        })
}

fn choose_branch<'a>(
    branches: &'a [StateInstructions],
    uniform: f64,
) -> Result<&'a StateInstructions, PairedRootError> {
    let total = branches.iter().try_fold(0.0_f64, |total, branch| {
        let weight = branch.percentage as f64;
        if weight.is_finite() && weight >= 0.0 {
            Some(total + weight)
        } else {
            None
        }
    });
    let total = total
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or(PairedRootError::InvalidBranchProbabilities)?;
    let target = uniform * total;
    let mut cumulative = 0.0;
    let mut last_positive = None;
    for branch in branches {
        let weight = branch.percentage as f64;
        if weight > 0.0 {
            cumulative += weight;
            last_positive = Some(branch);
            if target < cumulative {
                return Ok(branch);
            }
        }
    }
    last_positive.ok_or(PairedRootError::InvalidBranchProbabilities)
}

fn select_by_visits(
    side: &Side,
    options: &[MctsSideResult],
) -> Result<MoveChoice, PairedRootError> {
    options
        .iter()
        .max_by(|left, right| {
            left.visits.cmp(&right.visits).then_with(|| {
                canonical_action_name(side, &right.move_choice)
                    .cmp(&canonical_action_name(side, &left.move_choice))
            })
        })
        .map(|result| result.move_choice.clone())
        .ok_or(PairedRootError::NoLegalContinuation)
}

fn continuation_choices(
    state: &mut State,
    iterations: u32,
    seed: u64,
) -> Result<(MoveChoice, MoveChoice, u64), PairedRootError> {
    let (mut side_one_options, mut side_two_options) = state.get_all_options();
    if side_one_options.is_empty() || side_two_options.is_empty() {
        return Err(PairedRootError::NoLegalContinuation);
    }
    sort_options(&state.side_one, &mut side_one_options);
    sort_options(&state.side_two, &mut side_two_options);
    let MctsResult {
        s1,
        s2,
        iteration_count,
        ..
    } = perform_mcts_seeded(state, side_one_options, side_two_options, iterations, seed);
    Ok((
        select_by_visits(&state.side_one, &s1)?,
        select_by_visits(&state.side_two, &s2)?,
        iteration_count as u64,
    ))
}

fn advance_continuation(
    state: &mut State,
    iterations: u32,
    mcts_seed: u64,
    branch_uniform: f64,
) -> Result<u64, PairedRootError> {
    if state.battle_is_over() != 0.0 {
        return Ok(0);
    }
    let (side_one_choice, side_two_choice, executed) =
        continuation_choices(state, iterations, mcts_seed)?;
    let branches =
        generate_instructions_from_move_pair(state, &side_one_choice, &side_two_choice, true);
    let branch = choose_branch(&branches, branch_uniform)?;
    state.apply_instructions(&branch.instruction_list);
    Ok(executed)
}

fn score(state: &State, root_evaluation: f32) -> (f64, bool, Option<f64>) {
    match state.battle_is_over() {
        value if value > 0.0 => (1.0, true, None),
        value if value < 0.0 => (0.0, true, None),
        _ => {
            let centered = evaluate(state) - root_evaluation;
            (
                (1.0 / (1.0 + (-0.0125 * centered).exp())) as f64,
                false,
                Some(centered as f64),
            )
        }
    }
}

fn record_scores(
    result: &mut PairedRootPolicyEvaluation,
    baseline_score: f64,
    baseline_terminal: bool,
    baseline_evaluation_delta: Option<f64>,
    candidate_score: f64,
    candidate_terminal: bool,
    candidate_evaluation_delta: Option<f64>,
) {
    let delta = candidate_score - baseline_score;
    result.baseline_sum += baseline_score;
    result.candidate_sum += candidate_score;
    result.delta_sum += delta;
    result.delta_squared_sum += delta * delta;
    let candidate_catastrophic = delta <= -0.5;
    let baseline_catastrophic = delta >= 0.5;
    result.catastrophic_count += u32::from(candidate_catastrophic);
    result.candidate_catastrophic_count += u32::from(candidate_catastrophic);
    result.baseline_catastrophic_count += u32::from(baseline_catastrophic);
    if candidate_catastrophic {
        result.candidate_catastrophic_severity_sum += -delta;
    }
    if baseline_catastrophic {
        result.baseline_catastrophic_severity_sum += delta;
    }
    result.baseline_terminal_count += u32::from(baseline_terminal);
    result.candidate_terminal_count += u32::from(candidate_terminal);
    if let Some(evaluation_delta) = baseline_evaluation_delta {
        result.baseline_nonterminal_evaluation_delta_sum += evaluation_delta;
        result.baseline_nonterminal_count += 1;
    }
    if let Some(evaluation_delta) = candidate_evaluation_delta {
        result.candidate_nonterminal_evaluation_delta_sum += evaluation_delta;
        result.candidate_nonterminal_count += 1;
    }
    if delta > 0.0 {
        result.candidate_better_count += 1;
    } else if delta < 0.0 {
        result.baseline_better_count += 1;
    } else {
        result.equal_count += 1;
    }
}

fn validate_bounds(
    rollouts: u32,
    continuation_iterations: u32,
    continuation_steps: u32,
) -> Result<(), PairedRootError> {
    if rollouts == 0 || rollouts > MAX_PAIRED_ROLLOUTS {
        return Err(PairedRootError::InvalidBounds(
            "rollouts must be between 1 and 100000",
        ));
    }
    if continuation_iterations == 0 || continuation_iterations > MAX_CONTINUATION_ITERATIONS {
        return Err(PairedRootError::InvalidBounds(
            "continuation_iterations must be between 1 and 10000000",
        ));
    }
    if continuation_steps == 0 || continuation_steps > MAX_CONTINUATION_STEPS {
        return Err(PairedRootError::InvalidBounds(
            "continuation_steps must be between 1 and 100",
        ));
    }
    let requested = (rollouts as u64)
        .saturating_mul(continuation_iterations as u64)
        .saturating_mul(continuation_steps as u64)
        .saturating_mul(2);
    if requested > MAX_TOTAL_CONTINUATION_ITERATIONS {
        return Err(PairedRootError::InvalidBounds(
            "total requested continuation iterations exceed 100000000",
        ));
    }
    Ok(())
}

/// Independently evaluate two legal side-one root actions on a common tape.
///
/// Only aggregate paired statistics are returned. The input state is cloned
/// before any transition or search and is never modified.
pub fn paired_root_policy_evaluation(
    state: &State,
    baseline_action: &str,
    candidate_action: &str,
    rollouts: u32,
    continuation_iterations: u32,
    continuation_steps: u32,
    seed: u64,
) -> Result<PairedRootPolicyEvaluation, PairedRootError> {
    paired_root_policy_evaluation_with_opponent_priors(
        state,
        baseline_action,
        candidate_action,
        rollouts,
        continuation_iterations,
        continuation_steps,
        seed,
        None,
    )
}

/// Evaluate paired root actions while sampling the opponent from optional priors.
pub fn paired_root_policy_evaluation_with_opponent_priors(
    state: &State,
    baseline_action: &str,
    candidate_action: &str,
    rollouts: u32,
    continuation_iterations: u32,
    continuation_steps: u32,
    seed: u64,
    opponent_priors: Option<&[(String, f32)]>,
) -> Result<PairedRootPolicyEvaluation, PairedRootError> {
    validate_bounds(rollouts, continuation_iterations, continuation_steps)?;

    let root = state.clone();
    let (side_one_options, mut side_two_options) = root.root_get_all_options();
    let baseline = resolve_action(
        &root.side_one,
        &side_one_options,
        baseline_action,
        "baseline",
    )?;
    let candidate = resolve_action(
        &root.side_one,
        &side_one_options,
        candidate_action,
        "candidate",
    )?;
    if side_two_options.is_empty() {
        return Err(PairedRootError::NoLegalContinuation);
    }
    sort_options(&root.side_two, &mut side_two_options);
    let opponent_weights = opponent_weights(&root.side_two, &side_two_options, opponent_priors)?;
    let root_evaluation = evaluate(&root);

    let mut result = PairedRootPolicyEvaluation {
        pairs: rollouts,
        baseline_sum: 0.0,
        candidate_sum: 0.0,
        delta_sum: 0.0,
        delta_squared_sum: 0.0,
        catastrophic_count: 0,
        candidate_catastrophic_count: 0,
        baseline_catastrophic_count: 0,
        candidate_catastrophic_severity_sum: 0.0,
        baseline_catastrophic_severity_sum: 0.0,
        candidate_better_count: 0,
        baseline_better_count: 0,
        equal_count: 0,
        baseline_terminal_count: 0,
        candidate_terminal_count: 0,
        baseline_nonterminal_evaluation_delta_sum: 0.0,
        candidate_nonterminal_evaluation_delta_sum: 0.0,
        baseline_nonterminal_count: 0,
        candidate_nonterminal_count: 0,
        continuation_iterations_executed: 0,
    };

    for rollout in 0..rollouts {
        let opponent_index = opponent_index(
            tape_word(seed, rollout, ROOT_OPPONENT_CHANNEL),
            side_two_options.len(),
            opponent_weights.as_deref(),
        );
        let opponent = &side_two_options[opponent_index];
        let root_uniform = unit_uniform(tape_word(seed, rollout, ROOT_BRANCH_CHANNEL));

        let mut baseline_state = root.clone();
        let mut candidate_state = root.clone();
        let baseline_branches =
            generate_instructions_from_move_pair(&mut baseline_state, &baseline, opponent, true);
        let candidate_branches =
            generate_instructions_from_move_pair(&mut candidate_state, &candidate, opponent, true);
        let baseline_branch = choose_branch(&baseline_branches, root_uniform)?;
        let candidate_branch = choose_branch(&candidate_branches, root_uniform)?;
        baseline_state.apply_instructions(&baseline_branch.instruction_list);
        candidate_state.apply_instructions(&candidate_branch.instruction_list);

        for step in 0..continuation_steps {
            let mcts_seed = tape_word(seed, rollout, 2 + 2 * step as u64);
            let branch_uniform = unit_uniform(tape_word(seed, rollout, 3 + 2 * step as u64));
            result.continuation_iterations_executed += advance_continuation(
                &mut baseline_state,
                continuation_iterations,
                mcts_seed,
                branch_uniform,
            )?;
            result.continuation_iterations_executed += advance_continuation(
                &mut candidate_state,
                continuation_iterations,
                mcts_seed,
                branch_uniform,
            )?;
        }

        let (baseline_score, baseline_terminal, baseline_evaluation_delta) =
            score(&baseline_state, root_evaluation);
        let (candidate_score, candidate_terminal, candidate_evaluation_delta) =
            score(&candidate_state, root_evaluation);
        record_scores(
            &mut result,
            baseline_score,
            baseline_terminal,
            baseline_evaluation_delta,
            candidate_score,
            candidate_terminal,
            candidate_evaluation_delta,
        );
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choices::Choices;
    use crate::state::{PokemonMoveIndex, State};

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
        state
    }

    #[test]
    fn identical_actions_have_exact_zero_deltas_and_do_not_mutate_input() {
        let state = state_with_moves();
        let before = state.serialize();
        let result =
            paired_root_policy_evaluation(&state, "tackle", "tackle", 3, 7, 1, 42).unwrap();

        assert_eq!(result.baseline_sum, result.candidate_sum);
        assert_eq!(result.delta_sum, 0.0);
        assert_eq!(result.delta_squared_sum, 0.0);
        assert_eq!(
            result.catastrophic_count,
            result.candidate_catastrophic_count
        );
        assert_eq!(
            result.baseline_nonterminal_count,
            result.candidate_nonterminal_count
        );
        assert_eq!(
            result.baseline_nonterminal_evaluation_delta_sum,
            result.candidate_nonterminal_evaluation_delta_sum
        );
        assert_eq!(
            result.baseline_terminal_count + result.baseline_nonterminal_count,
            3
        );
        assert_eq!(
            result.candidate_terminal_count + result.candidate_nonterminal_count,
            3
        );
        assert_eq!(result.equal_count, 3);
        assert_eq!(result.continuation_iterations_executed, 42);
        assert_eq!(state.serialize(), before);
    }

    #[test]
    fn tape_is_deterministic_and_does_not_include_action_names() {
        let expected: Vec<_> = (0..4)
            .map(|rollout| tape_word(19, rollout, ROOT_OPPONENT_CHANNEL))
            .collect();
        assert_eq!(
            expected,
            (0..4)
                .map(|rollout| tape_word(19, rollout, ROOT_OPPONENT_CHANNEL))
                .collect::<Vec<_>>()
        );
        assert_ne!(expected[0], tape_word(20, 0, ROOT_OPPONENT_CHANNEL));
    }

    #[test]
    fn same_seed_is_equal_and_swapping_arms_negates_delta() {
        let state = state_with_moves();
        let first = paired_root_policy_evaluation(&state, "splash", "tackle", 4, 9, 1, 99).unwrap();
        let second =
            paired_root_policy_evaluation(&state, "splash", "tackle", 4, 9, 1, 99).unwrap();
        let swapped =
            paired_root_policy_evaluation(&state, "tackle", "splash", 4, 9, 1, 99).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.baseline_sum, swapped.candidate_sum);
        assert_eq!(first.candidate_sum, swapped.baseline_sum);
        assert_eq!(first.delta_sum, -swapped.delta_sum);
        assert_eq!(first.delta_squared_sum, swapped.delta_squared_sum);
        assert_eq!(first.catastrophic_count, first.candidate_catastrophic_count);
        assert_eq!(
            swapped.catastrophic_count,
            swapped.candidate_catastrophic_count
        );
        assert_eq!(
            first.candidate_catastrophic_count,
            swapped.baseline_catastrophic_count
        );
        assert_eq!(
            first.baseline_catastrophic_count,
            swapped.candidate_catastrophic_count
        );
        assert_eq!(
            first.candidate_catastrophic_severity_sum,
            swapped.baseline_catastrophic_severity_sum
        );
        assert_eq!(
            first.baseline_catastrophic_severity_sum,
            swapped.candidate_catastrophic_severity_sum
        );
        assert_eq!(
            first.baseline_nonterminal_evaluation_delta_sum,
            swapped.candidate_nonterminal_evaluation_delta_sum
        );
        assert_eq!(
            first.candidate_nonterminal_evaluation_delta_sum,
            swapped.baseline_nonterminal_evaluation_delta_sum
        );
        assert_eq!(
            first.baseline_nonterminal_count,
            swapped.candidate_nonterminal_count
        );
        assert_eq!(
            first.candidate_nonterminal_count,
            swapped.baseline_nonterminal_count
        );
    }

    #[test]
    fn catastrophic_threshold_is_inclusive_and_severity_is_bounded() {
        let state = state_with_moves();
        let mut result =
            paired_root_policy_evaluation(&state, "tackle", "tackle", 1, 1, 1, 0).unwrap();

        record_scores(&mut result, 1.0, true, None, 0.5, true, None);
        record_scores(&mut result, 0.5, true, None, 1.0, true, None);

        assert_eq!(result.catastrophic_count, 1);
        assert_eq!(result.candidate_catastrophic_count, 1);
        assert_eq!(result.baseline_catastrophic_count, 1);
        assert_eq!(result.candidate_catastrophic_severity_sum, 0.5);
        assert_eq!(result.baseline_catastrophic_severity_sum, 0.5);
        assert!(
            result.candidate_catastrophic_severity_sum
                >= 0.5 * result.candidate_catastrophic_count as f64
        );
        assert!(
            result.candidate_catastrophic_severity_sum
                <= result.candidate_catastrophic_count as f64
        );
        assert!(
            result.baseline_catastrophic_severity_sum
                >= 0.5 * result.baseline_catastrophic_count as f64
        );
        assert!(
            result.baseline_catastrophic_severity_sum <= result.baseline_catastrophic_count as f64
        );
    }

    #[test]
    fn opponent_priors_are_deterministic_biased_and_old_api_compatible() {
        let state = state_with_moves();
        let old = paired_root_policy_evaluation(&state, "splash", "tackle", 8, 1, 1, 73).unwrap();
        let no_priors = paired_root_policy_evaluation_with_opponent_priors(
            &state, "splash", "tackle", 8, 1, 1, 73, None,
        )
        .unwrap();
        let unmatched = vec![("missing".to_owned(), 1.0)];
        let unmatched_result = paired_root_policy_evaluation_with_opponent_priors(
            &state,
            "splash",
            "tackle",
            8,
            1,
            1,
            73,
            Some(&unmatched),
        )
        .unwrap();
        assert_eq!(old, no_priors);
        assert_eq!(old, unmatched_result);

        let (_, mut options) = state.root_get_all_options();
        sort_options(&state.side_two, &mut options);
        let tackle_priors = vec![("tackle".to_owned(), 1.0)];
        let weights = opponent_weights(&state.side_two, &options, Some(&tackle_priors))
            .unwrap()
            .unwrap();
        let tackle_index = options
            .iter()
            .position(|choice| canonical_action_name(&state.side_two, choice) == "tackle")
            .unwrap();
        assert!(weights[tackle_index] > OPPONENT_PRIOR_UNIFORM_MIX);
        assert!(weights.iter().all(|weight| *weight > 0.0));
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        let first = paired_root_policy_evaluation_with_opponent_priors(
            &state,
            "splash",
            "tackle",
            8,
            1,
            1,
            73,
            Some(&tackle_priors),
        )
        .unwrap();
        let second = paired_root_policy_evaluation_with_opponent_priors(
            &state,
            "splash",
            "tackle",
            8,
            1,
            1,
            73,
            Some(&tackle_priors),
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn opponent_priors_reject_invalid_entries() {
        let state = state_with_moves();
        for priors in [
            vec![("tackle".to_owned(), 0.0)],
            vec![("tackle".to_owned(), f32::NAN)],
            vec![("tackle".to_owned(), 0.5), ("tackle".to_owned(), 0.5)],
        ] {
            assert!(matches!(
                paired_root_policy_evaluation_with_opponent_priors(
                    &state,
                    "splash",
                    "tackle",
                    1,
                    1,
                    1,
                    0,
                    Some(&priors),
                ),
                Err(PairedRootError::InvalidOpponentPriors)
            ));
        }
    }

    #[test]
    fn rejects_illegal_actions_and_invalid_bounds() {
        let state = state_with_moves();
        assert!(matches!(
            paired_root_policy_evaluation(&state, "missing", "tackle", 1, 1, 1, 0),
            Err(PairedRootError::IllegalAction { .. })
        ));
        assert!(matches!(
            paired_root_policy_evaluation(&state, "tackle", "splash", 0, 1, 1, 0),
            Err(PairedRootError::InvalidBounds(_))
        ));
        assert!(matches!(
            paired_root_policy_evaluation(&state, "tackle", "splash", 1, 0, 1, 0),
            Err(PairedRootError::InvalidBounds(_))
        ));
        assert!(matches!(
            paired_root_policy_evaluation(&state, "tackle", "splash", 1, 1, 0, 0),
            Err(PairedRootError::InvalidBounds(_))
        ));
    }

    #[cfg(feature = "terastallization")]
    #[test]
    fn resolves_root_tera_action_names() {
        let state = state_with_moves();
        let result =
            paired_root_policy_evaluation(&state, "tackle", "tackle-tera", 1, 1, 1, 0).unwrap();
        assert_eq!(result.pairs, 1);
    }

    #[test]
    fn terminal_scores_are_mapped_to_zero_and_one() {
        let mut won = state_with_moves();
        for pokemon in &mut won.side_two.pokemon.pkmn {
            pokemon.hp = 0;
        }
        assert_eq!(score(&won, evaluate(&won)), (1.0, true, None));

        let mut lost = state_with_moves();
        for pokemon in &mut lost.side_one.pokemon.pkmn {
            pokemon.hp = 0;
        }
        assert_eq!(score(&lost, evaluate(&lost)), (0.0, true, None));
    }

    #[test]
    fn nonterminal_horizon_score_is_bounded() {
        let state = state_with_moves();
        let (value, terminal, evaluation_delta) = score(&state, evaluate(&state) - 10_000.0);
        assert!(!terminal);
        assert!((0.0..=1.0).contains(&value));
        assert_eq!(evaluation_delta, Some(10_000.0));
    }
}
