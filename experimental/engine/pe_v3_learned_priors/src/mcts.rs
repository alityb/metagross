use crate::engine::evaluate::evaluate;
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::StateInstructions;
use crate::learned_value::{learned_logit, learned_rollout_value};
use crate::state::State;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rng;
use std::collections::HashMap;
use std::env;
use std::time::Duration;

fn sigmoid(x: f32) -> f32 {
    // Tuned so that ~200 points is very close to 1.0
    1.0 / (1.0 + (-0.0125 * x).exp())
}

struct RolloutContext {
    hand_root_eval: f32,
    learned_root_logit: Option<f32>,
    learned_weight: f32,
}

fn learned_weight() -> f32 {
    env::var("METAGROSS_LEARNED_VALUE_WEIGHT")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        // Learned value candidates are experimental; preserve the hand
        // evaluator unless an evaluation explicitly opts into a blend.
        .unwrap_or(0.0)
        .clamp(0.0, 1.0)
}

#[derive(Debug)]
pub struct Node {
    pub root: bool,
    pub parent: *mut Node,
    pub times_visited: u32,
    pub value_sum: f32,
    pub terminal_value_sum: f32,
    pub terminal_visits: u32,
    pub depth: u8,
    pub sampled_features: Option<Vec<f32>>,

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: u8,
    pub s2_choice: u8,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Option<Vec<MoveNode>>,
    pub s2_options: Option<Vec<MoveNode>>,
}

impl Node {
    fn new() -> Node {
        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            value_sum: 0.0,
            terminal_value_sum: 0.0,
            terminal_visits: 0,
            depth: 0,
            sampled_features: None,
            s1_choice: 0,
            s2_choice: 0,
            s1_options: None,
            s2_options: None,
        }
    }
    unsafe fn populate(&mut self, s1_options: Vec<MoveChoice>, s2_options: Vec<MoveChoice>) {
        let s1_options_vec: Vec<MoveNode> = s1_options
            .iter()
            .map(|x| MoveNode {
                prior: None,
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
            })
            .collect();
        let s2_options_vec: Vec<MoveNode> = s2_options
            .iter()
            .map(|x| MoveNode {
                prior: None,
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
            })
            .collect();

        self.s1_options = Some(s1_options_vec);
        self.s2_options = Some(s2_options_vec);
    }

    /// Attach c_puct-scaled root priors to this node's options (root only).
    /// Options without a prior entry (None) keep plain UCB1 behavior.
    fn assign_root_priors(
        &mut self,
        s1_priors: Option<Vec<Option<f32>>>,
        s2_priors: Option<Vec<Option<f32>>>,
        c_puct: f32,
    ) {
        if let (Some(priors), Some(options)) = (s1_priors, self.s1_options.as_mut()) {
            for (node, p) in options.iter_mut().zip(priors.iter()) {
                node.prior = p.map(|v| c_puct * v);
            }
        }
        if let (Some(priors), Some(options)) = (s2_priors, self.s2_options.as_mut()) {
            for (node, p) in options.iter_mut().zip(priors.iter()) {
                node.prior = p.map(|v| c_puct * v);
            }
        }
    }

    pub fn maximize_ucb_for_side(&self, side_map: &[MoveNode]) -> usize {
        let mut choice = 0;
        let mut best_ucb1 = f32::MIN;
        for (index, node) in side_map.iter().enumerate() {
            let this_ucb1 = node.ucb1(self.times_visited);
            if this_ucb1 > best_ucb1 {
                best_ucb1 = this_ucb1;
                choice = index;
            }
        }
        choice
    }

    pub unsafe fn selection(
        &mut self,
        state: &mut State,
        children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
    ) -> (*mut Node, usize, usize) {
        if self.s1_options.is_none() {
            let (s1_options, s2_options) = state.get_all_options();
            self.populate(s1_options, s2_options);
        }

        let s1_mc_index = self.maximize_ucb_for_side(self.s1_options.as_ref().unwrap());
        let s2_mc_index = self.maximize_ucb_for_side(self.s2_options.as_ref().unwrap());
        let key = (self as *mut Node as usize, s1_mc_index, s2_mc_index);
        match children.get_mut(&key) {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Box<[Node]>;
                let chosen_child = self.sample_node(child_vec_ptr);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection(state, children)
            }
            None => (self as *mut Node, s1_mc_index, s2_mc_index),
        }
    }

    unsafe fn sample_node(&self, move_vector: *mut Box<[Node]>) -> *mut Node {
        let mut rng = rng();
        let weights: Vec<f64> = (*move_vector)
            .iter()
            .map(|x| x.instructions.percentage as f64)
            .collect();
        let dist = WeightedIndex::new(weights).unwrap();
        let chosen_node = &mut (&mut *move_vector)[dist.sample(&mut rng)];
        let chosen_node_ptr = chosen_node as *mut Node;
        chosen_node_ptr
    }

    pub unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
        children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
    ) -> *mut Node {
        let s1_move = &self.s1_options.as_ref().unwrap()[s1_move_index].move_choice;
        let s2_move = &self.s2_options.as_ref().unwrap()[s2_move_index].move_choice;
        // if the battle is over or both moves are none there is no need to expand
        if (state.battle_is_over() != 0.0 && !self.root)
            || (s1_move == &MoveChoice::None && s2_move == &MoveChoice::None)
        {
            return self as *mut Node;
        }
        let should_branch_on_damage = self.root || (*self.parent).root;
        let mut new_instructions =
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage);
        let mut this_pair_vec = Vec::with_capacity(new_instructions.len());
        for state_instructions in new_instructions.drain(..) {
            let mut new_node = Node::new();
            new_node.parent = self;
            new_node.depth = self.depth.saturating_add(1);
            new_node.instructions = state_instructions;
            new_node.s1_choice = s1_move_index as u8;
            new_node.s2_choice = s2_move_index as u8;
            this_pair_vec.push(new_node);
        }

        // sample a node from the new instruction list.
        // this is the node that the rollout will be done on.
        // into_boxed_slice drops the Vec's spare capacity and, more importantly,
        // makes it a type that cannot be resized, which ensures the node
        // addresses are stable for the children map keys
        let mut boxed = this_pair_vec.into_boxed_slice();
        let new_node_ptr = self.sample_node(&mut boxed);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);

        let key = (self as *mut Node as usize, s1_move_index, s2_move_index);
        children.insert(key, boxed);
        new_node_ptr
    }

    pub unsafe fn backpropagate(
        &mut self,
        score: f32,
        terminal: bool,
        state: &mut State,
        collector: &mut LeafCollector,
    ) {
        self.times_visited += 1;
        self.value_sum += score;
        if terminal {
            self.terminal_visits += 1;
            self.terminal_value_sum += score;
        }
        if !self.root && (2..=8).contains(&self.depth) && self.times_visited >= 16 {
            collector.eligible_seen += 1;
            // Reservoir sample across eligible deep nodes, rather than
            // retaining the first early winning branch encountered.
            if rng().random_range(0..collector.eligible_seen) == 0 {
                self.sampled_features = Some(crate::learned_value::extract_features_vec(state));
                collector.sample = Some(self as *mut Node);
                collector.sampled_state = Some(state.clone());
            }
        }
        if self.root {
            return;
        }

        let parent_s1_movenode =
            &mut (*self.parent).s1_options.as_mut().unwrap()[self.s1_choice as usize];
        parent_s1_movenode.total_score += score;
        parent_s1_movenode.visits += 1;

        let parent_s2_movenode =
            &mut (*self.parent).s2_options.as_mut().unwrap()[self.s2_choice as usize];
        parent_s2_movenode.total_score += 1.0 - score;
        parent_s2_movenode.visits += 1;

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(score, terminal, state, collector);
    }

    pub fn rollout(&mut self, state: &mut State, context: &RolloutContext) -> (f32, bool) {
        let battle_is_over = state.battle_is_over();
        if battle_is_over == 0.0 {
            let eval = evaluate(state);
            let hand_value = sigmoid(eval - context.hand_root_eval);
            if let Some(root_logit) = context.learned_root_logit {
                if let Some(learned_value) = learned_rollout_value(state, root_logit) {
                    return (
                        hand_value * (1.0 - context.learned_weight)
                            + learned_value * context.learned_weight,
                        false,
                    );
                }
            }
            (hand_value, false)
        } else {
            if battle_is_over == -1.0 {
                (0.0, true)
            } else {
                (battle_is_over, true)
            }
        }
    }
}

#[derive(Debug)]
pub struct MoveNode {
    // root prior (already scaled by c_puct); None = plain UCB1 node
    pub prior: Option<f32>,
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
}

impl MoveNode {
    pub fn ucb1(&self, parent_visits: u32) -> f32 {
        // every option gets one forced visit (stock behavior)
        if self.visits == 0 {
            return f32::INFINITY;
        }
        // PUCT when a (c_puct-scaled) root prior is attached; UCB1 otherwise
        if let Some(cp_prior) = self.prior {
            return (self.total_score / self.visits as f32)
                + cp_prior * (parent_visits as f32).sqrt() / (1.0 + self.visits as f32);
        }
        let score = (self.total_score / self.visits as f32)
            + (2.0 * (parent_visits as f32).ln() / self.visits as f32).sqrt();
        score
    }
    pub fn average_score(&self) -> f32 {
        let score = self.total_score / self.visits as f32;
        score
    }
}

#[derive(Clone)]
pub struct MctsSideResult {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
}

impl MctsSideResult {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        let score = self.total_score / self.visits as f32;
        score
    }
}

pub struct MctsResult {
    pub s1: Vec<MctsSideResult>,
    pub s2: Vec<MctsSideResult>,
    pub iteration_count: u32,
    pub leaf_sample: Option<MctsLeafSample>,
}

pub struct MctsLeafSample {
    pub root_features: Vec<f32>,
    pub leaf_features: Vec<f32>,
    pub target: f32,
    pub depth: u8,
    pub terminal_visits: u32,
    pub all_visits: u32,
    pub sampled_state: State,
}

struct LeafCollector {
    sample: Option<*mut Node>,
    eligible_seen: u32,
    sampled_state: Option<State>,
}

fn mcts_iteration(
    root_node: &mut Node,
    state: &mut State,
    context: &RolloutContext,
    children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
    collector: &mut LeafCollector,
) {
    let (mut new_node, s1_move, s2_move) = unsafe { root_node.selection(state, children) };
    new_node = unsafe { (*new_node).expand(state, s1_move, s2_move, children) };
    let (rollout_result, terminal) = unsafe { (*new_node).rollout(state, context) };
    unsafe { (*new_node).backpropagate(rollout_result, terminal, state, collector) }
}

enum SearchLimit {
    Time(Duration),
    Iterations(u32),
}

fn run_mcts_loop(
    root_node: &mut Node,
    state: &mut State,
    context: &RolloutContext,
    children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
    collector: &mut LeafCollector,
    limit: SearchLimit,
) {
    let start_time = std::time::Instant::now();
    loop {
        let batch_size = match limit {
            SearchLimit::Iterations(n) => n.saturating_sub(root_node.times_visited).min(1000),
            SearchLimit::Time(_) => 1000,
        };
        if batch_size == 0 {
            break;
        }
        for _ in 0..batch_size {
            mcts_iteration(root_node, state, context, children, collector);
        }
        if root_node.times_visited >= 10_000_000 {
            break;
        }
        match limit {
            SearchLimit::Time(max_time) => {
                if start_time.elapsed() >= max_time {
                    break;
                }
            }
            SearchLimit::Iterations(n) => {
                if root_node.times_visited >= n {
                    break;
                }
            }
        }
    }
}

/// Continues an independently owned leaf state to a true simulator terminal.
/// Each decision uses a fresh, bounded MCTS search and samples one engine
/// stochastic outcome, so this never observes the logged battle's future.
pub fn rollout_leaf_to_terminal(
    mut state: State,
    rollout_iterations: u32,
    max_decisions: u32,
) -> Option<f32> {
    for _ in 0..max_decisions {
        let terminal = state.battle_is_over();
        if terminal != 0.0 {
            return Some(terminal);
        }

        let (s1_options, s2_options) = state.root_get_all_options();
        let mut planning_state = state.clone();
        let result = perform_mcts(
            &mut planning_state,
            s1_options,
            s2_options,
            Duration::ZERO,
            rollout_iterations,
            None,
            None,
            2.0,
        );
        let s1_move = result
            .s1
            .iter()
            .max_by_key(|entry| entry.visits)?
            .move_choice
            .clone();
        let s2_move = result
            .s2
            .iter()
            .max_by_key(|entry| entry.visits)?
            .move_choice
            .clone();
        let outcomes = generate_instructions_from_move_pair(&mut state, &s1_move, &s2_move, false);
        let weights: Vec<f64> = outcomes
            .iter()
            .map(|outcome| outcome.percentage.max(0.0) as f64)
            .collect();
        let distribution = WeightedIndex::new(weights).ok()?;
        let choice = distribution.sample(&mut rng());
        state.apply_instructions(&outcomes[choice].instruction_list);
    }

    let terminal = state.battle_is_over();
    (terminal != 0.0).then_some(terminal)
}

pub fn perform_mcts(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
    s1_priors: Option<Vec<Option<f32>>>,
    s2_priors: Option<Vec<Option<f32>>>,
    c_puct: f32,
) -> MctsResult {
    let mut root_node = Node::new();
    unsafe {
        root_node.populate(side_one_options, side_two_options);
    }
    root_node.root = true;
    root_node.assign_root_priors(s1_priors, s2_priors, c_puct);
    let mut children: HashMap<(usize, usize, usize), Box<[Node]>> = HashMap::new();
    let root_features = crate::learned_value::extract_features_vec(state);
    let mut collector = LeafCollector {
        sample: None,
        eligible_seen: 0,
        sampled_state: None,
    };

    let context = RolloutContext {
        hand_root_eval: evaluate(state),
        learned_root_logit: learned_logit(state),
        learned_weight: learned_weight(),
    };
    let search_limit = if max_iterations > 0 {
        SearchLimit::Iterations(max_iterations)
    } else {
        SearchLimit::Time(max_time)
    };
    run_mcts_loop(
        &mut root_node,
        state,
        &context,
        &mut children,
        &mut collector,
        search_limit,
    );

    let leaf_sample = collector.sample.map(|node| unsafe {
        MctsLeafSample {
            root_features,
            leaf_features: (*node).sampled_features.clone().unwrap_or_default(),
            target: if (*node).terminal_visits > 0 {
                (*node).terminal_value_sum / (*node).terminal_visits as f32
            } else {
                (*node).value_sum / (*node).times_visited as f32
            },
            depth: (*node).depth,
            terminal_visits: (*node).terminal_visits,
            all_visits: (*node).times_visited,
            sampled_state: collector
                .sampled_state
                .take()
                .expect("sampled node state missing"),
        }
    });

    let result = MctsResult {
        s1: root_node
            .s1_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        s2: root_node
            .s2_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        iteration_count: root_node.times_visited,
        leaf_sample,
    };

    result
}
