# [DRAFT] The same network, twice: what a Pokémon agent taught us about self-conditioning, evaluation, and the fixes that lie

*Owner's editing notes are in [brackets]. All numbers are final and
traceable to `experimental/runs/iteration_log.md`. Nothing is pending.*

---

I have a Pokémon agent that plays `gen9randombattle` at elite human level —
settled 92.4–92.7 GXE on the Showdown ladder at the RD-25 convergence
floor, peak 93.6, and (the part I'm proudest of) it *reproduces*: months
later, fresh account, formal protocol, 91.7. To my knowledge it's the
strongest reported agent for the format — its two measurement accounts
sat at **rank #105 and #246** on the public top-500 leaderboard (hashed
snapshot, 2026-08-28), a later bounded push set an all-time peak of
**Elo 2320** (≈rank #40 territory), and its victims include the players
ranked **#2** and **#6** at their respective snapshots. This post is not
about how good it is. It's about a month of trying to make it better, failing in
increasingly interesting ways, and what the failures turned out to be
worth.

The agent is a 142M-parameter transformer policy (one expert-iteration
round on top of [Metamon](https://github.com/UT-Austin-RPL/metamon)'s
offline-RL model) guiding determinized Monte-Carlo search: sample 8–32
plausible hidden opponent teams, search each world for 500 ms with the
policy as root prior, aggregate. Standard imperfect-information stack,
unusually measured.

Some context for where this sits: the
[PokeAgent Challenge](https://pokeagentchallenge.com) (NeurIPS 2025's
competition track, now an open benchmark —
[paper](https://arxiv.org/abs/2603.15563)) ran the largest AI Pokémon
tournament to date and found the same shape this post does: RL and search
specialists decisively beat LLM generalists at battling. Their live
leaderboard runs team-building OU formats rather than random battles, so
metagross can't enter as-is — its policy and belief machinery are keyed to
the randbats team generator. But one finding below travels in their
direction: arena-style benchmarks rank agents largely by agent-vs-agent
play, and we measured — three separate times — that head-to-head rankings
can be anti-correlated with performance against a real population. If you
build or consume arena benchmarks, that failure mode is worth designing
against.

## One network, two views

The policy can be served two ways: *causal* — it reads the whole battle
history, the way it was trained — or *stateless* — same weights, current
state only. Which is better? We ran both to identical convergence on fresh
ladder accounts, same machine, same deterministic search budget, one
variable:

| Serving mode | Record | GXE | Glicko ± RD |
|---|---|---|---|
| causal-history | 149-81 | 89.4 | 1902 ± 25 |
| stateless | 174-83 | 91.7 | 1952 ± 25 |

Stateless slightly ahead. But head-to-head over 200 games, the two are
dead even: 48%. Hold that discrepancy — population says one thing,
head-to-head says another. It looked like noise. It wasn't.

## History makes the network overconfident — but not the way we guessed

Why would reading history *hurt*? We instrumented the priors and found
something clean: the causal view's entropy collapses as the game goes on —
from ≈0.96 nats early to ≈0.58 by turn 30+, replicated in two independent
datasets. The stateless view of the *same network* on the *same states*
stays flat.

Our favorite hypothesis wrote itself: late in a battle, most of the
context is the agent's own past decisions — the network sees a trajectory
full of its own confident choices and doubles down, the game-playing
analog of an LLM committing to its own prior outputs. Pokémon even gives
you the counterfactual language models can't: the identical network,
with and without its own history, on identical states. So we preregistered
the ablation, froze the thresholds, and ran it.

**It refuted us.** Masking the agent's own past actions (opponent-visible
history intact, full context length) removes only ~10% of the collapse:

| Condition | Entropy collapse (nats) |
|---|---|
| full history | 0.388 |
| truncated to 5 / 10 / 20 / 40 steps | 0.223 / 0.227 / 0.238 / 0.310 |
| own actions masked, full length | **0.351** |
| stateless | 0.093 |

The driver is the *volume of observed history itself*: the collapse grows
monotonically with window size, and even five steps of context produces
2.4× the stateless collapse. Not self-conditioning — history-induced
sharpening. The replay was exact (0.0 max probability difference against
the online logs, all 735 decisions), so this isn't probe noise; the sexy
story is just wrong, and I'm telling you because the prereg made not
telling you impossible.

## Two principled fixes, two instructive failures

A real mechanism suggests real fixes. We tried the two obvious ones,
both training-free, both preregistered.

**Fix the decision rule.** If the prior is overconfident, stop letting it
dominate action selection: Gumbel-style completed-Q — proven in perfect-
information MCTS — decides by `log π(a) + scaled Q̂(a)` instead of visit
counts. It flipped 37% of decisions and went **10-40**. The autopsy is the
interesting part: completed-Q assumes Q-estimates from one true world.
Ours are pooled across sampled hidden-team worlds, so acting greedily on
them is the classic *strategy fusion* fallacy, resurfacing one level up.
Visit counts, it turns out, were quietly doing robustness work: they only
concentrate where many worlds independently agree.

**Fix the calibration.** Flatten the prior back to its early-game entropy
with a turn-indexed temperature schedule, calibrated offline. This one
*worked*: mechanism telemetry confirmed restored search breadth, and it
beat the stateless champion **111-89 (55.5%)** over 200 games. Fifth
attempt to measure it validly, but it passed. Ship it?

We shipped it to the ladder. It lost **eight GXE** (81.7 vs the 89.4
baseline, ≈139 games).

## The sign flip

An intervention that wins against a strong opponent and craters against
the population — with zero opponent-adaptation anywhere in it. And we'd
seen this shape before: causal-vs-stateless was parity head-to-head and a
gap on the ladder. Twice in one campaign, head-to-head and population
measurements didn't just disagree in size. They disagreed in *direction*.

The stratification nails the mechanism. Bucket every ladder game by
opponent Elo at match start:

| Opponent Elo | flattened | plain-causal |
|---|---|---|
| below 1600 | 34-16 (68%) | 29-3 (91%) |
| 1600+ | 52-30 (63%) | 121-74 (62%) |

Against strong opponents, flattening costs *nothing* — indistinguishable.
The entire deficit lives below 1600: flattening plays weak opponents like
strong ones. Extra late-game exploration is positive-EV against an
opponent who punishes your overconfidence, and negative-EV against
opponents who blunder first if you just play the decisive line. **Search
breadth is not a calibration parameter. It's a strategic variable whose
optimum depends on who created the position.** "Well-calibrated" is
undefined without a population.

The uncomfortable corollary: single-opponent evaluation — the thing every
game-agent paper reports — can be *anti-correlated* with deployed
performance. Not noisy. Backwards.

## So we built the evaluator that can't lie this way

A frozen league: every candidate plays mirrored pairs against a fixed pool
spanning strength and style — the strong stateless peer, a self-mirror
sanity arm, heuristic no-prior search, and a weak hyper-aggressive bot —
and is graded on the *per-opponent vector*, never one number. Promotion
requires weak dominance over the reference vector, with the weak-opponent
cells as the exploitation guard.

Here's the calibration run (our champion as its own candidate) next to
the retrodiction — flattened-causal through the identical pool, on
identical team seeds, the first locally *verified*-mirrored games in the
project's history:

| Opponent | plain-causal (reference) | flattened |
|---|---|---|
| stateless peer | 62.5% (25-15) | 47.5% (19-21) |
| plain-causal | 45.0% (self-mirror) | 45.0% (direct H2H) |
| vanilla foul-play | 41.7% (10-14) | 33.3% (8-16) |
| max-damage | 91.7% (11-1) | 91.7% (11-1) |

Under the preregistered promotion rule — weak dominance over the
reference vector — flattening is **rejected**: below or equal on every
cell, badly below on the strong peer. The league agrees with the ladder,
where the head-to-head screen had said *ship*. That's the
deployment-relevant half of the retrodiction landing in one overnight of
free local compute. The other half — reproducing the 55.5% head-to-head
win — didn't appear at 40-game power (47.5%, CI [33, 62], which contains
both numbers); preregistration obliges me to tell you that too.

And one cell deserved its own double-take: the *reference* — our
champion, priors and all — went 10-14 against vanilla foul-play, the
identical search stack minus the neural priors. So we preregistered a
powered 200-game test. Verdict: **88-112 (44.0%, CI [37.3, 50.9])** — no
detectable head-to-head edge, upper bound at parity, from the same priors
that carry the agent to **rank #105** on a ladder where the unguided
stack lives 7–10 GXE lower. That's the sign flip's third and cleanest
instance, because this time the "intervention" is the policy guidance
itself — the agent's entire reason to exist. Its value is real, enormous,
and *entirely about exploiting the population*. Head-to-head against its
own skeleton, it measures as nothing.

## The meta-lesson: silent mechanisms

Getting these numbers took more attempts than I want to admit, and the
recurring villain was always the same: *the mechanism that silently isn't
on*. An intervention hook that imported cleanly and did nothing. A screen
that flattened both arms because a mode flag wasn't passed. And the best
one, found this week while root-causing a deterministic harness hang: our
mirrored-team enforcement — the variance-reduction layer under months of
screens — required an env var on the game server's side that our local
launchers never set. Actual battle teams vs registered manifests: 0/6
sampled pairs actually mirrored. The results stand (they were analyzed as
unpaired games, which is what they were), but the power we designed for
silently never existed — which is why our sequential tests kept running to
their caps undecided.

The rule that came out of all of it: **observable activation**. No
intervention result is valid without proof, in the artifact it was supposed
to affect, that it fired — and that it didn't fire in the control. Enforce
consumption, not registration. It sounds like bookkeeping. It's the
difference between measuring your idea and measuring your assumptions.

## What I'd tell you to steal

1. If your policy conditions on history, measure its calibration against
   a stateless serve of the same weights. Our collapse was huge,
   replicated, invisible until we looked — and its cause (context volume,
   not the agent's own actions) was the opposite of the intuitive story.
2. Visit counts are a robustness mechanism under determinization. Don't
   swap them for value-greedy rules that assume one world.
3. Evaluate against a population, stratify by opponent strength, and treat
   any single-opponent number as a diagnostic, not a gate. The sign flip
   is real and it will burn you politely.
4. Observable activation, everywhere, always.

*Everything here — preregistrations, invalidated runs included, raw logs,
the league harness — is in the [repo](https://github.com/alityb/metagross).
The full research record is in the final report; the iteration log has
every hash.*

[Optional closers, pick one: the humans result (in ambiguous states,
human experts match NO sampled world's recommendation above chance — the
whole solve-worlds-then-choose family may be the wrong shape, and robust
aggregation is what our interventions kept trading away) — or a short
"what's next" (aggregation-corrected search, the league as standing gate).]
