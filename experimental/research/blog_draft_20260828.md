# [DRAFT] I built a Pokémon agent with RL. Here's how every architecture choice actually behaved.

*Owner's editing notes are in [brackets]. All numbers are final and
traceable to `experimental/runs/iteration_log.md`. Nothing is pending.*

---

Competitive Pokémon is a genuinely hard RL problem: ~10^564 possible
states, partial observability (you can't see the opponent's team), heavy
stochasticity (moves miss, damage rolls vary), and an adversary. I built
an agent for Showdown's most-played format, `gen9randombattle`, and it
plays at elite human level: a settled **92.4–92.7 GXE** (GXE ≈ expected
win rate against a random ladder player) at full rating convergence, a
peak of 93.6, and — the part I'm proudest of — it **reproduces**: months
later, fresh account, formal protocol, 91.7. Its accounts sat at **rank
#105 and #246** of the public leaderboard (hashed snapshot, 2026-08-28),
a bounded push later set an all-time peak of **Elo 2320** (≈rank #40
territory), and its victims include the players ranked **#2 and #6** at
their respective snapshots.

This post is the architecture tour. For every major design choice I'll
show you what the alternatives were, how each one behaved when measured
properly, and the numbers — because almost none of them behaved the way
I expected, and the ways they surprised me generalize well beyond
Pokémon.

## The architecture

Three components:

1. **A learned policy**: a 142M-parameter transformer, trained with
   offline RL on human replays (starting from
   [Metamon](https://github.com/UT-Austin-RPL/metamon)'s open model),
   then improved with one round of expert iteration — my agent's search
   played thousands of self-play games, and the policy was fine-tuned on
   what the search chose and how it turned out.
2. **A belief layer**: since the opponent's team is hidden, the agent
   samples 8–32 plausible opponent teams ("worlds") from the format's
   team generator, filtered by everything revealed so far.
3. **Search**: each sampled world gets 500 ms of Monte-Carlo tree search
   in a Rust engine, with the policy's action distribution as the root
   prior. Results are aggregated across worlds by visit counts, and the
   most-visited action is played.

Policy tells search where to look; search plays out consequences across
possible hidden worlds; visit aggregation picks the move. And the search
genuinely earns its keep: across ≈18,800 telemetered decisions in three
independent datasets (both serving modes), the final move differs from the
policy's top pick **41–48% of the time**. The policy proposes; the search disposes — half
the time differently. Every choice in that stack has alternatives. Here is
what happened when I measured them.

## Choice 1: do the learned priors even matter?

The search stack without the neural policy is
[Foul Play](https://github.com/pmariglia/foul-play) — same engine, same
world sampling, heuristic guidance instead of learned priors. It's a
strong open-source agent on its own. So what do 142M parameters of
learned guidance buy?

At **population level: a lot** — and we measured it rather than trusting
folklore. Laddering the unguided stack itself (identical engine, identical
budget, zero priors) converged at **87.5–87.9 GXE** over ≈200 games, so
the priors are worth **≈4 GXE** on the real ladder (91.7 vs ≈87.8) — half
the community's assumed 7–10, but the difference between a strong account
and a top-100 one. (Also humbling: the unguided stack alone brushed Elo
2381 mid-run — search does most of the work.)

**Head-to-head: nothing.** In a preregistered 200-game match (mirrored
teams, alternating sides), priors-vs-no-priors went **88-112 — 44.0%,
CI [37.3, 50.9]**. The confidence interval's upper edge sits exactly at
parity: the guidance that carries the agent past the entire human ladder
is worth *at best nothing* against its own skeleton.

Hold that contradiction. It's not a fluke — it's the post's recurring
result, and by the end it has a mechanism.

## Choice 2: should the policy read the battle history?

The transformer was trained reading full battle trajectories, but you can
serve it two ways: **causal** (reads the whole history) or **stateless**
(same weights, current state only). Same network, one input-format
switch. We ran both to identical rating convergence — fresh accounts,
same machine, same deterministic search budget:

| Serving mode | Record | GXE | Glicko ± RD |
|---|---|---|---|
| history-reading | 149-81 | 89.4 | 1902 ± 25 |
| stateless | 174-83 | 91.7 | 1952 ± 25 |

Reading history made the agent slightly *worse* at population level — and
head-to-head over 200 games the two versions are dead even (48%). Both
facts demanded explanation.

**The mechanism**: the history-reading view miscalibrates itself. Its
action-distribution entropy collapses as the game goes on — ≈0.96 nats
early to ≈0.58 by turn 30+, replicated in two independent datasets —
while the stateless view of the *same network* on the *same states* stays
flat. More context makes the model more confident, far beyond what the
positions justify.

The obvious explanation is the LLM-flavored one: late-game context is
mostly the agent's own past decisions, so the network reads its own
confident choices and doubles down. Pokémon offers the counterfactual
language models can't run — identical network, with and without its own
history, on identical states — so I preregistered the ablation, froze the
decision thresholds, and ran it. Offline replay was exact: 0.0 maximum
probability difference against the live logs across all 735 decisions.

**It refuted me.** Masking the agent's own past actions (opponent-visible
history intact, full context length) removes only ~10% of the collapse:

| Condition | Entropy collapse (nats) |
|---|---|
| full history | 0.388 |
| truncated to 5 / 10 / 20 / 40 steps | 0.223 / 0.227 / 0.238 / 0.310 |
| own actions masked, full length | **0.351** |
| stateless | 0.093 |

The driver is the *volume of observed history itself*: collapse grows
monotonically with window size, and even five steps of context produces
1.7× the same-state stateless collapse. Not self-conditioning —
history-induced sharpening. The sexy story is simply wrong, and the
preregistration made not telling you impossible.

Three confounds could fake this result, so we killed each one on 735
true same-state pairs (both views of the same network, byte-identical
inputs). *Fewer legal moves late-game?* Normalizing entropy by available
actions makes the gap cleaner — the stateless view gets **more**
uncertain late (0.443 → 0.492) while the history view sharpens
(0.432 → 0.341). *State selection?* Identical states by construction.
*Just a temperature scalar?* Temperature preserves the action ranking —
but the two views pick **different top moves on ~35% of identical
states**, flat across the game, and ~0.13 nats of divergence survive the
optimal per-decision temperature fit. History doesn't just sharpen the
network's beliefs; it *changes* them. Which retroactively explains the
flattening failure in Choice 4: temperature can restore the entropy, but
not the beliefs underneath it.

**Verdict on the choice:** history conditioning buys nothing at
deployment that the belief-plus-search machinery doesn't already provide,
and it induces a real calibration pathology. The stateless serve of the
same weights is the better agent.

## Choice 3: how should search turn visits into a move?

Given an overconfident prior, the textbook fix is to stop letting it
dominate action selection. Gumbel-style **completed-Q** — a proven
improvement in perfect-information MCTS — decides by
`log π(a) + scaled Q̂(a)` instead of raw visit counts. Preregistered,
50-game screen: it flipped 37% of decisions and went **10-40**.

The autopsy is the valuable part. Completed-Q assumes Q-estimates from
one true world. Mine are pooled across sampled hidden worlds, and acting
greedily on values averaged over inconsistent hypotheses is the classic
*strategy fusion* fallacy from the imperfect-information literature —
resurfacing one level up, in the decision rule. Visit counts turn out to
be doing quiet robustness work: they only concentrate where many worlds
independently agree.

**Verdict:** under determinization, visit counts are a robustness
mechanism, not just an exploration record. Don't replace them with
value-greedy rules that assume a single world.

## Choice 4: can you fix the calibration directly?

If the history-reading prior is overconfident, flatten it: a turn-indexed
temperature schedule, calibrated offline to restore the early-game
entropy profile. This one *worked* — telemetry confirmed restored search
breadth, and it beat the stateless champion **111-89 (55.5%)** over 200
games. Fifth attempt before I got a valid measurement, but it passed.
Ship it?

I shipped it to the real ladder. It lost **eight GXE** (81.7 vs the 89.4
baseline over ≈139 games).

Bucketing every ladder game by opponent Elo at match start points at
why (with a caveat: matchmaking is endogenous — a lower-rated account
draws weaker pools — so within-bucket comparisons are suggestive, Fisher
p = 0.03 on the key cell, while the converged *ratings* above are the
matchmaking-invariant evidence):

| Opponent Elo | flattened | unflattened |
|---|---|---|
| below 1600 | 34-16 (68%) | 29-3 (91%) |
| 1600+ | 52-30 (63%) | 121-74 (62%) |

Against strong opponents, flattening costs *nothing*. The measurable
deficit concentrates against weak opponents, where the extra exploration
burns games a decisive line would have won. **Search breadth is not a calibration
parameter with a correct value — it's a strategic variable whose optimum
depends on who created the position.** "Well-calibrated" is undefined
without a population.

## The pattern all four choices share

Look back at the measurements:

- Learned priors: measured ≈4 GXE population edge, **no head-to-head
  edge** (44.0%).
- History conditioning: population deficit, **head-to-head parity**
  (48%).
- Temperature flattening: **head-to-head win** (55.5%), population
  disaster (−8 GXE).

Three times, head-to-head and population measurements didn't just
disagree in size — they disagreed in *direction*. The mechanism is
consistent: architecture choices that trade decisiveness for robustness
pay off against strong, systematic opponents and bleed against the weak,
erratic majority that a real population mostly is. Which means the
standard way agents get evaluated — beat a strong baseline head-to-head —
can point *backwards* relative to deployed performance.

So I built the evaluator that can't fail this way: a frozen league where
every candidate plays mirrored pairs against a fixed pool spanning
strength and style, graded on the per-opponent *vector*, never one
number, with weak-opponent cells as the exploitation guard. Its
calibration run reproduced the known numbers, and its retrodiction
correctly **rejected** the flattening change that head-to-head had
approved — in one overnight of free local compute, catching what took 139
ladder games to expose. (The half that didn't reproduce at 40-game power —
the head-to-head win itself — is reported in the repo per prereg, because
that's the deal.)

This lands beyond Pokémon. The
[PokeAgent Challenge](https://pokeagentchallenge.com) (NeurIPS 2025, now
an open benchmark — [paper](https://arxiv.org/abs/2603.15563)) found the
same architecture-level shape (RL and search specialists decisively beat
LLM generalists at battling), and arena-style, agent-vs-agent ranking is
becoming the default evaluation everywhere. The three sign flips above
are a documented failure mode of exactly that measurement family.

## Watch it play

Opponents upload replays, so some of the peak-run games are public. Two
worth two minutes each:

- [vs "fable foul play" (Elo 2379, then ranked ~#16)](https://replay.pokemonshowdown.com/gen9randombattle-2673127492)
  — judging by the name, another AI (an LLM-flavored fork of the same
  open-source search stack). Very possibly a bot-vs-bot match in the
  ladder's top 20; my agent won this one and lost the rematch.
- [vs pokeblade☆101 (Elo 2420, ranked #84)](https://replay.pokemonshowdown.com/gen9randombattle-2673131838)
  — a clean win against a top-100 human at 2400+.

## By the numbers

| | |
|---|---|
| Measured games this campaign | **≈1,830** (≈760 public-ladder, ≈1,070 controlled local) |
| Decisions made | ≈56,000 (30.5 per game) |
| Engine iterations per decision | ≈7–15 million (8–32 worlds × 236–472k each) |
| Campaign total engine iterations | order 10¹¹ |
| Search-override rate | 48.1% of decisions |
| Cloud spend | <$30 (everything else ran on one Mac) |
| Invalidated runs, documented | 5 (all in the repo) |

## What didn't matter (measured, so you don't have to)

Two whole programs of obvious improvements died before this post's story
started, and the numbers deserve their footnote: **deeper search
saturates** — 8,192- and 20,000-iteration search agree on 86.95% of top-1
actions; and **sharper hidden-team inference is near its ceiling** — a
learned action-likelihood model reached validation NLL 1.3305 against a
1.3318 marginal (i.e., nothing to learn at shallow context), and the only
real belief win was better consistency filtering (+4.7 posterior mass).
One humbling corpus result points at what's actually missing: in states
where the sampled worlds disagree, expert humans match the majority
world's recommendation 36%, a minority world's 31%, and *no sampled world
at all* 33% — chance across the board. Human play under ambiguity isn't
"solve worlds, then choose among their answers." Robust cross-world
aggregation is the capability this whole architecture family lacks — and
the one every failed fix above was accidentally trading away.

## The part nobody tells you: silent mechanisms

Getting these numbers took more attempts than I want to admit, and the
recurring villain was always *the mechanism that silently isn't on*: an
intervention hook that imported cleanly and did nothing; a screen that
modified both arms because a flag wasn't passed; and, best of all, the
team-mirroring layer under months of "mirrored" screens that turned out
to require an env var on the game server's side nobody had set — actual
battle teams vs registered manifests, 0/6 pairs mirrored. (Results stood,
because they'd been analyzed as unpaired games, which is what they were —
but the variance reduction I designed for never existed, which is why my
sequential tests kept running to their caps.)

The rule that came out of it: **observable activation**. No measurement
counts unless the artifact it was supposed to affect proves the mechanism
fired — and proves it *didn't* fire in the control. Enforce consumption,
not registration. It sounds like bookkeeping; it's the difference between
measuring your idea and measuring your assumptions.

## What I'd tell you to steal

1. If your policy conditions on history, measure its calibration against
   a stateless serve of the same weights. My collapse was huge,
   replicated, invisible until I looked — and its cause (context volume,
   not the agent reading its own actions) was the opposite of the
   intuitive story.
2. Under determinization, visit counts are a robustness mechanism. Don't
   swap them for value-greedy decision rules that assume one world.
3. Evaluate against a population and stratify by opponent strength.
   Head-to-head against a strong baseline — the thing everyone reports —
   pointed backwards three separate times here.
4. Observable activation, everywhere, always.

*Everything — preregistrations, invalidated runs included, raw logs,
hashed leaderboard snapshots, the league harness, and figure-ready data
(rating trajectories per arm, per-decision entropy curves, override
stats, under `experimental/research/blog_figures/`) — is in the
[repo](https://github.com/alityb/metagross). The full research record is
in the final report; the iteration log has every hash.*
