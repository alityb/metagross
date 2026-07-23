# Architecture

## Runtime Data Flow

```text
Pokemon Showdown
      |
      v
patched Foul Play client ---- public protocol ----> r1 policy server
      |                                                |
      | determinized worlds                            | player/opponent priors
      v                                                v
patched poke-engine <--------- root PUCT priors -------+
      |
      v
chosen Showdown action
```

The accepted bot has two Python processes:

1. `srcs/metagross/prior_server.py` tracks each public battle protocol, converts
   the current state to Metamon's observation representation, and evaluates
   `randbats_exit_r1` epoch 5. It returns a policy over legal player actions and,
   when available, a policy from the modeled opponent view.
2. `srcs/metagross/run_foul_play.py` runs Foul Play, forwards protocol lines to
   the server, fetches priors before every discretionary search, and passes them
   to the patched poke-engine root.

`srcs/metagross/launch.py` owns both processes and freezes the deployment
parameters.

## Search Contract

- Foul Play creates hidden-information worlds using its standard Random Battle
  inference.
- poke-engine searches each world with root-only PUCT guidance.
- `s1_priors` guides the bot's root action and `s2_priors` guides modeled
  opponent actions when the server can construct them.
- Priors do not replace engine simulation or Foul Play's world aggregation.
- Search must use one engine thread. The accepted engine's multi-thread branch
  does not forward external root priors.
- The production launcher requires player priors. It does not silently degrade
  to unguided Foul Play.

## Excluded Branches

Learned leaf values, MCTS policy distillation, generator-conditioned beliefs,
action-conditioned beliefs, shared-root regret matching, Tauros overrides, and
selective re-solving all remain in `experimental/`. None is enabled or imported
by the production runtime.
