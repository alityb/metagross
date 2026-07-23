# Current Best System: ExIt r1

## Result

| Metric | Value |
|---|---:|
| Account | `metaexitr1` |
| Format | `gen9randombattle` |
| Settled GXE | 92.4-92.7 |
| Rating deviation | 25 |
| Peak GXE observed | 93.6 |
| Prior best | 91.4 GXE |
| Improvement | about +1.1 to +1.3 GXE |

The public ladder runner is stopped. This result was obtained before the
evidence-gated round protocol in `docs/expert_iteration_protocol.md` existed;
be reused as a template without the caveats below.

## Deployed Agent

The live agent was the patched Foul Play search agent with a Metamon policy
server providing root priors.

| Component | Deployed choice |
|---|---|
| Search engine | Foul Play / poke-engine, patched PUCT-prior interface |
| Agent kind | `foul_play_root_priors_opp` |
| Search budget | 500ms, parallelism 8, one search thread |
| Belief worlds | Foul Play adaptive 16 worlds at 500ms or 32 at 250ms |
| PUCT coefficient | 2.0 |
| Policy checkpoint | `randbats_exit_r1`, epoch 5 |
| Deployment checkpoint path | `src/nets/checkpoints/randbats_full/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt` |
| Deployment parameter check | 642 compatible keys, 142,832,563 parameters |

Foul Play supplies the Pokémon rules/search infrastructure. The policy server,
root-prior injection, self-play generation, replay pipeline, ExIt fine-tuning,

## r1 Training Provenance

The r1 policy was fine-tuned from Kakuna/Metamon using an early self-play pile
and a human replay anchor.

| Item | r1 value |
|---|---:|
| Captured self-play replays | 12,960 |
| Parsed self-play POV trajectories | 23,870 |
| Human parsed trajectories available | about 23.5k |
| Training mix | self-play-heavy with retained human data (documented as 90/10) |
| Fine-tune epochs | 6 |
| Training hardware | H200 |
| Rating conditioning | enabled (`A_rating`) |

The original self-play search used 250ms / parallelism 1, materially weaker
than the deployed 500ms / parallelism 8 search. Despite that weak expert, r1
improved the public result. This motivates stronger-search data collection, but
does not prove a later round will improve.

## Verified Versus Configured

### Verified

- r1 deployment checkpoint loads through the real prior-server path.
- The root policy prior (C1) is active in the deployed search configuration.
- The 92.4-92.7 GXE result is settled at RD 25.
- The recovered poke-engine deployment wheel was checked against the original
  binary in a 100-game local equivalence test (47/100, not distinguishable from
  50%).

### Not retrospectively verified

- Opponent-policy priors (C2) were configured through
  `foul_play_root_priors_opp`, but historical server telemetry was not retained.
  Later AWS debugging exposed an opponent-view adapter bug. Do not claim a
  measured C2 contribution until new strict data reports nonempty C2 coverage.
- r1 data does not have the strict shard manifests required by the new protocol.
- r1 was judged primarily by the public ladder. It did not first pass the
  current paired H2H promotion gate.

## What Is Frozen

Treat this exact deployment as the baseline for every future candidate:

1. Do not overwrite the r1 checkpoint.
2. Retain the validated r1 self-play pile as a versioned legacy replay buffer;
   do not relabel it as strict strong-search data. Mixed/fallback r2 smoke data
   remains archive-only.
3. Compare candidates to r1 in paired H2H at identical 500ms/P8 budget before
   public ladder use.
4. Report both GXE/RD and H2H confidence intervals; neither alone is enough.

## Next Candidate Rules

The first candidate after r1 must use the protocol in
`docs/expert_iteration_protocol.md`:

- strict root-prior success, measured C2 status, and parse validation;
- human anchor retained in the dataset;
- no underscore usernames, preserving two POV trajectories per battle;
- checkpoint-pool/PFSP-lite data instead of self-only collection;
- paired H2H promotion before a bounded fresh-account ladder check.
