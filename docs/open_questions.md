# Open Questions

## Architecture

- Can Foul Play expose or preserve a reusable MCTS tree, or must speculative search cache only derived root policies?
- Can `poke-engine` produce successor states for hypothetical joint actions in a form that can be matched to Foul Play battle objects?
- What is the canonical public-state hash for a Showdown battle observation?
- How should hidden sampled worlds be represented without leaking into final action selection?

## Performance

- What is the actual distribution of opponent think-time on public ladder?
- How much idle time exists in local bot-vs-bot H2H, and is it representative?
- What cache hit rate is achievable with top-k opponent action speculation?
- Does background work interfere with foreground decision latency on local hardware?

## Evaluation

- What is the current clean stock Foul Play gen9 public ladder baseline under the same account/budget discipline?
- What wall-clock budget should be considered fair when using opponent think-time?
- Should the first speculative-search gate be 25ms, 100ms, or adaptive budget?

## Data And Models

- Is there enough teacher data to train a useful draft policy for gen9, or should the first draft policy be stock cheap search?
- Can randbats generator priors be calibrated before use in search?
- What state/action schema should unify Metamon and Foul Play traces?

## Engineering

- Which old Foul Play patches should be removed from the MVP path to avoid interference?
- Should the MVP run in a fresh venv and fresh Showdown checkout to prevent historical build drift?
- How should secrets and ladder credentials be passed without process-argument leakage?

## Product/Claim

- Is the desired claim “best bot,” “better than Foul Play,” or “novel search-efficiency technique”?
- What format is the target for a public post: `gen9randombattle`, `gen1ou`, or both?
- What is the minimum acceptable ladder evidence after local H2H success?
