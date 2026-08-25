# Shared-Root Replay Capture

The `poke-engine-0.0.47-shared-root-v7` contract attaches a versioned replay
capture to every weighted shared-root RM+ result. Constructing it does not add
searches, consume randomness, or change the selected policy. It does add
serialization, transfer, hashing, and private-dump I/O overhead. A capture or
envelope larger than 16 MB fails closed, and no game is authorized until
representative capture-latency evidence passes a separately preregistered gate.

## Captured Data

Each native result records:

- the solver configuration and ordered own-action support;
- normalized player and per-particle opponent priors;
- canonical serialized particles and normalized weights;
- the original positive input indices and weights merged into each canonical
  particle;
- ordered opponent-action supports;
- the complete `[particle][own action][opponent action]` payoff game;
- every continuation seed, requested and executed iteration count, visit count,
  total score, payoff, and IEEE-754 bit pattern;
- player policy, counterfactual values, per-particle opponent policies,
  diagnostics, and consistency digests.

When `METAGROSS_SEARCH_DUMP` is enabled, search-row schema 4 also records the
original source states, their SHA-256 identities, source and normalized weights,
request-authorized action aliases, solver/action/world seeds, priors, request
IDs, and source/wheel identities. The envelope has a canonical SHA-256
self-hash and binds the native capture by hash. Validation also binds duplicate
prior, seed, request, and engine fields to the search row; the self-hash is an
integrity checksum, not an authenticity signature.

Persisted schema-3 rows remain readable for historical audits, but they are not
exactly replayable because they contain state hashes instead of serialized
states. Live v7 remote responses fail closed if the native replay capture is
missing.

## Validation

Validate a schema-4 search dump without running the solver:

```bash
python3.11 -m srcs.metagross.shared_root_capture \
  --input search.jsonl \
  --output capture-audit.json
```

Add `--rerun` to reconstruct every state, rerun the native search with the
captured inputs, and require exact equality of the policy, matrices,
continuation metadata, diagnostics, and sampled mixed-policy action.

The frozen 26-root native-capture validation is recorded in
`experimental/runs/search_native_stage2_20260809/replay-capture-contract-validation-v1.json`.
The separate schema-4 validation report exercises the complete envelope,
source mapping, action draw, provenance checks, and `--rerun` path:
`experimental/runs/search_native_stage2_20260809/replay-schema4-exact-validation-v2.json`.
Its protocol binds the runner and every imported validation module by SHA-256.

## Handling

Replay captures contain hidden-team hypotheses and must remain private. Do not
publish search dumps or include them in dashboards. Captures contain derived
seeds but never the private run seed, credentials, bearer tokens, or Showdown
passwords.
