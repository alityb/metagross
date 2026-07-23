# Production Sources

- `metagross/`: production-only launcher, policy server, and Foul Play adapter.
- `vendor/poke-engine/`: patched 0.0.47 engine source used by the accepted bot.
- `patches/`: minimal compatibility patches for pinned upstream dependencies.
- `models/`: ignored local checkpoint artifacts.
- `runtime/`: ignored caches and transient runtime state.

No experimental agent is registered here. See `docs/architecture.md` for the
runtime contract.
