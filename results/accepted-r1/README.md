# Accepted r1 Result

`randbats_exit_r1` epoch 5 is the project's accepted and strongest validated
Pokemon Showdown agent.

| Field | Value |
|---|---|
| Account | `metaexitr1` |
| Format | `gen9randombattle` |
| Settled GXE | 92.4-92.7 |
| Rating deviation | 25 |
| Peak observed GXE | 93.6 |
| Agent | Foul Play plus Metamon root priors |
| Search | 500 ms, parallelism 8, one thread, `c_puct=2.0` |

The machine-readable result is in [`result.json`](result.json), and artifact
digests are in [`artifacts.json`](artifacts.json).

This was a public-ladder observation, not a controlled paired H2H estimate. It
predates the project's later formal gate, and the raw logs do not provide one
immutable final rating snapshot. The reported settled range and RD are retained
from the contemporaneous deployment record. See
[`docs/provenance.md`](../../docs/provenance.md).
