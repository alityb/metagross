# Setup

## Requirements

- macOS or Linux on x86-64/arm64
- Python 3.11
- Rust toolchain and a C compiler
- Git
- Enough RAM for the 142M-parameter policy and parallel Foul Play worlds

## Vendor Sources

The local workspace already contains the pinned checkouts under `srcs/vendor/`.
For a clean checkout, obtain:

```bash
git clone https://github.com/pmariglia/foul-play.git srcs/vendor/foul-play
git -C srcs/vendor/foul-play checkout e1e2ca650598621e85c3b6ab751c66e625489934

git clone https://github.com/UT-Austin-RPL/metamon.git srcs/vendor/metamon
git -C srcs/vendor/metamon checkout 0a00a759c9a4382a2877088d828302ec294a05a5
```

Apply the production compatibility patches:

```bash
git -C srcs/vendor/foul-play apply ../../patches/foul-play-r1.patch
git -C srcs/vendor/metamon apply ../../patches/metamon-r1.patch
```

The patched poke-engine source is tracked at `srcs/vendor/poke-engine`; its
delta from stock 0.0.47 is recorded in
`srcs/patches/poke-engine-0.0.47-priors-v2.patch`.

## Python Environments

Create the Foul Play environment and compile the Gen 9 engine with root-prior
support:

```bash
python3.11 -m venv .venv-fp-priors
.venv-fp-priors/bin/python -m pip install --upgrade pip maturin
.venv-fp-priors/bin/python -m pip install -r srcs/requirements-foul-play.txt
CARGO_TARGET_DIR=/tmp/metagross-poke-engine \
  .venv-fp-priors/bin/python -m pip install --force-reinstall --no-cache-dir \
  srcs/vendor/poke-engine \
  --config-settings='build-args=--no-default-features --features poke-engine/gen9,poke-engine/terastallization'
```

Create the Metamon environment:

```bash
python3.11 -m venv .venv-metamon
.venv-metamon/bin/python -m pip install --upgrade pip
.venv-metamon/bin/python -m pip install -e srcs/vendor/metamon
```

Metamon downloads base assets on first use. The production cache path is
`srcs/runtime/metamon-cache`.

## Checkpoint

Place the accepted checkpoint here:

```text
srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt
```

Verify it:

```bash
shasum -a 256 srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt
```

Expected digest:

```text
c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93
```

The model and downloaded cache are intentionally ignored by Git.

## Smoke Check

```bash
.venv-metamon/bin/python -m srcs.metagross.launch --help
.venv-fp-priors/bin/python -c \
  'import inspect, poke_engine; print(inspect.signature(poke_engine.monte_carlo_tree_search))'
```

The engine signature must contain `s1_priors`, `s2_priors`, and `c_puct`.
