"""Search-depth calibration: iterations delivered by a 500 ms wall-clock
search on this platform. Compare against the same probe on the M4."""
import json
import modal

app = modal.App("metagross-calibration")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "git")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "pip install maturin==1.7.4",
    )
    .add_local_dir("/Users/alityb/projects/metagross/srcs/vendor/poke-engine",
                   "/build/poke-engine", copy=True,
                   ignore=["**/target/**", "**/.git/**", "**/__pycache__/**",
                           "**/dist/**"])
    .run_commands(
        ". $HOME/.cargo/env && cd /build/poke-engine/poke-engine-py && "
        "maturin build --release "
        "--features poke-engine/terastallization --no-default-features "
        "-o /dist && pip install /dist/*.whl",
    )
)


@app.function(image=image, timeout=600, cpu=8)
def probe() -> dict:
    import platform
    import statistics
    import poke_engine as e
    first = e.Pokemon(id="pikachu", level=80,
                      moves=[e.Move(id="thunderbolt"), e.Move(id="voltswitch")],
                      tera_type="electric")
    reserve = e.Pokemon(id="eevee", level=80, moves=[e.Move(id="tackle")],
                        tera_type="normal")
    state = e.State(side_one=e.Side(pokemon=[first, reserve]),
                    side_two=e.Side(pokemon=[first, reserve]))
    runs = []
    for _ in range(8):
        r = e.monte_carlo_tree_search(state, duration_ms=500, iterations=0,
                                      threads=1)
        runs.append(int(r.total_visits))
    return {"platform": platform.platform(),
            "iterations_per_500ms": {"median": statistics.median(runs),
                                     "min": min(runs), "max": max(runs),
                                     "all": runs}}


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=1))
