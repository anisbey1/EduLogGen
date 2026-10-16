"""v1.6.0 studies: six-generator comparison and Transformer sensitivity.

    python studies/run_v16.py --part main         # GRU (matched) and Transformer
    python studies/run_v16.py --part costs        # cost of every generator
    python studies/run_v16.py --part context      # EdNet context length
    python studies/run_v16.py --part fraction     # training-data size

``main`` adds the fidelity benchmark (all metrics, ``session_fidelity_v1``)
and the memorisation diagnostics of ``gru_matched`` and ``transformer`` to
the published OULAD and EdNet results (other generators are unchanged; seeds
are derived per generator). ``costs`` records fit time, generation time,
peak memory, parameter count, and model size of every generator.
``context`` varies the Transformer's ``max_seq_len`` on EdNet (OULAD sessions
have at most seven events, so context length cannot matter there).
``fraction`` fits on nested 10/25/50/100% subsets of the training learners.
Sensitivity runs use the fidelity metrics without the slow privacy
diagnostics (definitions unchanged). Every task runs in its own process
(``worker.py``); results are saved after each task, and finished tasks are
skipped when the script is run again.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OULAD_RESULTS = HERE / "oulad" / "results" / "results.json"
EDNET_RESULTS = HERE / "ednet" / "results" / "results.json"
SENSITIVITY = HERE / "sensitivity" / "results.json"
OULAD_DATA = HERE / "oulad" / "data"
EDNET_DATA = HERE / "ednet" / "data" / "ednet-kt4-2000.csv"
ALL = (
    "independent",
    "markov",
    "markov2",
    "semi_markov",
    "gru",
    "gru_matched",
    "transformer",
)
NEW = ("gru_matched", "transformer")
CONTEXTS = (32, 64, 128, 256)
FRACTIONS = (0.1, 0.25, 0.5, 1.0)
FRACTION_GENERATORS = ("markov", "semi_markov", "gru_matched", "transformer")


def worker(**kwargs: Any) -> dict[str, Any]:
    """Run one worker task and return its JSON result."""
    command = [sys.executable, str(HERE / "worker.py")]
    for key, value in kwargs.items():
        if value is not None:
            command += [f"--{key.replace('_', '-')}", str(value)]
    started = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"{' '.join(command)} failed:\n{done.stderr[-3000:]}")
    result: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    result["wall_s"] = time.perf_counter() - started
    print(f"  {kwargs} -> {result['wall_s']:.0f}s", flush=True)
    return result


def datasets(results_oulad: dict[str, Any]) -> list[tuple[str, Path, str | None, Path]]:
    out = [
        ("oulad", OULAD_DATA, p, OULAD_RESULTS) for p in results_oulad["presentations"]
    ]
    out.append(("ednet", EDNET_DATA, None, EDNET_RESULTS))
    return out


def save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def main() -> None:
    """Run the requested part of the v1.6.0 studies."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--part", choices=("main", "costs", "context", "fraction"), required=True
    )
    args = parser.parse_args()
    results = {
        "oulad": json.loads(OULAD_RESULTS.read_text()),
        "ednet": json.loads(EDNET_RESULTS.read_text()),
    }
    sensitivity = (
        json.loads(SENSITIVITY.read_text())
        if SENSITIVITY.exists()
        else {"context": {}, "fraction": {}}
    )

    if args.part in ("main", "costs"):
        for name, data, presentation, path in datasets(results["oulad"]):
            store = results[name]
            key = presentation or next(iter(store["presentations"]))
            entry = store["presentations"][key]
            for label in NEW if args.part == "main" else ALL:
                if args.part == "main":
                    if label not in entry["fidelity"]["generators"]:
                        fid = worker(
                            study=name,
                            data=data,
                            presentation=presentation,
                            task="fidelity",
                            label=label,
                        )
                        entry["fidelity"]["generators"][label] = fid
                        save(path, store)
                    if label not in entry["memorisation"]:
                        mem = worker(
                            study=name,
                            data=data,
                            presentation=presentation,
                            task="memorisation",
                            label=label,
                        )
                        entry["memorisation"][label] = mem
                        save(path, store)
                else:
                    costs = entry.setdefault("costs", {})
                    if label in costs:
                        continue
                    mem = entry["memorisation"].get(label, {})
                    if "cost" in mem:  # measured with the memorisation run
                        costs[label] = {
                            **mem["cost"],
                            "peak_rss_mb": mem["peak_rss_mb"],
                        }
                    else:
                        run = worker(
                            study=name,
                            data=data,
                            presentation=presentation,
                            task="memorisation",
                            label=label,
                        )
                        costs[label] = {
                            **run["cost"],
                            "peak_rss_mb": run["peak_rss_mb"],
                        }
                    save(path, store)

    if args.part == "context":
        for length in CONTEXTS:
            key = str(length)
            if key not in sensitivity["context"]:
                sensitivity["context"][key] = worker(
                    study="ednet",
                    data=EDNET_DATA,
                    task="context",
                    label="transformer",
                    max_seq_len=length,
                )
                save(SENSITIVITY, sensitivity)

    if args.part == "fraction":
        for name, data, presentation, _ in datasets(results["oulad"]):
            key = presentation or "ednet-kt4-2000"
            block = sensitivity["fraction"].setdefault(key, {})
            for fraction in FRACTIONS:
                for label in FRACTION_GENERATORS:
                    run_key = f"{label}@{fraction}"
                    if run_key not in block:
                        block[run_key] = worker(
                            study=name,
                            data=data,
                            presentation=presentation,
                            task="fraction",
                            label=label,
                            fraction=fraction,
                        )
                        save(SENSITIVITY, sensitivity)


if __name__ == "__main__":
    main()
