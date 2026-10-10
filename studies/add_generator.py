"""Add one generator to existing study results without recomputing the rest.

    python studies/add_generator.py --study oulad --generator gru \
        --data studies/oulad/data --results studies/oulad/results/results.json
    python studies/add_generator.py --study ednet --generator gru \
        --data studies/ednet/data/ednet-kt4-2000.csv \
        --results studies/ednet/results/results.json

Runs the fidelity benchmark and the memorisation diagnostics for that
generator only and merges them into ``results.json``. Seeds are derived per
generator, so the result equals what a full run of ``run.py`` would give.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import study  # noqa: E402

SEED = 0


def main() -> None:
    """Merge one generator's results into a study's results file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", choices=("oulad", "ednet"), required=True)
    parser.add_argument("--generator", required=True, choices=sorted(study.GENERATORS))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    if args.study == "oulad":
        sys.path.insert(0, str(Path(__file__).resolve().parent / "oulad"))
        from run import load  # type: ignore[import-not-found]

        datasets = {
            name: (
                lambda n=name: load(
                    args.data / f"{n}.csv",
                    Path(__file__).parent / "oulad" / "mapping.yaml",
                )
            )
            for name in results["presentations"]
        }
    else:
        import eduloggen as elg
        from eduloggen.analysis import sessionize

        def load_ednet() -> object:
            return sessionize(
                elg.ingest(
                    args.data, Path(__file__).parent / "ednet" / "mapping.yaml"
                ).dataset,
                strategy="idle_timeout",
                idle_timeout_s=results.get("idle_timeout_s", 1800.0),
            )

        datasets = {name: load_ednet for name in results["presentations"]}
    for name, load_data in datasets.items():
        data = load_data()
        entry = results["presentations"][name]
        fid, fid_s = study.timed(
            lambda d=data: study.fidelity(d, SEED, (args.generator,))
        )
        entry["fidelity"]["generators"][args.generator] = fid["generators"][
            args.generator
        ]
        mem, mem_s = study.timed(
            lambda d=data: study.memorisation(d, SEED, (args.generator,))
        )
        entry["memorisation"][args.generator] = mem[args.generator]
        entry.setdefault("timing_added", {})[args.generator] = {
            "fidelity": fid_s,
            "memorisation": mem_s,
        }
        print(f"{name}: fidelity {fid_s:.0f}s, memorisation {mem_s:.0f}s", flush=True)
        args.results.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
