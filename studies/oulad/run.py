"""Reproducible OULAD study for the EduLogGen software paper.

Run ``prepare.py`` first, then:

    python studies/oulad/run.py --data studies/oulad/data \
        --oulad OULAD --output studies/oulad/out

For each module presentation (seed 0) this runs the analyses in
``studies/study.py``: the fidelity benchmark, behavioural profiles with
stability, outcome association and recovery, memorisation diagnostics, and
anomaly detection on real and synthetic backgrounds. Results go to
``<output>/results.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eduloggen as elg  # noqa: E402
import study  # noqa: E402
from eduloggen.analysis import sessionize  # noqa: E402
from eduloggen.models import Dataset  # noqa: E402

SEED = 0
DAY_S = 86_400.0


def load(path: Path, mapping: Path) -> Dataset:
    """Ingest a prepared presentation file; sessions are learner-weeks."""
    return sessionize(elg.ingest(path, mapping).dataset, strategy="explicit")


def final_results(oulad: Path, presentation: str) -> dict[str, str]:
    """Final result (Pass, Fail, Withdrawn, Distinction) per student."""
    module, code = presentation.split("-")
    with (oulad / "studentInfo.csv").open(newline="") as handle:
        return {
            row["id_student"]: row["final_result"]
            for row in csv.DictReader(handle)
            if row["code_module"] == module and row["code_presentation"] == code
        }


def main() -> None:
    """Run the study on every prepared presentation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--oulad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mapping", type=Path, default=Path(__file__).with_name("mapping.yaml")
    )
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "environment": {
            "eduloggen": elg.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "os": platform.system(),
        },
        "seed": SEED,
        "presentations": {},
    }
    for path in sorted(args.data.glob("*.csv")):
        name = path.stem
        if args.only and name not in args.only:
            continue
        print(f"== {name}", flush=True)
        started = time.perf_counter()
        data = load(path, args.mapping)
        entry: dict[str, Any] = {
            "n_events": data.n_events,
            "n_sessions": data.n_sessions,
            "n_learners": len(data.learner_ids),
            "load_s": time.perf_counter() - started,
        }
        outcomes = final_results(args.oulad, name)
        steps = {
            "fidelity": lambda d=data: study.fidelity(d, SEED),
            "profiles": lambda d=data, o=outcomes: study.profiles(d, SEED, o),
            "memorisation": lambda d=data: study.memorisation(
                d, SEED, study.MEMORISATION
            ),
            "detection": lambda d=data: study.detection(d, SEED, DAY_S),
        }
        for step, run in steps.items():
            entry[step], entry[f"{step}_s"] = study.timed(run)
            print(f"   {step}: {entry[f'{step}_s']:.1f}s", flush=True)
        results["presentations"][name] = entry
        (args.output / "results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
    print(f"wrote {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
