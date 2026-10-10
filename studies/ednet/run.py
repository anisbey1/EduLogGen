"""Fine-grained EdNet-KT4 study for the EduLogGen software paper.

Run ``prepare.py`` first, then:

    python studies/ednet/run.py --data studies/ednet/data/ednet-kt4-2000.csv \
        --output studies/ednet/out

Sessions are built with a 30-minute idle timeout. The analyses are those of
``studies/study.py`` (EdNet has no final results, so profiles are not related
to outcomes). Results go to ``<output>/results.json``.
"""

from __future__ import annotations

import argparse
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

SEED = 0
IDLE_TIMEOUT_S = 1800.0
REPETITION_GAP_S = 5.0


def main() -> None:
    """Run the study on the prepared EdNet sample."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mapping", type=Path, default=Path(__file__).with_name("mapping.yaml")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = sessionize(
        elg.ingest(args.data, args.mapping).dataset,
        strategy="idle_timeout",
        idle_timeout_s=IDLE_TIMEOUT_S,
    )
    entry: dict[str, Any] = {
        "n_events": data.n_events,
        "n_sessions": data.n_sessions,
        "n_learners": len(data.learner_ids),
        "load_s": time.perf_counter() - started,
    }
    print(f"{data.n_events} events, {data.n_sessions} sessions", flush=True)
    steps = {
        "fidelity": lambda: study.fidelity(data, SEED),
        "profiles": lambda: study.profiles(data, SEED),
        "memorisation": lambda: study.memorisation(data, SEED, study.MEMORISATION),
        "detection": lambda: study.detection(data, SEED, REPETITION_GAP_S),
    }
    for step, run in steps.items():
        entry[step], entry[f"{step}_s"] = study.timed(run)
        print(f"   {step}: {entry[f'{step}_s']:.1f}s", flush=True)
    results = {
        "environment": {
            "eduloggen": elg.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "os": platform.system(),
        },
        "seed": SEED,
        "idle_timeout_s": IDLE_TIMEOUT_S,
        "presentations": {args.data.stem: entry},
    }
    (args.output / "results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"wrote {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
