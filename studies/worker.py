"""Run one study task for one generator in a fresh process (v1.6.0 studies).

Each task runs in its own process so that peak memory (resident set size)
belongs to that generator alone. Prints one JSON object on the last line.

    python studies/worker.py --study ednet --data studies/ednet/data/ednet-kt4-2000.csv \
        --task memorisation --label transformer
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import study  # noqa: E402

SEED = 0
HERE = Path(__file__).resolve().parent


def load(name: str, data: Path, presentation: str | None) -> Any:
    """Load and sessionize a study dataset as the published studies do."""
    if name == "oulad":
        sys.path.insert(0, str(HERE / "oulad"))
        from run import load as load_oulad  # type: ignore[import-not-found]

        return load_oulad(data / f"{presentation}.csv", HERE / "oulad" / "mapping.yaml")
    import eduloggen as elg
    from eduloggen.analysis import sessionize

    return sessionize(
        elg.ingest(data, HERE / "ednet" / "mapping.yaml").dataset,
        strategy="idle_timeout",
        idle_timeout_s=1800.0,
    )


def peak_rss_mb() -> float:
    """Peak resident set size of this process (macOS: bytes, Linux: KiB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if platform.system() == "Darwin" else peak / 2**10


def main() -> None:
    """Run the requested task and print its JSON result."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", choices=("oulad", "ednet"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--presentation")
    parser.add_argument(
        "--task",
        choices=("fidelity", "memorisation", "context", "fraction"),
        required=True,
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--fraction", type=float)
    args = parser.parse_args()
    data = load(args.study, args.data, args.presentation)
    if args.task == "fidelity":
        result = study.fidelity(data, SEED, (args.label,))["generators"][args.label]
    elif args.task == "memorisation":
        result = study.memorisation(data, SEED, (args.label,))[args.label]
    else:
        train, holdout = study.split(data, SEED)
        generator, params = study.GENERATORS[args.label]
        params = dict(params)
        if args.task == "context":
            params["max_seq_len"] = args.max_seq_len
        else:
            train = study.nested_subset(train, args.fraction, SEED)
        result = study.fast_fidelity(
            train, holdout, generator, params, SEED, args.label
        )
    result["peak_rss_mb"] = peak_rss_mb()
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
