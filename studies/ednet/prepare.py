"""Convert a random subset of EdNet-KT4 into an EduLogGen event file.

EdNet (Choi et al., 2020; CC BY-NC 4.0) records every action of students of
the Santa TOEIC app with a millisecond timestamp: entering and leaving items,
choosing and erasing answers, submitting, playing and pausing audio and
video. KT4 has one CSV per student (``timestamp, action_type, item_id,
cursor_time, source, user_answer, platform``). This script reads the KT4
archive directly, draws a seeded random sample of students, and writes one
CSV with:

- event type = action plus the kind of item it concerns, e.g.
  ``respond:question``, ``play_audio:bundle``, ``enter:lecture``;
- activity = item id; metadata = platform and source.

Sessions are not given; the study builds them with a 30-minute idle timeout.

Usage:
    python studies/ednet/prepare.py --kt4 KT4 \
        --output studies/ednet/data --students 2000 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import zipfile
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

KINDS = {
    "b": "bundle",
    "q": "question",
    "e": "explanation",
    "l": "lecture",
}


def event_type(action: str, item: str) -> str:
    """Action plus item kind; purchase actions keep their own name."""
    kind = KINDS.get(item[:1], "")
    return f"{action}:{kind}" if kind else action


def main() -> None:
    """Sample students from the KT4 archive and write one event file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--kt4", type=Path, required=True, help="KT4 folder or zip archive"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--students", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"ednet-kt4-{args.students}.csv"
    with ExitStack() as stack:
        if args.kt4.is_dir():
            files = {p.name: p for p in args.kt4.glob("u*.csv")}
            members = sorted(files)

            def open_member(name: str) -> IO[str]:
                return files[name].open(encoding="utf-8", newline="")

        else:
            archive = stack.enter_context(zipfile.ZipFile(args.kt4))
            names = [n for n in archive.namelist() if Path(n).name.startswith("u")]
            by_name = {Path(n).name: n for n in names if n.endswith(".csv")}
            members = sorted(by_name)

            def open_member(name: str) -> IO[str]:
                raw = archive.open(by_name[name])
                return io.TextIOWrapper(raw, encoding="utf-8")

        chosen = sorted(random.Random(args.seed).sample(members, args.students))
        n_events = 0
        with target.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["event_id", "student", "time", "item", "action", "platform", "source"]
            )
            for member in chosen:
                student = Path(member).stem
                with open_member(member) as text:
                    for i, row in enumerate(csv.DictReader(text)):
                        stamp = datetime.fromtimestamp(
                            int(row["timestamp"]) / 1000, UTC
                        )
                        writer.writerow(
                            [
                                f"{student}-{i}",
                                student,
                                stamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                row["item_id"],
                                event_type(row["action_type"], row["item_id"]),
                                row.get("platform", ""),
                                row.get("source", ""),
                            ]
                        )
                        n_events += 1
    print(f"{target}: {n_events} events from {len(chosen)} of {len(members)} students")


if __name__ == "__main__":
    main()
