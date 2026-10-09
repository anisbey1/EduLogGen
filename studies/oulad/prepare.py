"""Convert OULAD into EduLogGen event files (one CSV per module presentation).

OULAD (Kuzilek et al., 2017; CC BY 4.0) records clicks per student, resource,
and **day** (days relative to the module start); it has no time of day, and
the order of rows within a day carries no meaning. This script therefore
builds a representation in which everything is observed:

- one event per active learner-day; its event type is the activity type with
  the most clicks that day and its activity is the most-clicked resource;
- one session per learner and module week (``floor(date / 7)``), so a
  session is the ordered sequence of a learner's active days in that week
  and gaps between events are whole days;
- timestamps are the presentation start (1 February for "B", 1 October for
  "J" presentations) plus the day offset, at 12:00 UTC. Hour of day and
  weekday are therefore not observed and must not be analysed.

Usage:
    python studies/oulad/prepare.py --oulad OULAD --output studies/oulad/data
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

START_MONTH = {"B": 2, "J": 10}


def presentation_start(code: str) -> datetime:
    """1 February (B) or 1 October (J) of the presentation year, 12:00 UTC."""
    return datetime(int(code[:4]), START_MONTH[code[4]], 1, 12, tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oulad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--presentations",
        nargs="*",
        default=None,
        help="e.g. AAA-2014J (default: all)",
    )
    args = parser.parse_args()
    wanted = set(args.presentations) if args.presentations else None

    site_type: dict[str, str] = {}
    with (args.oulad / "vle.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            site_type[row["id_site"]] = row["activity_type"]

    # (presentation, student, day) -> clicks per activity type and per site
    by_type: defaultdict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    by_site: defaultdict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    with (args.oulad / "studentVle.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            pres = f"{row['code_module']}-{row['code_presentation']}"
            if wanted is not None and pres not in wanted:
                continue
            key = (pres, row["id_student"], int(row["date"]))
            clicks = int(row["sum_click"])
            by_type[key][site_type.get(row["id_site"], "unknown")] += clicks
            by_site[key][row["id_site"]] += clicks

    args.output.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[list[object]]] = defaultdict(list)
    for key in sorted(by_type, key=lambda k: (k[0], int(k[1]), k[2])):
        pres, student, day = key
        # ties broken alphabetically so the conversion is deterministic
        event_type = min(by_type[key].items(), key=lambda kv: (-kv[1], kv[0]))[0]
        site = min(by_site[key].items(), key=lambda kv: (-kv[1], kv[0]))[0]
        start = presentation_start(pres.split("-")[1])
        timestamp = start + timedelta(days=day)
        week = math.floor(day / 7)
        files[pres].append(
            [
                f"{student}-{day}",
                student,
                timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                site,
                event_type,
                f"{student}-w{week}",
                pres,
                sum(by_type[key].values()),
            ]
        )
    header = [
        "event_id",
        "student",
        "time",
        "site",
        "activity_type",
        "session",
        "presentation",
        "clicks",
    ]
    for pres, rows in sorted(files.items()):
        with (args.output / f"{pres}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        learners = len({r[1] for r in rows})
        print(f"{pres}: {len(rows)} learner-days, {learners} learners")


if __name__ == "__main__":
    main()
