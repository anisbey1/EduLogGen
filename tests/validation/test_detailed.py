"""Tests for the fine-grained real-versus-synthetic comparison and its CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import eduloggen as elg
from eduloggen.analysis import sessionize
from eduloggen.cli import main
from eduloggen.core import AnalysisError
from eduloggen.datasets import demo_dataset
from eduloggen.generators import get_generator
from eduloggen.models import Dataset
from eduloggen.validation import compare_detailed


@pytest.fixture(scope="module")
def real() -> Dataset:
    return sessionize(demo_dataset(50, seed=3))


def test_identical_datasets_have_no_gaps(real: Dataset) -> None:
    c = compare_detailed(real, real)
    assert all(t["difference"] == 0 for t in c.tokens)
    assert all(t["difference"] == 0 for t in c.transitions)
    assert c.novel_transitions["count"] == 0
    assert c.missing_transitions["count"] == 0
    assert all(r["difference"] == 0 for r in c.hours)
    assert c.groups == ()
    assert c.tokens[0]["real_dwell_median_s"] == c.tokens[0]["synthetic_dwell_median_s"]


def test_independent_baseline_shows_novel_transitions(real: Dataset) -> None:
    generator = get_generator("independent")
    synthetic = generator.generate(generator.fit(real), 200, seed=0)
    c = compare_detailed(real, synthetic, top=5)
    assert c.novel_transitions["count"] > 0
    assert c.novel_transitions["synthetic_share"] > 0.2
    assert len(c.transitions) == 5
    gaps = [abs(t["difference"]) for t in c.transitions]
    assert gaps == sorted(gaps, reverse=True)
    assert len(c.novel_transitions["examples"]) <= 5


def test_sorted_tokens_and_lengths(real: Dataset) -> None:
    generator = get_generator("markov")
    synthetic = generator.generate(generator.fit(real), 200, seed=1)
    c = compare_detailed(real, synthetic)
    gaps = [abs(t["difference"]) for t in c.tokens]
    assert gaps == sorted(gaps, reverse=True)
    assert sum(r["real_share"] for r in c.session_lengths) == pytest.approx(1.0)
    assert sum(r["synthetic_share"] for r in c.weekdays) == pytest.approx(1.0)
    assert len(c.hours) == 24


def test_long_sessions_are_bucketed() -> None:
    from ..generators.conftest import build

    long = build([tuple("ab" * 20)])
    c = compare_detailed(long, long)
    assert [r["length"] for r in c.session_lengths] == ["30+"]


def test_by_group(real: Dataset) -> None:
    generator = get_generator("semi_markov")
    synthetic = generator.generate(generator.fit(real), 200, seed=2)
    c = compare_detailed(real, synthetic, by="weekday", timezone="Europe/Paris")
    assert c.by == "weekday"
    assert {g["group"] for g in c.groups} >= {"1-Mon"}
    both = [g for g in c.groups if g["real_sessions"] and g["synthetic_sessions"]]
    assert all(
        set(g["metrics"])
        == {"event_type_tvd", "bigram_tvd", "session_length_ks", "interevent_time_ks"}
        for g in both
    )
    text = c.to_markdown(top=3)
    assert "## By weekday" in text
    assert "## Session start hours (Europe/Paris), largest gaps" in text
    assert json.loads(json.dumps(c.to_dict()))["timezone"] == "Europe/Paris"
    with pytest.raises(AnalysisError):
        compare_detailed(real, synthetic, by="planet")


def test_groups_only_in_one_dataset(real: Dataset) -> None:
    from ..generators.conftest import build

    other = build([("view", "attempt")] * 3)
    c = compare_detailed(real, other, by="course")
    missing = [
        g for g in c.groups if not g["real_sessions"] or not g["synthetic_sessions"]
    ]
    assert missing
    assert all(g["metrics"] == {} for g in missing)
    assert "| - |" in c.to_markdown()


def test_top_level_api(real: Dataset) -> None:
    assert elg.activity_profiles(real)
    assert elg.temporal_profile(real).weekly
    assert elg.analyze_by(real, "course").groups
    assert elg.compare_detailed(real, real).novel_transitions["count"] == 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@pytest.fixture
def corpora(tmp_path: Path) -> Path:
    for command in (
        "demo --output {d}/demo --learners 30",
        "fit --input {d}/demo --output {d}/m",
        "generate --model {d}/m --n-sessions 80 --seed 1 --output {d}/syn",
    ):
        assert main([*command.format(d=tmp_path).split(), "--quiet"]) == 0
    (tmp_path / "groups.csv").write_text(
        "learner_id,group\n"
        + "".join(
            f"learner-{i:03d},{'early' if i < 15 else 'late'}\n" for i in range(1, 31)
        )
    )
    return tmp_path


def test_cli_analyze_detail_and_by(
    corpora: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command = (
        "analyze --input {d}/demo --detail --by learner_group --groups {d}/groups.csv "
        "--timezone Europe/Paris --deadlines 2026-02-05 --output {d}/a"
    )
    assert main([*command.format(d=corpora).split(), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "activities.json" in out and "strata.md" in out
    files = {p.name for p in (corpora / "a").iterdir()}
    assert {"activities.md", "temporal.json", "strata.json", "analysis.json"} <= files
    temporal = json.loads((corpora / "a" / "temporal.json").read_text())
    assert temporal["timezone"] == "Europe/Paris"
    assert temporal["deadlines"][0]["deadline"] == "2026-02-05"
    strata = json.loads((corpora / "a" / "strata.json").read_text())
    assert {g["group"] for g in strata["groups"]} <= {"early", "late"}

    assert (
        main(["analyze", "--input", str(corpora / "demo"), "--detail", "--quiet"]) == 0
    )
    printed = capsys.readouterr().out
    assert "## Activities" in printed and "## Activity over time (UTC)" in printed


def test_cli_validate_detailed(
    corpora: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command = (
        "validate --real {d}/demo --synthetic {d}/syn --detailed --by weekday "
        "--output {d}/v"
    )
    assert main([*command.format(d=corpora).split(), "--quiet"]) == 0
    assert "# Where synthetic differs from real" in capsys.readouterr().out
    detailed = json.loads((corpora / "v" / "detailed.json").read_text())
    assert detailed["by"] == "weekday"


def test_cli_errors(corpora: Path, capsys: pytest.CaptureFixture[str]) -> None:
    by_without_detail = "validate --real {d}/demo --synthetic {d}/syn --by weekday"
    assert main([*by_without_detail.format(d=corpora).split(), "--quiet"]) == 2
    assert "--by needs --detailed" in capsys.readouterr().err
    (corpora / "bad.csv").write_text("id,cohort\n1,x\n")
    bad = "analyze --input {d}/demo --by learner_group --groups {d}/bad.csv"
    assert main([*bad.format(d=corpora).split(), "--quiet"]) == 2
    assert "learner_id" in capsys.readouterr().err
    missing = "analyze --input {d}/demo --by learner_group --groups {d}/none.csv"
    assert main([*missing.format(d=corpora).split(), "--quiet"]) == 2
    no_groups = "analyze --input {d}/demo --by learner_group"
    assert main([*no_groups.format(d=corpora).split(), "--quiet"]) == 2
    capsys.readouterr()


def test_activity_heatmap_plot(real: Dataset) -> None:
    pytest.importorskip("matplotlib")
    from eduloggen.visualization import (
        DATASET_PLOTS,
        plot_activity_heatmap,
        require_matplotlib,
    )

    figure = plot_activity_heatmap(real, real, timezone="Europe/Paris")
    assert [ax.get_title() for ax in figure.axes[:2]] == [
        "Session starts (real)",
        "Session starts (synthetic)",
    ]
    require_matplotlib().close(figure)
    assert "activity_heatmap" in DATASET_PLOTS
    require_matplotlib().close(DATASET_PLOTS["activity_heatmap"](real, None))
