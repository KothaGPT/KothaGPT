import sys
from pathlib import Path

from scripts.check_plans import (
    _covered,
    review_plan,
    run,
    verify_coverage,
    verify_links,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_plans_review_clean():
    problems = run(ROOT)
    assert problems == []


def test_traceability_must_reference_real_workstreams(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    bad = docs / "bad-plan.md"
    bad.write_text(
        "# Bad Plan\n\nGoal: test.\n\n"
        "## Current state\n\n| Area | Exists | Gaps |\n| --- | --- | --- |\n| x | y | z |\n\n"
        "## Workstreams\n\n### WS-1 — One\n\n"
        "## Sequencing & dependencies\n\nWS-1\n\n"
        "## Traceability (requested items → workstreams)\n\n"
        "| Requested item | Workstream |\n| --- | --- |\n| Thing | WS-9 |\n\n"
        "## Tests\n\n```bash\npytest # WS-9\n```\n",
        encoding="utf-8",
    )
    report = review_plan(bad)
    assert "traceability references missing workstream WS-9" in report.problems
    assert "Tests references missing workstream WS-9" in report.problems


def test_workstream_numbering_gaps_detected(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    bad = docs / "gap-plan.md"
    bad.write_text(
        "# Gap Plan\n\nGoal: test.\n\n"
        "## Current state\n\n| Area | Exists | Gaps |\n| --- | --- | --- |\n| x | y | z |\n\n"
        "## Workstreams\n\n### WS-1 — One\n\n### WS-3 — Three\n\n"
        "## Sequencing & dependencies\n\nWS-1, WS-3\n\n"
        "## Traceability (requested items → workstreams)\n\n"
        "| Requested item | Workstream |\n| --- | --- |\n| One | WS-1 |\n| Three | WS-3 |\n\n"
        "## Tests\n\n```bash\npytest # WS-1 WS-3\n```\n",
        encoding="utf-8",
    )
    report = review_plan(bad)
    assert any("not contiguous" in p for p in report.problems)


def test_broken_docs_links_detected(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    plan = docs / "a-plan.md"
    plan.write_text("# A Plan\n\nGoal: test.\n\nSee docs/nope.md\n", encoding="utf-8")
    (tmp_path / "TODO").write_text("TODO\n", encoding="utf-8")
    roadmap = docs / "roadmap.md"
    roadmap.write_text("# Roadmap\n\n- [ ] A (`docs/a-plan.md`)\n", encoding="utf-8")
    problems = verify_links([review_plan(plan)], tmp_path)
    assert any("missing docs/nope.md" in p for p in problems)


def test_plan_must_be_linked_from_roadmap_or_todo(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    plan = docs / "orphan-plan.md"
    plan.write_text("# Orphan\n\nGoal: test.\n", encoding="utf-8")
    (docs / "roadmap.md").write_text("# Roadmap\n", encoding="utf-8")
    (tmp_path / "TODO").write_text("TODO\n", encoding="utf-8")
    problems = verify_links([review_plan(plan)], tmp_path)
    assert any("not linked" in p for p in problems)


def test_checker_runnable_as_module():
    import subprocess

    result = subprocess.run(
        [sys.executable, "scripts/check_plans.py", "--root", str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "0 with problems" in result.stdout


def test_checklist_full_coverage():
    problems = verify_coverage(ROOT)
    assert problems == []


def test_coverage_flags_uncovered_item(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x-plan.md").write_text(
        "# X Plan\n\nGoal: test.\n\n## Workstreams\n\n### WS-1 — One\n",
        encoding="utf-8",
    )
    (docs / "checklist.md").write_text(
        "# Checklist\n\n## Phase One — plan: docs/x-plan.md\n\n- [ ] Alpha capability\n"
        "- [ ] Unrelated never-mentioned thing\n",
        encoding="utf-8",
    )
    problems = verify_coverage(tmp_path)
    assert any("Unrelated never-mentioned thing" in p for p in problems)


def test_covered_matches_latin_terms_in_legacy_headers():
    plan_text = "# Bangla Plan\n\n### WS-1 — Bangla tokenizer: create / select\n"
    assert _covered("বাংলা tokenizer তৈরি/নির্বাচন", [plan_text])
    assert not _covered("Completely unrelated safety thing", [plan_text])
