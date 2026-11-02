"""Auto review & verify the implementation plans in ``docs/``.

Reviews every ``docs/*-plan.md`` for required structure and verifies internal +
cross-document consistency:

- ``docs/*.md`` referenced by any plan, the roadmap, or TODO must exist;
- every plan file must be linked from ``docs/roadmap.md`` or ``TODO``;
- **new-format plans** (those with ``### WS-N`` workstreams *and* a
  ``## Traceability`` table) are additionally reviewed for:
  - required sections: Current state, Workstreams, Sequencing & dependencies,
    Traceability, Tests;
  - sequential, gap-free workstream numbering (``### WS-N — ...``);
  - every ``WS-N`` in the Traceability table mapping to a real workstream
    (duplicate mappings are warnings — grouped workstreams are allowed);
  - every non-zero workstream appearing in the Traceability table;
  - every ``WS-N`` in the Tests section mapping to a real workstream.

Usage:
    python scripts/check_plans.py [--root PATH]

Exit code 0 when clean, 1 when problems are found. Also importable from tests.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

PLAN_GLOB = "docs/*-plan.md"
DOCS_LINK = re.compile(r"docs/[A-Za-z0-9._-]+\.md")
WS_HEADER = re.compile(r"^### WS-(\d+)\s*—\s*(.*)$", re.MULTILINE)
WS_TOKEN = re.compile(r"WS-(\d+)")
H2 = re.compile(r"^##\s+(.*)$")

CHECKLIST_PATH = "docs/checklist.md"
PHASE_RE = re.compile(r"^##\s+(.+?)\s*—\s*plan:\s*(.+)$")
ITEM_RE = re.compile(r"^-\s+\[[ xX]\]\s*(.+)$")
TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+/-]*")
_LATIN = re.compile(r"[A-Za-z0-9_.+/-]+")
# Generic Bengali verb/helper tokens that do not identify an item.
_STOP = {"তৈরি", "করা", "করবে", "হবে", "নির্ধারণ"}

COVERAGE_THRESHOLD = 0.6

REQUIRED_SECTIONS = (
    "Current state",
    "Workstreams",
    "Sequencing & dependencies",
    "Traceability",
    "Tests",
)


@dataclass
class PlanReport:
    path: Path
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    workstreams: dict[int, str] = field(default_factory=dict)
    new_format: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems


def _section_lines(lines: list[str], heading: str) -> list[str]:
    """Return the body lines under an ``## heading`` until the next ``##``."""
    start = -1
    for i, line in enumerate(lines):
        m = H2.match(line)
        if m and m.group(1).startswith(heading):
            start = i + 1
            break
    if start < 0:
        return []
    body: list[str] = []
    for line in lines[start:]:
        if H2.match(line):
            break
        body.append(line)
    return body


def _strip_fence(body: list[str]) -> list[str]:
    """Drop fenced code blocks so table/header parsing is not confused by them."""
    out: list[str] = []
    in_fence = False
    for line in body:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return out


def review_plan(path: Path) -> PlanReport:
    report = PlanReport(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    if "# " not in text[:200]:
        report.problems.append("missing H1 title near the top")

    if "Goal:" not in text[:800]:
        report.problems.append("missing 'Goal:' statement near the top")

    for m in WS_HEADER.finditer(text):
        num = int(m.group(1))
        if num in report.workstreams:
            report.warnings.append(f"duplicate workstream header WS-{num}")
        report.workstreams[num] = m.group(2).strip()

    # New-format plans carry workstreams AND a traceability table; older plans
    # (dataset-pipeline A/B/C phases, foundation waves, web-app sprints) skip
    # the strict WS review.
    report.new_format = bool(report.workstreams) and bool(_section_lines(lines, "Traceability"))

    if not report.new_format:
        return report

    for section in REQUIRED_SECTIONS:
        if not _section_lines(lines, section):
            report.problems.append(f"missing required section '## {section}'")

    nums = sorted(report.workstreams)
    if nums:
        start = nums[0]
        expected = list(range(start, nums[-1] + 1))
        if nums != expected:
            report.problems.append(
                f"workstream numbering is not contiguous (expected {expected}, got {nums})"
            )

    # Traceability: every WS in the table must be real; duplicates are warnings.
    table = _strip_fence(_section_lines(lines, "Traceability"))
    traced: list[int] = []
    for line in table:
        row = [c.strip() for c in line.strip().strip("|").split("|") if c.strip()]
        if not row or row[0].startswith("-") or row[0].lower() == "requested item":
            continue
        m = WS_TOKEN.fullmatch(row[-1])
        if not m:
            report.problems.append(f"traceability row has no valid WS target: {line.strip()}")
            continue
        num = int(m.group(1))
        if num not in report.workstreams:
            report.problems.append(f"traceability references missing workstream WS-{num}")
        if num in traced:
            report.warnings.append(f"traceability maps WS-{num} to multiple items")
        traced.append(num)

    untraced = sorted(n for n in report.workstreams if n not in traced and n != 0)
    if untraced:
        report.problems.append(
            "workstreams missing from traceability: " + ", ".join(f"WS-{n}" for n in untraced)
        )

    # Tests: every WS referenced must be real.
    tests = _section_lines(lines, "Tests")
    for m in WS_TOKEN.finditer("\n".join(tests)):
        num = int(m.group(1))
        if num not in report.workstreams:
            report.problems.append(f"Tests references missing workstream WS-{num}")

    return report


def verify_links(reports: list[PlanReport], root: Path) -> list[str]:
    problems: list[str] = []
    plan_files = sorted(root.glob(PLAN_GLOB))

    linked_in_roadmap_or_todo: set[str] = set()
    for doc in (root / "docs/roadmap.md", root / "TODO"):
        if not doc.exists():
            problems.append(f"missing {doc.relative_to(root)}")
            continue
        for m in DOCS_LINK.finditer(doc.read_text(encoding="utf-8")):
            linked_in_roadmap_or_todo.add(m.group(0))

    # Every plan must be referenced by the roadmap or TODO.
    for plan in plan_files:
        if f"docs/{plan.name}" not in linked_in_roadmap_or_todo:
            problems.append(f"plan not linked from docs/roadmap.md or TODO: docs/{plan.name}")

    # Every docs link inside every plan + roadmap + TODO must resolve.
    for path in [*plan_files, root / "docs/roadmap.md", root / "TODO"]:
        text = path.read_text(encoding="utf-8")
        for m in DOCS_LINK.finditer(text):
            target = root / m.group(0)
            if not target.exists():
                problems.append(f"{path.relative_to(root)} links to missing {m.group(0)}")

    return problems


def _latin_tokens(text: str) -> list[str]:
    """ASCII-only tokens (English/technical terms), lowercased."""
    return [t.lower() for t in _LATIN.findall(text) if t.lower() not in _STOP]


def _item_tokens(text: str) -> list[str]:
    toks: list[str] = []
    for raw in text.replace("—", " ").split():
        toks.extend(t.lower() for t in TOKEN_RE.findall(raw))
    return [t for t in toks if t not in _STOP]


def _norm(text: str) -> str:
    return " ".join(text.replace("—", " ").replace("/", " ").split()).lower()


def _covered(item: str, plan_texts: list[str]) -> bool:
    norm_item = _norm(item)
    for plan_text in plan_texts:
        if norm_item in _norm(plan_text):
            return True

    latin = _latin_tokens(item)
    for plan_text in plan_texts:
        plan_tokens = set(_latin_tokens(plan_text))
        if latin and sum(1 for t in latin if t in plan_tokens) / len(latin) >= COVERAGE_THRESHOLD:
            return True

    all_toks = [t for t in _item_tokens(item) if not _LATIN.fullmatch(t)]
    for plan_text in plan_texts:
        plan_tokens = set(_item_tokens(plan_text))
        if (
            all_toks
            and sum(1 for t in all_toks if t in plan_tokens) / len(all_toks) >= COVERAGE_THRESHOLD
        ):
            return True
    return False


def verify_coverage(root: Path) -> list[str]:
    problems: list[str] = []
    checklist = root / CHECKLIST_PATH
    if not checklist.exists():
        return [f"missing master checklist: {CHECKLIST_PATH}"]

    phases: list[dict] = []
    current: dict | None = None
    for line in checklist.read_text(encoding="utf-8").splitlines():
        m = PHASE_RE.match(line)
        if m:
            current = {
                "title": m.group(1).strip(),
                "plans": [p.strip() for p in m.group(2).split(",")],
                "items": [],
            }
            phases.append(current)
            continue
        im = ITEM_RE.match(line)
        if im and current is not None:
            current["items"].append(im.group(1).strip())

    for phase in phases:
        plan_texts: list[str] = []
        missing_plans: list[str] = []
        for plan in phase["plans"]:
            path = root / plan
            if path.exists():
                plan_texts.append(path.read_text(encoding="utf-8"))
            else:
                missing_plans.append(plan)
        if missing_plans:
            problems.append(
                f"checklist phase '{phase['title']}' maps to missing plan(s): {missing_plans}"
            )
            continue

        uncovered = [item for item in phase["items"] if not _covered(item, plan_texts)]
        covered = len(phase["items"]) - len(uncovered)
        print(f"  {phase['title']}: {covered}/{len(phase['items'])} items covered")
        for item in uncovered:
            problems.append(f"checklist item not covered by {', '.join(phase['plans'])}: {item}")

    return problems


def run(root: Path) -> list[str]:
    problems: list[str] = []
    reports: list[PlanReport] = []
    for path in sorted(root.glob(PLAN_GLOB)):
        report = review_plan(path)
        reports.append(report)
        flag = "OK" if report.ok else f"{len(report.problems)} problem(s)"
        fmt = "new-format" if report.new_format else "legacy"
        print(f"{path.relative_to(root)} [{fmt}]: {flag}")
        for problem in report.problems:
            print(f"  - {problem}")
            problems.append(f"{path.relative_to(root)}: {problem}")
        for warning in report.warnings:
            print(f"  ! {warning}")

    for problem in verify_links(reports, root):
        print(f"  [link] {problem}")
        problems.append(problem)

    print("\nCoverage (master checklist → plans):")
    for problem in verify_coverage(root):
        print(f"  [coverage] {problem}")
        problems.append(problem)

    total = len(reports)
    bad = sum(1 for r in reports if not r.ok)
    warns = sum(len(r.warnings) for r in reports)
    print(f"\n{total} plans reviewed, {bad} with problems, {warns} warnings")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repo root (default: current dir)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    problems = run(root)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
