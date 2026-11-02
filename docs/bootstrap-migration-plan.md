# Bootstrap Migration — Implementation Plan

Goal: replace the 7-line Linux-only `scripts/bootstrap.sh` with a single
platform-aware entry point that detects (or accepts) the target platform,
shows exactly what it would run via `print_bootstrap_preview(platform)`
before executing anything, and delegates real work to per-platform modules
under `scripts/linux/`, `scripts/macos/`, `scripts/windows/` — while
`make bootstrap` delegates to the same script so there is one source of
truth for environment setup.

Guiding principles:

- One entry point, three platforms: users never guess whether to run
  `bootstrap.sh`, a Makefile target, or a PowerShell script; the dispatcher
  picks the right module (or is told via `--platform`).
- Preview before mutate: every destructive/setup step is printable without
  being executed (`--dry-run`), so CI and new contributors can audit it.
- Parity with `Makefile`: the current script installs only the Python venv;
  `make bootstrap` also runs `corepack enable` + `pnpm install`. After the
  migration both paths do exactly the same thing because they are the same
  code.
- Idempotent by default: re-running bootstrap on an already-bootstrapped tree
  is a no-op (skip venv creation, skip pip upgrade when current).
- Build on what exists: `scripts/install_rust` already sets the
  platform-agnostic tone (no unix-only paths); platform modules reuse it
  instead of duplicating rustup logic.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Entry script | `scripts/bootstrap.sh` (venv + pip install of `services/api/requirements.txt`) | `source .venv/bin/activate` fails on Windows; no OS detection; no flags |
| Makefile | `make bootstrap` duplicates venv+pip and adds `corepack enable` + `pnpm install` | two drifting definitions of "bootstrapped"; script and target diverge |
| Platform layout | empty `scripts/linux/`, `scripts/macos/`, `scripts/windows/` dirs | no modules inside; no dispatch contract |
| Rust tooling | `scripts/install_rust` (CI-aware, idempotent) | not called by any bootstrap path |
| CI | `.github/workflows/ci.yml` jobs: validate / plans / test | no bootstrap verification; a broken script ships unnoticed |
| Docs | README quick start says `cp .env.example .env && make bootstrap` | no documented per-platform prerequisites (brew/apt/winget) |

---

## Workstreams

### WS-1 — Dispatcher skeleton & shared library (`scripts/bootstrap.sh` + `scripts/lib/`)

Goal: a single `scripts/bootstrap.sh` that resolves the target platform and
loads the right module behind a stable contract.

- Rewrite `scripts/bootstrap.sh` with `set -euo pipefail`, script-dir
  resolution, and argument parsing: `--platform linux|macos|windows`,
  `--dry-run` (alias `--preview`), `--yes` (non-interactive), `--help`.
- Add `detect_platform()`: `uname -s` → `Darwin`→macos, `Linux`→linux;
  MSYS/MinGW/Cygwin or native Windows → windows. `--platform` overrides
  detection (required for CI previews of foreign platforms).
- Add `scripts/lib/bootstrap_common.sh`: color/log helpers (`info`, `warn`,
  `fail`), `run_or_print()` (executes unless dry-run), and idempotency guards
  (`venv_python_exists()`, aware of `.venv/bin/python` vs
  `.venv/Scripts/python.exe`).
- Module contract: each platform module defines `bootstrap_steps()` that
  emits one shell command per line; the dispatcher owns execution/logging so
  modules stay dumb data.
- Deliverables: rewritten `scripts/bootstrap.sh`, `scripts/lib/bootstrap_common.sh`.
- Metric: `bash -n` clean; wrong/unknown platform exits non-zero with usage;
  detection correct on Linux, macOS, and Git-Bash/PowerShell.

### WS-2 — Preview mode (`print_bootstrap_preview(platform)`)

Goal: show, don't do — a first-class dry-run that prints the exact per-platform
plan.

- Implement `print_bootstrap_preview() { local platform="$1"; ... }` in the
  dispatcher: header (`== Kotha GPT bootstrap preview — <platform> ==`),
  numbered steps sourced from the module's `bootstrap_steps()`, and a trailing
  hint (`re-run with --yes to execute`). Pure output: no filesystem writes,
  no network calls, safe to run anywhere.
- `--dry-run` routes every `run_or_print()` call through the printer, so the
  preview and the execution path can never diverge.
- Exit code 0 on preview; unknown platform still exits 1.
- Deliverables: preview function + flag wiring.
- Metric: `--dry-run` output contains every command the real run executes,
  byte-for-byte, verified by the WS-5 test; dry-run leaves the working tree
  untouched (no `.venv` created).

### WS-3 — Platform modules (`scripts/linux/`, `scripts/macos/`, `scripts/windows/`)

Goal: fill the empty platform dirs with the actual setup steps per OS.

- `scripts/linux/bootstrap.sh`: apt/dnf/pacman hint block for build deps,
  `python3 -m venv .venv`, POSIX activate path, `pip install -r
  services/api/requirements.txt`, optional `scripts/install_rust` reuse.
- `scripts/macos/bootstrap.sh`: Homebrew presence check + zsh-compatible
  notes, same venv/pip core, `scripts/install_rust` reuse.
- `scripts/windows/bootstrap.ps1` (+ thin `bootstrap.sh` shim for Git-Bash):
  `py -m venv .venv`, `Scripts\Activate.ps1` execution-policy note,
  equivalent pip step; winget hints for Python 3.12+.
- All modules emit steps through the WS-1 contract; no module writes state
  directly.
- Toolchain install: when Python, Go, or Rust (`cargo`) is missing, modules
  emit install steps through the platform package manager — apt/dnf/pacman
  (linux), Homebrew (macos), winget (windows, including `Rustlang.Rustup`);
  Rust on linux/macos reuses `scripts/install_rust`. Present tools are never
  reinstalled (idempotent), and the check phase validates the installer
  instead of the toolchain.
- Deliverables: three modules + a short "per-platform prerequisites" section
  in the README quick start.
- Metric: on a fresh machine each module's emitted steps reach
  `Kotha GPT bootstrap complete.` (or the PS equivalent); re-run skips done
  steps (idempotency guard fires).

### WS-4 — Makefile delegation & parity (`make bootstrap`)

Goal: kill the duplicate definition; `make bootstrap` becomes a thin wrapper.

- Change the `bootstrap` target to `./scripts/bootstrap.sh --yes`; move the
  `corepack enable` + `pnpm install` steps into the shared step list so both
  entry points keep Node setup (guarded by `command -v corepack`).
- Keep `make dev` / CI assumptions intact: `.venv` location and activated
  interpreter unchanged (`.venv/bin/uvicorn ...` keeps working).
- Update README quick start if the recommended command changes (it should
  not — `make bootstrap` remains the documented path).
- Deliverables: updated `Makefile`, README touch-up.
- Metric: fresh-clone `make bootstrap` produces a working `make test` +
  `pnpm lint` environment identical to today's target; second run is a no-op.

### WS-5 — CI wiring & automated tests

Goal: the bootstrap path is verified on every push, like the rest of the repo.

- Add `tests/test_bootstrap.py` (pytest, matching repo conventions): runs
  `scripts/bootstrap.sh --dry-run --platform {linux,macos,windows}` and
  asserts expected commands per platform (venv module path, activate path,
  requirements file), asserts exit codes for bad platforms, and asserts
  dry-run creates nothing (no `.venv` afterwards).
- Extend `.github/workflows/ci.yml` with a `bootstrap` job: `shellcheck`
  (if available, else `bash -n` fallback) over `scripts/**/*.sh`, then the
  pytest suite above; matrix-ready for a future macos/windows runner.
- Wire into `make` as `bootstrap-check` so local devs run the same gate.
- Deliverables: test file, CI job, Makefile target.
- Metric: CI fails on any preview/execution divergence or shellcheck
  regression; suite runs in <30s with no network access.

---

## Sequencing & dependencies

```
WS-1 dispatcher + contract ──> WS-2 preview ──> WS-3 platform modules
        │                                              │
        └──────────────> WS-4 Makefile parity <────────┘
                              │
                              v
                        WS-5 CI + tests (gates WS-1..WS-4 output)
```

WS-1 defines the module contract everything else consumes. WS-2 needs only
the contract (stubs suffice) and locks the preview/execute equivalence early.
WS-3 fills real steps per platform against that contract. WS-4 flips
`make bootstrap` over once modules are real. WS-5 lands last so it tests the
final surface, then guards it forever.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| Platform detection & CLI flags | WS-1 |
| Dispatcher + shared library contract | WS-1 |
| `print_bootstrap_preview(platform)` dry-run | WS-2 |
| Linux bootstrap module | WS-3 |
| macOS bootstrap module | WS-3 |
| Windows bootstrap module | WS-3 |
| Toolchain install (Python/Go/Rust) | WS-3 |
| `make bootstrap` single source of truth | WS-4 |
| CI gate + regression tests | WS-5 |

## Tests

```bash
bash -n scripts/bootstrap.sh                       # WS-1 syntax gate
./scripts/bootstrap.sh --dry-run                   # WS-2 preview, no side effects
./scripts/bootstrap.sh --dry-run --platform windows  # WS-2/WS-3 foreign-platform preview
./scripts/bootstrap.sh --bad-flag                  # WS-1 exits non-zero with usage
make bootstrap                                     # WS-4 fresh-clone parity
make bootstrap-check                               # WS-5 shellcheck + pytest gate
pytest tests/test_bootstrap.py                     # WS-5 preview/execution equivalence
```

The dry-run guarantee is the invariant under test: preview output equals the
commands a real run executes, on every platform, with zero filesystem or
network side effects. After WS-4, `make bootstrap` and
`./scripts/bootstrap.sh --yes` are the same code path, so the README quick
start needs no behavioral caveat.
