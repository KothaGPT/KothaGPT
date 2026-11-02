#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT

PLATFORM=""
DRY_RUN=0
ASSUME_YES=0
STEPS=()

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap.sh [options]

Installs missing Python/Go/Rust toolchains via the platform package manager,
then bootstraps the Python venv and Node tooling.

Options:
  --platform <linux|macos|windows>   Target platform (default: auto-detect)
  --dry-run, --preview               Print the steps without executing
  --yes, -y                          Execute without confirmation prompt
  -h, --help                         Show this help and exit
EOF
}

fail() {
  printf '\033[0;31mERROR:\033[0m %s\n' "$*" >&2
  exit 1
}

detect_platform() {
  local uname_s
  uname_s="$(uname -s)"
  case "$uname_s" in
    Darwin) echo "macos" ;;
    MINGW* | MSYS* | CYGWIN*) echo "windows" ;;
    Linux)
      case "${OSTYPE:-}" in
        msys* | cygwin*) echo "windows" ;;
        *) echo "linux" ;;
      esac
      ;;
    *)
      if [[ "${OS:-}" == "Windows_NT" ]]; then
        echo "windows"
      else
        fail "cannot detect platform from '$uname_s'; pass --platform linux|macos|windows"
      fi
      ;;
  esac
}

parse_args() {
  while (($#)); do
    case "$1" in
      --platform)
        [[ $# -ge 2 ]] || fail "--platform requires a value (linux|macos|windows)"
        PLATFORM="$2"
        shift 2
        ;;
      --platform=*)
        PLATFORM="${1#*=}"
        shift
        ;;
      --dry-run | --preview)
        DRY_RUN=1
        shift
        ;;
      --yes | -y)
        ASSUME_YES=1
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        fail "unknown option: $1 (see --help)"
        ;;
    esac
  done
  if [[ -n "$PLATFORM" ]]; then
    case "$PLATFORM" in
      linux | macos | windows) ;;
      *) fail "invalid --platform '$PLATFORM' (expected linux|macos|windows)" ;;
    esac
  fi
}

load_module() {
  local platform="$1" module
  module="$SCRIPT_DIR/$platform/bootstrap.sh"
  [[ -f "$module" ]] || fail "no bootstrap module for platform '$platform' (expected $module)"
  source "$module"
}

collect_steps() {
  local raw line
  raw="$(bootstrap_steps)"
  STEPS=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && STEPS+=("$line")
  done <<<"$raw"
}

print_bootstrap_preview() {
  local platform="$1"
  local step n=0
  info "Kotha GPT bootstrap preview — $platform"
  for step in ${STEPS[@]+"${STEPS[@]}"}; do
    n=$((n + 1))
    printf '  %2d. %s\n' "$n" "$step"
  done
  if (( n == 0 )); then
    warn "no steps emitted for platform '$platform'"
  fi
}

confirm_execution() {
  local reply
  if (( ASSUME_YES )); then
    return 0
  fi
  if [[ ! -t 0 ]]; then
    fail "non-interactive session; re-run with --yes to execute"
  fi
  read -r -p "Proceed with bootstrap? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || fail "aborted"
}

execute_steps() {
  local step
  for step in ${STEPS[@]+"${STEPS[@]}"}; do
    run_or_print "$step"
  done
}

main() {
  parse_args "$@"
  source "$SCRIPT_DIR/lib/bootstrap_common.sh"
  if [[ -z "$PLATFORM" ]]; then
    PLATFORM="$(detect_platform)"
  fi
  export BOOTSTRAP_PLATFORM="$PLATFORM"
  load_module "$PLATFORM"
  if (( ! DRY_RUN )); then
    bootstrap_check
  fi
  collect_steps
  print_bootstrap_preview "$PLATFORM"
  if (( DRY_RUN )); then
    exit 0
  fi
  confirm_execution
  execute_steps
  info "Kotha GPT bootstrap complete."
}

main "$@"
