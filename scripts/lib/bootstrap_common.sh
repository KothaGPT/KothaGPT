#!/usr/bin/env bash

info() {
  printf '\033[1;32m==>\033[0m %s\n' "$*"
}

warn() {
  printf '\033[1;33mWARNING:\033[0m %s\n' "$*" >&2
}

fail() {
  printf '\033[0;31mERROR:\033[0m %s\n' "$*" >&2
  exit 1
}

emit() {
  printf '%s\n' "$1"
}

require_cmd() {
  local cmd="$1" hint="${2:-}"
  if command -v "$cmd" >/dev/null 2>&1; then
    return 0
  fi
  warn "'$cmd' is required but was not found in PATH. $hint"
  return 1
}

node_check() {
  require_cmd corepack "Install Node.js 18+ (bundles corepack): https://nodejs.org"
}

tool_missing() {
  ! command -v "$1" >/dev/null 2>&1
}

rust_steps() {
  if tool_missing cargo; then
    emit "bash \"$REPO_ROOT/scripts/install_rust\""
  fi
}

venv_python() {
  case "${BOOTSTRAP_PLATFORM:-}" in
    windows) printf '%s\n' "$REPO_ROOT/.venv/Scripts/python.exe" ;;
    *) printf '%s\n' "$REPO_ROOT/.venv/bin/python" ;;
  esac
}

venv_is_ready() {
  [[ -x "$(venv_python)" ]]
}

python_venv_steps() {
  local pybin="$1" activate_rel="$2"
  if ! venv_is_ready; then
    emit "$pybin -m venv \"$REPO_ROOT/.venv\""
  fi
  emit ". \"$REPO_ROOT/.venv/$activate_rel\" && python -m pip install --upgrade pip"
  emit ". \"$REPO_ROOT/.venv/$activate_rel\" && python -m pip install -r \"$REPO_ROOT/services/api/requirements.txt\""
  emit ". \"$REPO_ROOT/.venv/$activate_rel\" && python -m pip install ruff"
}

node_steps() {
  emit "corepack enable"
  emit "pnpm install"
}

run_or_print() {
  local cmd="$1"
  if (( ${DRY_RUN:-0} )); then
    printf '  \033[2m[dry-run]\033[0m %s\n' "$cmd"
  else
    info "+ $cmd"
    bash -c "$cmd"
  fi
}
