#!/usr/bin/env bash

bootstrap_check() {
  if tool_missing python3 || tool_missing go || tool_missing cargo; then
    require_cmd brew "Install Homebrew first: https://brew.sh"
  fi
  node_check
}

bootstrap_steps() {
  if tool_missing python3; then
    emit "brew install python@3.12"
  fi
  if tool_missing go; then
    emit "brew install go"
  fi
  rust_steps
  python_venv_steps python3 "bin/activate"
  node_steps
}
