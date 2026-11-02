#!/usr/bin/env bash

bootstrap_check() {
  if tool_missing py || tool_missing go || tool_missing cargo; then
    require_cmd winget "Install winget (App Installer) from the Microsoft Store."
  fi
  node_check
}

bootstrap_steps() {
  if tool_missing py; then
    emit "winget install -e --id Python.Python.3.12"
  fi
  if tool_missing go; then
    emit "winget install -e --id GoLang.Go"
  fi
  rust_steps
  python_venv_steps py "Scripts/activate"
  node_steps
}
