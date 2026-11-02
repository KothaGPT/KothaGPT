#!/usr/bin/env bash

require_pkg_manager() {
  if command -v apt-get >/dev/null 2>&1 \
    || command -v dnf >/dev/null 2>&1 \
    || command -v pacman >/dev/null 2>&1; then
    return 0
  fi
  warn "no supported package manager (apt-get/dnf/pacman); install missing toolchains manually"
  return 1
}

bootstrap_check() {
  if tool_missing python3 || tool_missing go || tool_missing cargo; then
    require_pkg_manager
  fi
  node_check
}

python_install_steps() {
  tool_missing python3 || return 0
  if command -v apt-get >/dev/null 2>&1; then
    emit "sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip"
  elif command -v dnf >/dev/null 2>&1; then
    emit "sudo dnf install -y python3 python3-pip"
  elif command -v pacman >/dev/null 2>&1; then
    emit "sudo pacman -S --noconfirm python python-pip"
  fi
}

go_install_steps() {
  tool_missing go || return 0
  if command -v apt-get >/dev/null 2>&1; then
    emit "sudo apt-get update && sudo apt-get install -y golang-go"
  elif command -v dnf >/dev/null 2>&1; then
    emit "sudo dnf install -y golang"
  elif command -v pacman >/dev/null 2>&1; then
    emit "sudo pacman -S --noconfirm go"
  fi
}

bootstrap_steps() {
  python_install_steps
  go_install_steps
  rust_steps
  python_venv_steps python3 "bin/activate"
  node_steps
}
