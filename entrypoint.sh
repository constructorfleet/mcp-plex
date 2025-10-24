#!/usr/bin/env bash
set -euo pipefail

VENV_BIN="/opt/venv/bin"
export PATH="${VENV_BIN}:${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# default command comes from Docker CMD; fall back to mcp-server if empty
CMD_NAME="${1:-mcp-server}"

# If the first arg is one of our scripts, run it; otherwise treat the whole line as a command.
case "$CMD_NAME" in
  mcp-server|load-data)
    shift || true
    # make sure the console script actually exists
    if ! command -v "$CMD_NAME" >/dev/null 2>&1; then
      echo "Console script '$CMD_NAME' not found in ${VENV_BIN}. Did install fail?" >&2
      ls -l "${VENV_BIN}" >&2 || true
      exit 127
    fi
    exec "$CMD_NAME" "$@"
    ;;
  *)
    # power-user mode: allow custom commands like "python -m something" or "bash"
    exec "$CMD_NAME" "$@"
    ;;
esac