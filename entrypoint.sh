#!/usr/bin/env bash
set -euo pipefail

VENV_BIN="/opt/venv/bin"
export PATH="${VENV_BIN}:${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

resolve_secret_file_env() {
  local secret_name file_name secret_path secret_value

  for secret_name in PLEX_TOKEN TMDB_API_KEY QDRANT_API_KEY; do
    file_name="${secret_name}_FILE"
    secret_path="${!file_name:-}"

    if [[ -n "${secret_path}" && -z "${!secret_name:-}" ]]; then
      if [[ ! -r "${secret_path}" ]]; then
        echo "Secret file '${secret_path}' referenced by ${file_name} is not readable" >&2
        exit 1
      fi

      secret_value="$(<"${secret_path}")"
      export "${secret_name}=${secret_value}"
    fi

    unset "${file_name}"
  done
}

resolve_secret_file_env

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