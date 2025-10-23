#!/usr/bin/env sh
set -e

if [ "$#" -eq 0 ]; then
    set -- load-data
fi

command="$1"
shift

command_path="/app/.venv/bin/$command"

exec "$command_path" "$@"
