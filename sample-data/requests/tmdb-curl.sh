#!/bin/bash
set -euo pipefail

if [[ -z "${TMDB_API_KEY:-}" ]]; then
  echo "TMDB_API_KEY must be set" >&2
  exit 1
fi

curl -H "Authorization: Bearer $TMDB_API_KEY" -H 'Accept: application/json' \
  'https://api.themoviedb.org/3/tv/157239?append_to_response=reviews,actors,directors,writers'

