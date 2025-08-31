#!/bin/bash
set -euo pipefail

if [[ -z "${PLEX_URL:-}" || -z "${PLEX_TOKEN:-}" ]]; then
  echo "PLEX_URL and PLEX_TOKEN must be set" >&2
  exit 1
fi

curl -H 'Accept: application/json' \
  "$PLEX_URL/library/metadata/61960?checkFiles=1&includeAllConcerts=1&includeBandwidths=1&includeChapters=1&includeChildren=1&includeConcerts=1&includeExtras=1&includeFields=1&includeGeolocation=1&includeLoudnessRamps=1&includeMarkers=1&includeOnDeck=1&includePopularLeaves=1&includePreferences=1&includeRelated=1&includeRelatedCount=1&includeReviews=1&includeStations=1&includeGuids=1&X-Plex-Token=$PLEX_TOKEN"

