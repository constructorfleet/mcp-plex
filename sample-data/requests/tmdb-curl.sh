#!/bin/bash
curl -H 'Authorization: Bearer $TMDB_TOKEN' -H 'Accept: application/json' 'https://api.themoviedb.org/3/tv/157239?append_to_response=reviews,actors,directors,writers'