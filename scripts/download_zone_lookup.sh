#!/bin/bash
# scripts/download_zone_lookup.sh
# Downloads the NYC TLC taxi zone lookup table (LocationID -> Borough/Zone name).
set -e

mkdir -p data
curl -L "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv" \
     -o data/taxi_zone_lookup.csv

echo "Done. Saved to data/taxi_zone_lookup.csv"
