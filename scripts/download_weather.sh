#!/bin/bash
# scripts/download_weather.sh
# Downloads NOAA GHCN-Daily weather for NYC Central Park (station USW00094728),
# filtered to 2019-2022 (matches the taxi dataset's year range).
set -e

mkdir -p data
curl -L "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv" \
  | python3 -c "
import csv, sys
w = csv.writer(sys.stdout)
w.writerow(['STATION', 'DATE', 'PRCP', 'SNOW'])
for row in csv.DictReader(sys.stdin):
    if '2019-01-01' <= row['DATE'] <= '2022-12-31':
        w.writerow([row['STATION'], row['DATE'], row['PRCP'].strip() or '0', row['SNOW'].strip() or '0'])
" > data/weather_central_park.csv

echo "Done. Saved to data/weather_central_park.csv"
