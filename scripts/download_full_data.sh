#!/bin/bash
# scripts/download_full_data.sh
# Downloads Yellow Taxi Parquet (2019–2022, 48 files) and uploads to S3.
set -e

BUCKET="${CS675_BUCKET:?Set CS675_BUCKET env var}"
PROFILE="${AWS_PROFILE:-ds}"
mkdir -p /tmp/taxi_download

for YEAR in 2019 2020 2021 2022; do
  for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    FILE="yellow_tripdata_${YEAR}-${MONTH}.parquet"
    URL="https://d37ci6vzurychx.cloudfront.net/trip-data/${FILE}"
    echo "Downloading $FILE..."
    curl -L "$URL" -o "/tmp/taxi_download/${FILE}"
    aws s3 cp "/tmp/taxi_download/${FILE}" \
      "s3://${BUCKET}/data/taxi/year=${YEAR}/month=${MONTH}/${FILE}" \
      --profile "${PROFILE}"
    rm "/tmp/taxi_download/${FILE}"
    echo "  Uploaded."
  done
done

echo "Done. 48 files uploaded."
