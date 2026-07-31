# Makefile
.PHONY: up down shell test run-explore run-preprocess run-analytics download-data

up:
	docker compose up -d

down:
	docker compose down

shell:
	docker compose exec pyspark bash

test:
	docker compose exec pyspark bash -c "cd /home/jovyan && python -m pytest tests/ -v"

run-explore:
	docker compose exec pyspark bash -c "cd /home/jovyan && python work/01_explore.py"

run-preprocess:
	docker compose exec pyspark bash -c "cd /home/jovyan && python work/02_preprocess.py"

run-analytics:
	docker compose exec pyspark bash -c "cd /home/jovyan && python work/03_analytics.py"

download-data:
	mkdir -p data/taxi
	curl -L "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet" \
	     -o data/taxi/yellow_2022-01.parquet
run-ml:
	docker compose exec pyspark bash -c "cd /home/jovyan && python work/04_ml.py"
