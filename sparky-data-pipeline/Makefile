.PHONY: setup build enrich ingest generate online-demo batch all clean teardown

setup:
	mkdir -p data output
	docker compose up -d postgres
	@echo "Waiting for Postgres..."; sleep 15
	docker compose exec postgres pg_isready -U sparky
	@echo "Postgres running. Place CSVs in data/"

build:
	docker compose build --no-cache

enrich:
	cp build_training_table.py data/
	cd data && python3 ../build_training_table.py

ingest: build
	docker compose --profile ingest run --rm ingest

generate: build
	docker compose --profile generate run --rm data-generator

online-demo: build
	docker compose --profile online-demo run --rm online-feature-demo

batch: build
	docker compose --profile batch run --rm batch-pipeline

all: setup build ingest enrich generate online-demo batch
	@echo "All pipelines complete"; ls -lh output/

clean:
	rm -rf output/*

teardown:
	docker compose down -v
