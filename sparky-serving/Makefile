.PHONY: build-baseline build-onnx build-optimized build-triton \
       run-baseline run-onnx run-optimized run-triton \
       benchmark stop clean

# --- Build ---
build-baseline:
	docker compose -f docker-compose-baseline.yaml build

build-onnx:
	docker compose -f docker-compose-onnx.yaml build

build-optimized:
	docker compose -f docker-compose-optimized.yaml build

build-triton:
	docker compose -f docker-compose-triton.yaml build

# --- Run ---
run-baseline:
	docker compose -f docker-compose-baseline.yaml up -d
	@echo "Baseline running on http://localhost:8000"

run-onnx:
	docker compose -f docker-compose-onnx.yaml up -d
	@echo "ONNX running on http://localhost:8000"

run-optimized:
	docker compose -f docker-compose-optimized.yaml up -d
	@echo "Optimized running on http://localhost:8000"

run-triton:
	docker compose -f docker-compose-triton.yaml up -d
	@echo "Triton running on http://localhost:8000"

# --- Benchmark (run against whichever server is up) ---
benchmark:
	mkdir -p results
	python3 scripts/benchmark.py --url http://localhost:8000

# --- Stop whichever is running ---
stop:
	docker compose -f docker-compose-baseline.yaml down 2>/dev/null || true
	docker compose -f docker-compose-onnx.yaml down 2>/dev/null || true
	docker compose -f docker-compose-optimized.yaml down 2>/dev/null || true
	docker compose -f docker-compose-triton.yaml down 2>/dev/null || true

# --- Clean ---
clean:
	rm -rf results/*
