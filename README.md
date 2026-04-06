# Training

Use one `m1.xxlarge` VM at `KVM@TACC` with Docker, Docker Compose, and a floating IP.

Inputs at the repo root:
- `train.csv`
- `val.csv`
- `test.csv`

## Build
```bash
docker build -t sparky-train .
```

## MLflow
```bash
docker compose -f docker-compose.mlops.yml up -d
```

```bash
ssh -L 5000:localhost:5000 cc@<floating-ip>
```

Open `http://localhost:5000`.

## Train
```bash
export SPARKY_PROJECT_SUFFIX=<projectNN>
export SPARKY_CHAMELEON_SITE=KVM@TACC
export SPARKY_COMPUTE_FLAVOR=m1.xxlarge
```

Baseline:
```bash
docker run --rm --network host \
  -e MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  -e SPARKY_PROJECT_SUFFIX \
  -e SPARKY_CHAMELEON_SITE \
  -e SPARKY_COMPUTE_FLAVOR \
  -v "$PWD:/app" -w /app \
  sparky-train \
  python -m src.train --config configs/train/baseline_popularity.yaml
```

Main candidate:
```bash
docker run --rm --network host \
  -e MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  -e SPARKY_PROJECT_SUFFIX \
  -e SPARKY_CHAMELEON_SITE \
  -e SPARKY_COMPUTE_FLAVOR \
  -v "$PWD:/app" -w /app \
  sparky-train \
  python -m src.train --config configs/train/xgb_ranker.yaml
```
