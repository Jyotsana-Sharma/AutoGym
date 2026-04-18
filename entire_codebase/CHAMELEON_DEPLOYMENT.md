# Chameleon Deployment: AutoGym / SparkyFitness ML System

This is the reproducible deployment path for the integrated system. The grader
or operator should not need to replay a long manual setup history. After the VM
is ready, the system is started from the repository with Docker Compose.

## What Runs

One Compose stack starts the SparkyFitness app, ML serving API, MLflow model
registry, retraining API, data/retraining jobs, PostgreSQL, Prometheus, Grafana,
and drift monitoring.

From scratch:

```bash
docker compose --profile pipeline up -d --build
```

Normal runtime after a model already exists in MLflow:

```bash
docker compose --profile runtime up -d
```

The `pipeline` profile creates Docker volumes automatically, prepares data,
trains/evaluates/registers the model, and starts the full system. The `runtime`
profile skips one-shot jobs and loads the current Production model from MLflow
Model Registry.

The Compose project is explicitly named `sparky-ml` in `docker-compose.yml`.
If Docker prints a warning such as `project has been loaded without an explicit
name from a symlink`, the VM is not using the latest compose file yet.

## Storage

Persistent source data lives in Chameleon Swift:

| Storage | Purpose | Created by |
|---|---|---|
| `proj04-sparky-raw-data` | Raw Food.com CSV files | Chameleon Swift |
| `proj04-sparky-training-data` | Optional compiled train/val/test copies | Chameleon Swift |
| Docker volumes in `docker-compose.yml` | MLflow registry/artifacts, Postgres data, model cache, Grafana/Prometheus state | Docker Compose |

Docker volumes are declared in [docker-compose.yml](/Users/jyotsanasharma/Desktop/AutoGym/entire_codebase/docker-compose.yml). Do not manually create them for grading or from-scratch reproducibility.

Important:

- If the VM is deleted, local Docker volumes are gone.
- If only `~/AutoGym` / `~/sparky-ml` directories were removed but Docker volumes remain, the MLflow registry may still contain the Production model.
- From-scratch reproducibility must not depend on pre-existing Docker volumes.

## 1. Create the Chameleon VM

Use KVM@TACC.

Recommended VM:

| Setting | Value |
|---|---|
| Image | `CC-Ubuntu22.04` |
| Flavor | `m1.xlarge` |
| Disk | 60 GB recommended, 30 GB minimum |
| Network | `sharednet1` |

Open these inbound TCP ports in the security group:

| Port | Service |
|---|---|
| 22 | SSH |
| 3004 | SparkyFitness UI |
| 3010 | SparkyFitness API |
| 5000 | MLflow |
| 8000 | ML serving API |
| 8080 | Retrain API |
| 3000 | Grafana |
| 9090 | Prometheus |

Attach a floating IP and SSH into the VM:

```bash
ssh cc@YOUR_IP
```

## 2. Install Docker

Run on the VM:

```bash
sudo apt-get update -y
sudo apt-get install -y git curl make docker-compose-plugin
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker cc
newgrp docker

docker --version
docker compose version
```

If `docker compose version` works, the VM is ready.

## 3. Clone the Repository

Run on the VM:

```bash
git clone --branch clean_branch --single-branch --recurse-submodules \
  https://github.com/Jyotsana-Sharma/AutoGym.git ~/AutoGym

ln -s ~/AutoGym/entire_codebase ~/sparky-ml
cd ~/sparky-ml
```

If the submodule did not populate:

```bash
cd ~/AutoGym
git submodule update --init --recursive
cd ~/sparky-ml
```

## 4. Configure `.env`

```bash
cp .env.example .env
nano .env
```

Set these values:

```bash
# Chameleon Swift credentials
OS_AUTH_URL=https://chi.tacc.chameleoncloud.org:5000/v3
OS_APPLICATION_CREDENTIAL_ID=your_credential_id
OS_APPLICATION_CREDENTIAL_SECRET=your_credential_secret
OS_REGION_NAME=CHI@TACC

# Public URLs / secrets
MLFLOW_TRACKING_URI=http://YOUR_IP:5000
SPARKY_FITNESS_FRONTEND_URL=http://YOUR_IP:3004
POSTGRES_PASSWORD=choose_a_strong_password
GRAFANA_PASSWORD=choose_a_strong_password
ROLLBACK_WEBHOOK_TOKEN=choose_a_random_token

# SparkyFitness database and auth secrets
SPARKY_FITNESS_DB_NAME=sparkyfitness_db
SPARKY_FITNESS_DB_USER=sparky
SPARKY_FITNESS_DB_PASSWORD=choose_a_strong_password
SPARKY_FITNESS_APP_DB_USER=sparky_app
SPARKY_FITNESS_APP_DB_PASSWORD=choose_another_strong_password
SPARKY_FITNESS_API_ENCRYPTION_KEY=paste_32_char_random_string_here
BETTER_AUTH_SECRET=paste_another_32_char_random_string_here

# Internal service URLs
ML_RECOMMENDATION_URL=http://sparky-serving:8000
ML_MODEL_NAME=sparky-ranker
```

Generate secret values with:

```bash
openssl rand -hex 32
```

## 5. Start the Full System From Scratch

This is the main reproducibility command:

```bash
cd ~/sparky-ml
docker compose --profile pipeline up -d --build
```

This command:

1. Builds all images from the single multi-stage Dockerfile.
2. Creates required Docker volumes automatically.
3. Builds the SparkyFitness API/UI images with the ML integration already baked in.
4. Runs data compilation.
5. Trains, evaluates, and registers a model in MLflow.
6. Starts serving, app, retraining API, monitoring, and drift monitor.

Check status:

```bash
docker compose --profile pipeline ps
```

Expected steady-state services include:

```text
sparky-postgres
sparky-mlflow
sparky-serving
sparky-retrain-api
sparky-drift-monitor
sparky-prometheus
sparky-grafana
sparky-postgres-exporter
sparkyfitness-db
sparkyfitness-server
sparkyfitness-frontend
```

The one-shot containers `sparkyfitness-setup`, `sparky-batch-pipeline`, and
`sparky-trainer` may show as exited after they complete successfully.

Expected port mappings include:

```text
sparky-serving            0.0.0.0:8000->8000/tcp
sparky-retrain-api        0.0.0.0:8080->8080/tcp
sparkyfitness-server      0.0.0.0:3010->3010/tcp
sparkyfitness-frontend    0.0.0.0:3004->8080/tcp
sparky-mlflow             0.0.0.0:5000->5000/tcp
sparky-grafana            0.0.0.0:3000->3000/tcp
sparky-prometheus         0.0.0.0:9090->9090/tcp
```

## 6. Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8080/health
curl -sf http://localhost:8080/model/production | python3 -m json.tool
curl -I http://localhost:3004
```

Expected ML serving output:

```json
{"status":"healthy","model_version":"v1","model_source":"mlflow_registry","model_name":"sparky-ranker"}
```

The exact model version may change after retraining, but `model_source` should
be `mlflow_registry` once a Production model exists. This means serving fetched
the model from MLflow Model Registry rather than retraining on startup.

Open in a browser:

| Service | URL |
|---|---|
| SparkyFitness UI | `http://YOUR_IP:3004` |
| ML serving docs | `http://YOUR_IP:8000/docs` |
| MLflow | `http://YOUR_IP:5000` |
| Grafana | `http://YOUR_IP:3000` |
| Prometheus | `http://YOUR_IP:9090` |

## 7. Normal Restart

After the first successful bootstrap, restart only the runtime services:

```bash
cd ~/sparky-ml
docker compose --profile runtime up -d
```

Runtime does not retrain. It loads the current Production model from MLflow
Model Registry and polls for newer Production versions.

If you pulled new Dockerfile or Compose changes, rebuild the runtime app images:

```bash
cd ~/sparky-ml
git pull
git submodule update --init --recursive
docker compose --profile runtime up -d --build --force-recreate
```

## 8. If Directories Were Removed But Docker Volumes Still Exist

If you deleted `~/AutoGym` or `~/sparky-ml` but did not remove Docker volumes,
the model registry may still exist. Re-clone the repo and try runtime before
retraining:

```bash
git clone --branch clean_branch --single-branch --recurse-submodules \
  https://github.com/Jyotsana-Sharma/AutoGym.git ~/AutoGym

ln -s ~/AutoGym/entire_codebase ~/sparky-ml
cd ~/sparky-ml
cp .env.example .env
nano .env

docker volume ls --format '{{.Name}}' \
  | grep -E 'mlflow-db|mlflow-artifacts|models-cache|training-data' || true

docker compose --profile runtime up -d --build
curl -sf http://localhost:8080/model/production | python3 -m json.tool
```

If the last command returns model metadata, no retraining is needed. If it
returns 404/no Production model, run:

```bash
docker compose --profile pipeline up -d --build
```

## 9. If The Frontend Port Resets Or The Site Cannot Be Reached

If `docker compose --profile runtime ps` shows:

```text
sparkyfitness-frontend    0.0.0.0:3004->8080/tcp
```

but this fails:

```bash
curl -I http://localhost:3004
```

inspect the frontend logs:

```bash
docker logs sparkyfitness-frontend --tail=120
```

The current image should run Vite on port `8080`. If logs show Vite listening
on `5173`, or if Docker still prints the symlink project-name warning, pull the
latest code and rebuild the app images:

```bash
cd ~/sparky-ml
git pull
git submodule update --init --recursive
head -12 docker-compose.yml

docker compose --profile runtime build --no-cache \
  sparkyfitness-frontend sparkyfitness-server
docker compose --profile runtime up -d --force-recreate \
  sparkyfitness-frontend sparkyfitness-server

docker logs sparkyfitness-frontend --tail=80
curl -I http://localhost:3004
```

The top of `docker-compose.yml` should include:

```yaml
name: sparky-ml
```

If `curl -I http://localhost:3004` works on the VM but
`http://YOUR_IP:3004` is not reachable from your laptop, check the Chameleon
security group and confirm inbound TCP port `3004` is open.

## 10. Common Operations

```bash
docker compose logs -f serving
docker compose logs -f trainer
docker compose --profile runtime ps

make train          # async retrain via API
make promote        # promote latest Staging model
make rollback       # rollback Production model
make smoke-test     # serving health + sample prediction
```

Stop containers but keep volumes:

```bash
docker compose down
```

Destructive cleanup, including local MLflow registry and databases:

```bash
docker compose down -v
```

## 11. Notes for Graders

The intended from-scratch reproduction path is:

```bash
git clone --branch clean_branch --single-branch --recurse-submodules \
  https://github.com/Jyotsana-Sharma/AutoGym.git
cd AutoGym/entire_codebase
cp .env.example .env
docker compose --profile pipeline up -d --build
```

The `.env` file is only needed for deployment-specific secrets and Chameleon
Swift credentials. Docker volumes, networks, and service wiring are defined in
the repository and created by Docker Compose.

## 12. If Docker Says "No Space Left on Device"

This means the VM root disk is full while Docker is building or extracting image
layers. It is usually a VM capacity issue, not an application bug.

First inspect disk usage:

```bash
df -h
docker system df
docker volume ls
```

Safe cleanup that preserves Docker volumes, including MLflow registry volumes:

```bash
docker builder prune -af
docker system prune -af
```

Do **not** run these unless you intentionally want to delete the local MLflow
registry, model artifacts, and databases:

```bash
docker compose down -v
docker volume prune
```

Then retry with lower build parallelism to reduce peak temporary disk use:

```bash
cd ~/sparky-ml
COMPOSE_PARALLEL_LIMIT=1 docker compose --profile pipeline up -d --build
```

If the VM still runs out of space, recreate or resize the VM with a larger root
disk. Use 60 GB or more for the full pipeline build.
