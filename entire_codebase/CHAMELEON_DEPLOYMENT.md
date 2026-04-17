# Deploying SparkyFitness + ML Recommendation System on Chameleon Cloud

## Overview

This guide deploys the full system on a **KVM@TACC** virtual machine.
Your raw training data already lives in **CHI@TACC Swift object storage**
(`proj04-sparky-raw-data`) and stays there permanently — the KVM instance
pulls from it automatically during the data pipeline step.

```
CHI@TACC (data lives here, never moves)
  Swift: proj04-sparky-raw-data
    └── RAW_recipes.csv
    └── RAW_interactions.csv
  Swift: proj04-sparky-training-data
    └── train.csv / val.csv / test.csv
         │
         │  pulled over HTTPS using Application Credential
         ▼
KVM@TACC Virtual Machine  (everything runs here)
  ├── ML System  ~/sparky-ml/
  │   ├── sparky-postgres      :5433   ML training database
  │   ├── sparky-mlflow        :5000   Model registry + experiment tracking
  │   ├── sparky-serving       :8000   XGBoost ranking API  ← SparkyFitness calls this
  │   ├── sparky-retrain-api   :8080   Retraining webhook
  │   ├── sparky-drift-monitor         KS-test drift detection
  │   ├── sparky-prometheus    :9090   Metrics collection
  │   └── sparky-grafana       :3000   Monitoring dashboards
  │
  └── SparkyFitness  ~/SparkyFitness/
      ├── sparkyfitness-db             User data database
      ├── sparkyfitness-server :3010   Node.js API
      └── sparkyfitness-frontend :3004 React UI  ← open this in browser
```

**Estimated time:** 60–75 minutes (mostly Docker builds and model training)

---

## Phase 1 — Chameleon Account Setup

### Step 1 — Log in to KVM@TACC

1. Go to **https://kvm.tacc.chameleoncloud.org/project/**
2. Log in with your Chameleon credentials
3. Confirm the project selector (top-left) shows **proj04**

> KVM@TACC is used for the VM. Your data remains in CHI@TACC Swift —
> the two sites share the same project and Application Credential.

---

### Step 2 — Confirm your Application Credential exists

Your Application Credential was created on CHI@TACC and works across both sites.

1. Go to **Identity → Application Credentials**
2. If `sparky-ml` is listed — note the **ID** (you will need it for `.env`)
3. If it is missing, create a new one:
   - Name: `sparky-ml`
   - Expiration: leave blank
   - Roles: default
   - **Copy the ID and Secret immediately** — secret is shown only once

---

### Step 3 — Confirm your SSH key is uploaded

1. Go to **Identity → Key Pairs**
2. If your key is listed — you are ready
3. If missing — click **Import Public Key**, paste your public key:

```bash
# Run on your laptop to get your public key
cat ~/.ssh/id_rsa.pub
# or
cat ~/.ssh/id_ed25519.pub
```

---

## Phase 2 — Create a Lease

KVM@TACC requires a reservation lease before launching an instance.

### Step 4 — Create a lease

1. Go to **Reservations → Leases → + Create Lease**
2. Fill in:

| Field | Value |
|---|---|
| Lease Name | `sparky-lease` |
| Start Date | Now (or 2 minutes from now) |
| End Date | 7 days from today |

3. Click the **Reservations** tab inside the form → **+ Add Reservation**

| Field | Value |
|---|---|
| Reservation Type | Virtual Instance |
| Amount | 1 |
| Flavor | `m1.xlarge` |

4. Click **Create Lease**
5. Wait until the lease status shows **Active** (~1–2 minutes)

---

## Phase 3 — Security Group

### Step 5 — Create a security group

On KVM@TACC, Security Groups are under **Network → Security Groups**.

1. Go to **Network → Security Groups → + Create Security Group**
2. Name: `sparky-ports`, Description: `SparkyFitness ML system`
3. Click **Create Security Group**
4. Click **Manage Rules → + Add Rule** for each row below:

| Port | Protocol | Direction | Purpose |
|---|---|---|---|
| 22 | TCP | Ingress | SSH |
| 3004 | TCP | Ingress | SparkyFitness UI |
| 3010 | TCP | Ingress | SparkyFitness API |
| 5000 | TCP | Ingress | MLflow UI |
| 8000 | TCP | Ingress | ML Serving API |
| 8080 | TCP | Ingress | Retrain webhook |
| 3000 | TCP | Ingress | Grafana |
| 9090 | TCP | Ingress | Prometheus |

---

## Phase 4 — Launch the Instance

### Step 6 — Launch a VM

1. Go to **Compute → Instances → Launch Instance**
2. Fill in each tab of the wizard:

**Details tab**

| Field | Value |
|---|---|
| Instance Name | `sparky-ml` |
| Count | 1 |

**Source tab**

| Field | Value |
|---|---|
| Boot Source | Image |
| Image | `CC-Ubuntu22.04` (search and select) |
| Volume Size | `30` GB |
| Delete Volume on Instance Delete | Yes |

**Flavor tab**

Select `m1.xlarge` — 8 vCPU, 16 GB RAM.

**Networks tab**

Select `sharednet1`.

**Security Groups tab**

Select both `default` and `sparky-ports`.

**Key Pair tab**

Select your uploaded key.

3. Click **Launch Instance** — wait until status shows **Active** (~2 minutes)

---

### Step 7 — Attach a Floating IP

1. Go to **Compute → Instances**
2. Find `sparky-ml` → click the **Actions** dropdown → **Associate Floating IP**
3. If the list is empty click **Allocate IP to Project** first, then associate
4. **Write down the floating IP** — you will use it everywhere below

Replace `YOUR_IP` with this floating IP for the rest of the guide.

---

## Phase 5 — Server Setup

### Step 8 — SSH into the instance

Run this on your **laptop**:

```bash
ssh cc@YOUR_IP
```

> The default user on all Chameleon Ubuntu images is `cc`.
> If the connection times out, wait 2 more minutes — cloud-init is still running.

---

### Step 9 — Install Docker

Run all of the following on the **Chameleon instance**:

```bash
# Update packages first
sudo apt-get update -y && sudo apt-get upgrade -y

# Install Docker using the official convenience script
# This handles GPG keys, repository setup, and package install in one step
curl -fsSL https://get.docker.com | sudo sh

# Install supporting tools
sudo apt-get install -y make git curl docker-compose-plugin

# Allow running Docker without sudo
sudo usermod -aG docker cc
newgrp docker

# Verify both commands work
docker --version
docker compose version
```

Expected output:
```
Docker version 26.x.x
Docker Compose version v2.x.x
```

> **Note:** `docker-ce`, `docker-ce-cli`, and `containerd.io` are package names
> installed by the script above — they are NOT commands you run directly.
> The only command you use is `docker`.
>
> If `docker compose version` says "command not found":
> ```bash
> sudo apt-get install -y docker-compose-plugin
> ```

---

### Step 10 — Install Node.js 20 and pnpm

SparkyFitness requires Node.js 20 and pnpm.

```bash
# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install pnpm
sudo npm install -g pnpm

# Verify
node --version    # must be v20.x.x
pnpm --version
```

---

## Phase 6 — Get the Code onto the Instance

All commands in this phase run on the **Chameleon instance** (inside your SSH session).

### Step 11 — Clone the ML system from GitHub

The ML system lives at:
`https://github.com/Jyotsana-Sharma/AutoGym` — branch `clean_branch` — subfolder `entire_codebase/`

```bash
# Clone the repo
git clone --branch clean_branch --single-branch \
  https://github.com/Jyotsana-Sharma/AutoGym.git ~/AutoGym

# Remove ~/sparky-ml if it already exists (plain directory or wrong symlink)
# Safe to run even if it doesn't exist yet
if [ -L ~/sparky-ml ]; then
  rm ~/sparky-ml                   # remove wrong symlink
elif [ -d ~/sparky-ml ]; then
  # save any .env inside before deleting
  [ -f ~/sparky-ml/.env ] && cp ~/sparky-ml/.env ~/AutoGym/entire_codebase/.env
  rm -rf ~/sparky-ml
fi

# Create the correct symlink
ln -s ~/AutoGym/entire_codebase ~/sparky-ml

# Verify the symlink (must show arrow → not a plain directory listing)
ls -la ~ | grep sparky-ml
# Expected:  lrwxrwxrwx ... sparky-ml -> /home/cc/AutoGym/entire_codebase
ls ~/sparky-ml
# Should show: Makefile  configs  docker  src  requirements  docker-compose.yml ...
```

> If the repo is private, use a Personal Access Token:
> ```bash
> git clone --branch clean_branch --single-branch \
>   https://<your_github_username>:<your_token>@github.com/Jyotsana-Sharma/AutoGym.git ~/AutoGym
> ```

---

### Step 12 — Clone SparkyFitness and apply the recommendation feature

The integration files are **already on the VM** at
`~/AutoGym/entire_codebase/sparkyfitness-integration/` (cloned in Step 11).
No fork, no rsync from your laptop needed.

```bash
# 1. Clone the upstream SparkyFitness app
git clone https://github.com/CodeWithCJ/SparkyFitness.git ~/SparkyFitness

# Convenience variables
INTEG=~/AutoGym/entire_codebase/sparkyfitness-integration
SF=~/SparkyFitness

# 2. Copy backend integration files
mkdir -p $SF/SparkyFitnessServer/schemas \
         $SF/SparkyFitnessServer/models \
         $SF/SparkyFitnessServer/services \
         $SF/SparkyFitnessServer/routes \
         $SF/SparkyFitnessServer/db

cp $INTEG/SparkyFitnessServer/schemas/recommendationSchemas.ts \
   $SF/SparkyFitnessServer/schemas/

cp $INTEG/SparkyFitnessServer/models/recommendationRepository.ts \
   $SF/SparkyFitnessServer/models/

cp $INTEG/SparkyFitnessServer/services/recommendationService.ts \
   $SF/SparkyFitnessServer/services/

cp $INTEG/SparkyFitnessServer/routes/recommendationRoutes.ts \
   $SF/SparkyFitnessServer/routes/

cp $INTEG/SparkyFitnessServer/db/add_recommendations.sql \
   $SF/SparkyFitnessServer/db/

# 3. Copy frontend integration files
mkdir -p $SF/SparkyFitnessFrontend/src/api \
         $SF/SparkyFitnessFrontend/src/hooks/Foods \
         $SF/SparkyFitnessFrontend/src/pages/Foods

cp $INTEG/SparkyFitnessFrontend/src/api/recommendations.ts \
   $SF/SparkyFitnessFrontend/src/api/

cp $INTEG/SparkyFitnessFrontend/src/hooks/Foods/useRecommendations.ts \
   $SF/SparkyFitnessFrontend/src/hooks/Foods/

cp $INTEG/SparkyFitnessFrontend/src/pages/Foods/RecipeRecommendations.tsx \
   $SF/SparkyFitnessFrontend/src/pages/Foods/

# 4. Wire the route into SparkyFitnessServer.ts
# (guarded — upstream repo may already have these lines)
grep -q "recommendationRoutes" $SF/SparkyFitnessServer/SparkyFitnessServer.ts || \
  sed -i "s|import foodRoutes from './routes/foodRoutes.js';|import foodRoutes from './routes/foodRoutes.js';\nimport recommendationRoutes from './routes/recommendationRoutes.js';|" \
      $SF/SparkyFitnessServer/SparkyFitnessServer.ts

grep -q "api/recommendations" $SF/SparkyFitnessServer/SparkyFitnessServer.ts || \
  sed -i "s|app.use('/api/foods', foodRoutes);|app.use('/api/foods', foodRoutes);\napp.use('/api/recommendations', recommendationRoutes);|" \
      $SF/SparkyFitnessServer/SparkyFitnessServer.ts

# 5. Wire the RecipeRecommendations panel into the Foods page
# (guarded — upstream repo may already have these lines)
grep -q "RecipeRecommendations" $SF/SparkyFitnessFrontend/src/pages/Foods/Foods.tsx || \
  sed -i "s|import MealManagement|import RecipeRecommendations from './RecipeRecommendations';\nimport MealManagement|" \
      $SF/SparkyFitnessFrontend/src/pages/Foods/Foods.tsx

grep -q "RecipeRecommendations" $SF/SparkyFitnessFrontend/src/pages/Foods/Foods.tsx || \
  sed -i "s|<MealManagement|<Card>\n          <CardContent className=\"pt-6\">\n            <RecipeRecommendations limit={6} \/>\n          <\/CardContent>\n        <\/Card>\n        <MealManagement|" \
      $SF/SparkyFitnessFrontend/src/pages/Foods/Foods.tsx

# 6. Verify the edits landed correctly
grep -n "recommendationRoutes\|RecipeRecommendations" \
    $SF/SparkyFitnessServer/SparkyFitnessServer.ts \
    $SF/SparkyFitnessFrontend/src/pages/Foods/Foods.tsx
```

You should see two matching lines in each file — the import and the usage.

> **Important:** The upstream `CodeWithCJ/SparkyFitness` repository already contains
> broken stub versions of `recommendationRoutes.ts`, `recommendationService.ts`, and
> `recommendationRepository.ts` (class-based implementations with no `export default`).
> The `cp` commands above **must run** to overwrite them with the correct versions.
> If you skip this step or clone SparkyFitness after a repo update that changes these
> files, the server will crash-loop with:
> ```
> SyntaxError: The requested module './routes/recommendationRoutes.js'
>   does not provide an export named 'default'
> ```
> Fix: re-run all `cp` commands from step 3 of this section, then rebuild.

---

### Step 13 — Download the raw data on the VM

The full data pipeline runs entirely on the Chameleon VM.
Your raw CSVs are already in **CHI@TACC Swift** (`proj04-sparky-raw-data`).
Download them directly onto the VM now:

```bash
# Install pip3 if not already present
sudo apt install -y python3-pip

# Install swiftclient on the VM
pip3 install python-swiftclient

# Create the data directory using the explicit path (avoids symlink ordering issues)
mkdir -p ~/AutoGym/entire_codebase/data

# Download both CSVs from CHI@TACC Swift directly onto the VM
python3 - <<'EOF'
import swiftclient, os, pathlib

conn = swiftclient.Connection(
    authurl="https://chi.tacc.chameleoncloud.org:5000/v3",
    auth_version="3",
    os_options={
        "auth_type": "v3applicationcredential",
        "application_credential_id": os.environ["OS_APPLICATION_CREDENTIAL_ID"],
        "application_credential_secret": os.environ["OS_APPLICATION_CREDENTIAL_SECRET"],
        "region_name": "CHI@TACC",
    }
)

data_dir = pathlib.Path.home() / "AutoGym" / "entire_codebase" / "data"
data_dir.mkdir(parents=True, exist_ok=True)

for filename in ["RAW_recipes.csv", "RAW_interactions.csv"]:
    print(f"Downloading {filename} ...")
    _, content = conn.get_object("proj04-sparky-raw-data", filename)
    (data_dir / filename).write_bytes(content)
    size_mb = len(content) / 1e6
    print(f"  Saved {filename} ({size_mb:.1f} MB)")

conn.close()
print("Done.")
EOF
```

> Get your credentials from **CHI@TACC → Identity → Application Credentials**.
> Export them before running the script:
>
> ```bash
> export OS_APPLICATION_CREDENTIAL_ID=your_id_here
> export OS_APPLICATION_CREDENTIAL_SECRET=your_secret_here
> ```

Verify the files downloaded correctly:

```bash
ls -lh ~/AutoGym/entire_codebase/data/
# Expected:
# RAW_recipes.csv       ~50 MB
# RAW_interactions.csv  ~200 MB
```

**If Swift is unreachable — download from Kaggle directly on the VM:**

```bash
# Install Kaggle CLI
pip3 install kaggle

# Set up Kaggle credentials
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<EOF
{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_API_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json

# Download Food.com dataset directly onto the VM
cd ~/AutoGym/entire_codebase/data
kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions
unzip food-com-recipes-and-user-interactions.zip
mv recipes.csv RAW_recipes.csv 2>/dev/null || true
mv interactions.csv RAW_interactions.csv 2>/dev/null || true

ls -lh ~/AutoGym/entire_codebase/data/
```

---

## Phase 7 — Configure Environment Variables

### Step 14 — Configure the ML system `.env`

Back on the **Chameleon instance**:

```bash
cd ~/sparky-ml
cp .env.example .env
nano .env
```

Update these values — leave everything else as the default:

```bash
# ── CHI@TACC Swift credentials (data lives here) ──────────────────────────
# These point to CHI@TACC even though your VM is on KVM@TACC — this is correct.
OS_AUTH_URL=https://chi.tacc.chameleoncloud.org:5000/v3
OS_APPLICATION_CREDENTIAL_ID=your_credential_id_here
OS_APPLICATION_CREDENTIAL_SECRET=your_credential_secret_here
OS_REGION_NAME=CHI@TACC

# ── PostgreSQL ─────────────────────────────────────────────────────────────
POSTGRES_PASSWORD=choose_a_strong_password

# ── MLflow — use YOUR KVM floating IP ─────────────────────────────────────
MLFLOW_TRACKING_URI=http://YOUR_IP:5000

# ── Serving ───────────────────────────────────────────────────────────────
LOG_PREDICTIONS=true
MODEL_POLL_INTERVAL_SEC=60
```

Save with `Ctrl+O`, exit with `Ctrl+X`.

---

### Step 15 — Configure SparkyFitness `.env`

```bash
cd ~/SparkyFitness/docker
cp .env.example .env
nano .env
```

Fill in the **required** values below. Everything else can be left blank.

```bash
# ── Database — values you invent (used to create the DB on first boot) ──────
SPARKY_FITNESS_DB_NAME=sparkyfitness_db
SPARKY_FITNESS_DB_USER=sparky
SPARKY_FITNESS_DB_PASSWORD=choose_a_strong_password
SPARKY_FITNESS_APP_DB_USER=sparky_app
SPARKY_FITNESS_APP_DB_PASSWORD=choose_another_strong_password
SPARKY_FITNESS_DB_HOST=sparkyfitness-db

# ── Server — keep as shown ───────────────────────────────────────────────────
SPARKY_FITNESS_SERVER_HOST=sparkyfitness-server
SPARKY_FITNESS_SERVER_PORT=3010

# ── Frontend URL — use YOUR KVM floating IP ──────────────────────────────────
# CRITICAL: This must match the IP you use in your browser exactly.
# A typo here causes login to silently fail ("almost there" screen, no session).
# After saving, verify with: grep FRONTEND_URL .env
SPARKY_FITNESS_FRONTEND_URL=http://YOUR_IP:3004

# ── Secret keys — generate with openssl (see below) ─────────────────────────
SPARKY_FITNESS_API_ENCRYPTION_KEY=paste_32_char_random_string_here
BETTER_AUTH_SECRET=paste_another_32_char_random_string_here

# ── ML Recommendation feature ────────────────────────────────────────────────
ML_RECOMMENDATION_URL=http://YOUR_IP:8000
ML_MODEL_NAME=sparky-ranker

# ── Optional — leave blank to disable these features ────────────────────────
# SPARKY_FITNESS_EMAIL_HOST=        # SMTP server (for password reset emails)
# SPARKY_FITNESS_EMAIL_PORT=        # SMTP port
# SPARKY_FITNESS_EMAIL_USER=        # SMTP username
# SPARKY_FITNESS_EMAIL_PASS=        # SMTP password
# SPARKY_FITNESS_EMAIL_FROM=        # From address for emails
# SPARKY_FITNESS_EMAIL_SECURE=      # true/false for TLS
# SPARKY_FITNESS_ADMIN_EMAIL=       # Email that gets admin panel access
# SPARKY_FITNESS_DISABLE_SIGNUP=    # Set to true to block new registrations
# SPARKY_FITNESS_LOG_LEVEL=         # Defaults to info
```

> The `WARN: variable is not set` messages for the optional fields are harmless —
> they just mean those features (email, admin panel) are disabled, which is fine.

**Generate the two secret keys** — run this twice and use each output:

```bash
openssl rand -hex 32
```

Save with `Ctrl+O`, exit with `Ctrl+X`.

---

## Phase 8 — Start the ML System

### Step 16 — Build all Docker images

```bash
cd ~/sparky-ml

# Builds all images — takes 5–10 minutes on first run
make build
```

Wait until the build completes before moving on.

---

### Step 17 — Start infrastructure (PostgreSQL + MLflow)

```bash
make run-infra

# This command waits until both services pass health checks before returning.
# Verify MLflow is up:
curl http://localhost:5000/health
# Expected response: {"status": "ok"}
```

---

### Step 18 — Verify Swift connectivity (optional but recommended)

```bash
# Quick check that your KVM instance can reach CHI@TACC Swift
python3 - <<'EOF'
import os, swiftclient
try:
    conn = swiftclient.Connection(
        authurl=os.environ.get("OS_AUTH_URL", "https://chi.tacc.chameleoncloud.org:5000/v3"),
        auth_version="3",
        os_options={
            "auth_type": "v3applicationcredential",
            "application_credential_id": open(os.path.expanduser("~/sparky-ml/.env")).read().split("OS_APPLICATION_CREDENTIAL_ID=")[1].split()[0],
            "application_credential_secret": open(os.path.expanduser("~/sparky-ml/.env")).read().split("OS_APPLICATION_CREDENTIAL_SECRET=")[1].split()[0],
        }
    )
    containers = [c['name'] for c in conn.get_account()[1]]
    print("Swift reachable. Containers:", containers)
except Exception as e:
    print("Swift check failed:", e)
EOF
```

Expected output:
```
Swift reachable. Containers: ['proj04-sparky-raw-data', 'proj04-sparky-training-data']
```

If this fails, check your `OS_APPLICATION_CREDENTIAL_ID` and `OS_APPLICATION_CREDENTIAL_SECRET` in `.env`.

---

### Step 19 — Run the data pipeline

```bash
make data

# The pipeline:
#   1. Downloads RAW_recipes.csv + RAW_interactions.csv from CHI@TACC Swift
#   2. Runs feature engineering (build_training_table.py)
#   3. Writes train/val/test splits to the Docker training-data volume
#   4. Uploads versioned splits back to proj04-sparky-training-data in Swift
#
# Takes 3–8 minutes. Completed output looks like:
# DONE: Batch Pipeline Complete
# Train: ~80,000 rows  Val: ~10,000 rows  Test: ~10,000 rows
```

---

### Step 20 — Train the model

```bash
make train-direct
```

> Note: `make train-direct` uses `--rm` so the trainer container is automatically
> deleted after it finishes. Do not run `docker logs -f sparky-trainer` — the
> output prints directly to your terminal. Training takes 5–15 minutes.

Completed output looks like:
```
ndcg_at_10=0.8148  ✓ passes threshold 0.55
Registered as version 1 in Staging. NDCG@10=0.8148
```

---

### Step 21 — Promote model to Production

`make promote` requires the retrain-api container to be running, which is not
started yet at this point. Promote directly via the MLflow REST API instead:

```bash
curl -s -X POST http://localhost:5000/api/2.0/mlflow/model-versions/transition-stage \
  -H "Content-Type: application/json" \
  -d '{"name": "sparky-ranker", "version": "1", "stage": "Production", "archive_existing_versions": false}' \
  | python3 -m json.tool
```

Expected response contains `"current_stage": "Production"`.

---

### Step 22 — Start serving and monitoring

```bash
# Serving API (port 8000) + retrain webhook (port 8080)
make run-serving

# Prometheus + Grafana + drift monitor
make run-monitoring

# Confirm all containers are running
make status
```

Expected output:
```
sparky-postgres        Up    0.0.0.0:5433->5432/tcp
sparky-mlflow          Up    0.0.0.0:5000->5000/tcp
sparky-serving         Up    0.0.0.0:8000->8000/tcp
sparky-retrain-api     Up    0.0.0.0:8080->8080/tcp
sparky-drift-monitor   Up
sparky-prometheus      Up    0.0.0.0:9090->9090/tcp
sparky-grafana         Up    0.0.0.0:3000->3000/tcp
```

---

### Step 23 — Test the ML serving API

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-1",
    "model_name": "sparky-ranker",
    "instances": [{
      "user_id": "1", "recipe_id": "100",
      "rating": 4.2,
      "minutes": 30, "n_ingredients": 8, "n_steps": 5,
      "avg_rating": 4.2, "n_reviews": 150, "cuisine": "italian",
      "calories": 45.0, "total_fat": 10.0, "sugar": 5.0,
      "sodium": 8.0, "protein": 20.0, "saturated_fat": 4.0, "carbohydrate": 15.0,
      "total_fat_g": 7.8, "sugar_g": 2.5, "sodium_g": 0.18,
      "protein_g": 10.0, "saturated_fat_g": 3.1, "carbohydrate_g": 41.25,
      "has_egg": 0, "has_fish": 0, "has_milk": 1, "has_nuts": 0,
      "has_peanut": 0, "has_sesame": 0, "has_shellfish": 0, "has_soy": 0, "has_wheat": 1,
      "daily_calorie_target": 2000, "protein_target_g": 150,
      "carbs_target_g": 200, "fat_target_g": 65,
      "user_vegetarian": 0, "user_vegan": 0, "user_gluten_free": 0,
      "user_dairy_free": 0, "user_low_sodium": 0, "user_low_fat": 0,
      "history_pc1": 0.1, "history_pc2": -0.3, "history_pc3": 0.5,
      "history_pc4": 0.0, "history_pc5": 0.2, "history_pc6": -0.1
    }]
  }' | python3 -m json.tool
```

You should get a JSON response with a `predictions` array containing a numeric score.

---

## Phase 9 — Start SparkyFitness

### Step 24 — Create the build override file

The upstream `docker-compose.prod.yml` pulls pre-built Docker Hub images that do
not contain the recommendation feature. This override file swaps in local builds
so your integration code is included.

```bash
cd ~/SparkyFitness/docker

cat > docker-compose.sparky-build.yml << 'EOF'
services:
  sparkyfitness-server:
    build:
      context: ..
      dockerfile: docker/Dockerfile.backend
    container_name: sparkyfitness-server
    environment:
      ML_RECOMMENDATION_URL: ${ML_RECOMMENDATION_URL:-http://localhost:8000}

  sparkyfitness-frontend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.frontend
EOF
```

---

### Step 25 — Build and start SparkyFitness

```bash
cd ~/SparkyFitness/docker

# Build from local source and start (takes 3–5 minutes on first run)
docker compose \
  -f docker-compose.prod.yml \
  -f docker-compose.sparky-build.yml \
  --env-file .env \
  up -d --build

# Check all containers started
docker compose \
  -f docker-compose.prod.yml \
  -f docker-compose.sparky-build.yml \
  --env-file .env \
  ps
```

Wait until all three show healthy or running:

```
sparkyfitness-db        running (healthy)
sparkyfitness-server    running (healthy)
sparkyfitness-frontend  running
```

If `sparkyfitness-server` is not healthy after 60 seconds:
```bash
docker logs sparkyfitness-server --tail 30
```

---

### Step 26 — Connect SparkyFitness to the ML network

The SparkyFitness backend needs to reach the ML serving container.
Run this once after every fresh deploy:

```bash
# Connect the server container to the ML network
docker network connect sparky-ml_sparky-net sparkyfitness-server

# Verify the connection was made
docker network inspect sparky-ml_sparky-net | grep sparkyfitness
# Should show sparkyfitness-server in the output
```

---

## Phase 10 — Open the App and See the Feature

### Step 27 — Open SparkyFitness in your browser

On your laptop:

```
http://YOUR_IP:3004
```

---

### Step 28 — Set up your account and goals

1. Click **Sign Up** and create an account
2. Complete onboarding — set your **daily nutrition goals** (Calories, Protein, Carbs, Fat targets)
   - These are what the ML model uses to rank meals for you personally
3. Go to **Settings → Nutrients** to confirm goals are saved

---

### Step 29 — Add meal templates

The recommendation system ranks meals that exist in SparkyFitness's meal library.

1. Go to **Foods** in the left sidebar
2. Scroll down to **Meal Management → + Create Meal**
3. Add a few meals with foods and nutritional data
4. Mark at least some as **Public** so they appear in the candidate pool

---

### Step 30 — See the recommendations

1. Go to **Foods** in the left sidebar
2. The **"Recommended for You"** panel appears at the top of the page
3. Each card shows:
   - Meal name and description
   - Nutrition badges (kcal, P, C, F)
   - Personalised reason: e.g. *"High protein — fits your 150 g target"*
   - **Add to Diary**, **★ Save**, **✕ Dismiss** buttons
4. Clicking **Add to Diary** logs the meal and sends feedback to the ML retraining loop

---

### Step 31 — Confirm ML is being called

```bash
# Watch the serving logs while loading the Foods page
docker logs -f sparky-serving
```

When you open the Foods page you should see:
```
POST /predict  200 OK  batch_size=12  latency=45ms  model_version=1
```

---

## All URLs After Deployment

Replace `YOUR_IP` with your KVM floating IP.

| Service | URL | Credentials |
|---|---|---|
| **SparkyFitness UI** — recommendation feature here | `http://YOUR_IP:3004` | your account |
| SparkyFitness API | `http://YOUR_IP:3010` | — |
| ML Serving API health | `http://YOUR_IP:8000/health` | — |
| ML Serving API docs | `http://YOUR_IP:8000/docs` | — |
| MLflow model registry | `http://YOUR_IP:5000` | — |
| Grafana dashboards | `http://YOUR_IP:3000` | admin / sparky_admin |
| Prometheus metrics | `http://YOUR_IP:9090` | — |
| Retrain webhook status | `http://YOUR_IP:8080/status` | — |

---

## Restarting After a VM Reboot

If the instance reboots (but is not deleted):

```bash
# ML system
cd ~/sparky-ml
make run-infra && make run-serving && make run-monitoring

# SparkyFitness
cd ~/SparkyFitness/docker
docker compose -f docker-compose.prod.yml -f docker-compose.sparky-build.yml --env-file .env up -d

# Reconnect networks
docker network connect sparky-ml_sparky-net sparkyfitness-server
```

---

## Resuming After Lease Expiry / Instance Deleted

When a KVM lease expires the VM is deleted but **CHI@TACC Swift data is never deleted**.

**What persists on CHI@TACC:**
- `proj04-sparky-raw-data` — raw CSVs
- `proj04-sparky-training-data` — compiled train/val/test splits
- Application Credential and SSH key

**What needs to be redone:**
- New lease (Step 4)
- New instance (Steps 5–7)
- Docker + Node install (Steps 8–10)
- Clone ML code + SparkyFitness + apply integration (Steps 11–12)
- Both `.env` files with the **new floating IP** (Steps 14–15)
- Full deployment (Steps 16–31)

```bash
# Quick redeploy after new instance is up

# 1. Clone ML system (already has sparkyfitness-integration/ inside)
git clone --branch clean_branch --single-branch \
  https://github.com/Jyotsana-Sharma/AutoGym.git ~/AutoGym
ln -s ~/AutoGym/entire_codebase ~/sparky-ml

# Clone upstream SparkyFitness and apply the recommendation feature
git clone https://github.com/CodeWithCJ/SparkyFitness.git ~/SparkyFitness

INTEG=~/AutoGym/entire_codebase/sparkyfitness-integration
SF=~/SparkyFitness

mkdir -p $SF/SparkyFitnessServer/schemas $SF/SparkyFitnessServer/models \
         $SF/SparkyFitnessServer/services $SF/SparkyFitnessServer/routes \
         $SF/SparkyFitnessServer/db $SF/SparkyFitnessFrontend/src/api \
         $SF/SparkyFitnessFrontend/src/hooks/Foods \
         $SF/SparkyFitnessFrontend/src/pages/Foods

cp $INTEG/SparkyFitnessServer/schemas/recommendationSchemas.ts   $SF/SparkyFitnessServer/schemas/
cp $INTEG/SparkyFitnessServer/models/recommendationRepository.ts $SF/SparkyFitnessServer/models/
cp $INTEG/SparkyFitnessServer/services/recommendationService.ts  $SF/SparkyFitnessServer/services/
cp $INTEG/SparkyFitnessServer/routes/recommendationRoutes.ts     $SF/SparkyFitnessServer/routes/
cp $INTEG/SparkyFitnessServer/db/add_recommendations.sql         $SF/SparkyFitnessServer/db/
cp $INTEG/SparkyFitnessFrontend/src/api/recommendations.ts       $SF/SparkyFitnessFrontend/src/api/
cp $INTEG/SparkyFitnessFrontend/src/hooks/Foods/useRecommendations.ts \
   $SF/SparkyFitnessFrontend/src/hooks/Foods/
cp $INTEG/SparkyFitnessFrontend/src/pages/Foods/RecipeRecommendations.tsx \
   $SF/SparkyFitnessFrontend/src/pages/Foods/

grep -q "recommendationRoutes" $SF/SparkyFitnessServer/SparkyFitnessServer.ts || \
  sed -i "s|import foodRoutes from './routes/foodRoutes.js';|import foodRoutes from './routes/foodRoutes.js';\nimport recommendationRoutes from './routes/recommendationRoutes.js';|" \
      $SF/SparkyFitnessServer/SparkyFitnessServer.ts
grep -q "api/recommendations" $SF/SparkyFitnessServer/SparkyFitnessServer.ts || \
  sed -i "s|app.use('/api/foods', foodRoutes);|app.use('/api/foods', foodRoutes);\napp.use('/api/recommendations', recommendationRoutes);|" \
      $SF/SparkyFitnessServer/SparkyFitnessServer.ts
grep -q "RecipeRecommendations" $SF/SparkyFitnessFrontend/src/pages/Foods/Foods.tsx || \
  sed -i "s|import MealManagement|import RecipeRecommendations from './RecipeRecommendations';\nimport MealManagement|" \
      $SF/SparkyFitnessFrontend/src/pages/Foods/Foods.tsx
grep -q "RecipeRecommendations" $SF/SparkyFitnessFrontend/src/pages/Foods/Foods.tsx || \
  sed -i "s|<MealManagement|<Card>\n          <CardContent className=\"pt-6\">\n            <RecipeRecommendations limit={6} \/>\n          <\/CardContent>\n        <\/Card>\n        <MealManagement|" \
      $SF/SparkyFitnessFrontend/src/pages/Foods/Foods.tsx

# 2. Configure .env files with the NEW floating IP
cp ~/sparky-ml/.env.example ~/sparky-ml/.env
nano ~/sparky-ml/.env          # update YOUR_IP and credentials

cp ~/SparkyFitness/docker/.env.example ~/SparkyFitness/docker/.env
nano ~/SparkyFitness/docker/.env   # update YOUR_IP and credentials

# 3. Download raw data from CHI@TACC Swift directly onto the VM
export OS_APPLICATION_CREDENTIAL_ID=your_id
export OS_APPLICATION_CREDENTIAL_SECRET=your_secret
pip3 install python-swiftclient
mkdir -p ~/AutoGym/entire_codebase/data
python3 - <<'EOF'
import swiftclient, os, pathlib
conn = swiftclient.Connection(
    authurl="https://chi.tacc.chameleoncloud.org:5000/v3",
    auth_version="3",
    os_options={
        "auth_type": "v3applicationcredential",
        "application_credential_id": os.environ["OS_APPLICATION_CREDENTIAL_ID"],
        "application_credential_secret": os.environ["OS_APPLICATION_CREDENTIAL_SECRET"],
        "region_name": "CHI@TACC",
    }
)
d = pathlib.Path.home() / "AutoGym" / "entire_codebase" / "data"
for f in ["RAW_recipes.csv", "RAW_interactions.csv"]:
    _, c = conn.get_object("proj04-sparky-raw-data", f)
    (d / f).write_bytes(c)
    print(f"Downloaded {f} ({len(c)/1e6:.1f} MB)")
conn.close()
EOF

# 4. ML system — build, run data pipeline, train, serve
cd ~/sparky-ml
make build
make run-infra
make data          # feature engineering on the downloaded CSVs
make train-direct
# Promote via MLflow REST API (retrain-api not running yet at this point)
curl -s -X POST http://localhost:5000/api/2.0/mlflow/model-versions/transition-stage \
  -H "Content-Type: application/json" \
  -d '{"name": "sparky-ranker", "version": "1", "stage": "Production", "archive_existing_versions": false}' \
  | python3 -m json.tool
make run-serving
make run-monitoring

# 4. SparkyFitness
cd ~/SparkyFitness/docker
docker compose -f docker-compose.prod.yml -f docker-compose.sparky-build.yml --env-file .env up -d --build

# 5. Connect networks
docker network connect sparky-ml_sparky-net sparkyfitness-server

# 6. Open browser at http://NEW_IP:3004
```

---

## Common ML Operations

```bash
cd ~/sparky-ml

make train            # trigger async retraining via webhook API
make retrain-status   # check if a retraining job is running
make promote          # promote latest Staging model to Production
make rollback         # roll back to previous Production model
make logs-serving     # live serving logs
make smoke-test       # health check + sample prediction
make fairness         # run per-group NDCG fairness check
make drift-check      # run one-off KS drift check
make stop             # stop all ML containers
make clean            # destroy all containers and volumes (destructive)
```

---

## Troubleshooting

### Recommendations panel does not appear in SparkyFitness

```bash
# 1. Check ML API is running
curl http://localhost:8000/health

# 2. Check network link between containers
docker network inspect sparky-ml_sparky-net | grep sparkyfitness

# 3. If network link is missing
docker network connect sparky-ml_sparky-net sparkyfitness-server

# 4. Check SparkyFitness server logs for ML call errors
docker logs sparkyfitness-server --tail 50 | grep -i recommend
```

### Swift download fails during `make data`

```bash
# Check credentials in .env
grep OS_APPLICATION ~/sparky-ml/.env

# Make sure OS_AUTH_URL points to CHI@TACC, not KVM@TACC
# It must be: https://chi.tacc.chameleoncloud.org:5000/v3

# Test connectivity manually
curl -s https://chi.tacc.chameleoncloud.org:5000/v3 | python3 -m json.tool
```

If Swift is unreachable, download from Kaggle directly on the VM:
```bash
pip3 install kaggle
mkdir -p ~/.kaggle
echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_API_KEY"}' \
  > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
cd ~/AutoGym/entire_codebase/data
kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions
unzip food-com-recipes-and-user-interactions.zip
mv recipes.csv RAW_recipes.csv 2>/dev/null || true
mv interactions.csv RAW_interactions.csv 2>/dev/null || true
```

### `make train-direct` fails — NDCG below threshold

```bash
# Lower threshold temporarily for testing
echo "NDCG_THRESHOLD=0.0" >> ~/sparky-ml/.env
make train-direct
# Remove the override once real training succeeds
```

### MLflow not accessible

```bash
docker logs sparky-mlflow --tail 20
# Check if port 5000 is taken
sudo lsof -i :5000
```

### Out of disk space

```bash
docker system df
docker system prune -f       # remove stopped containers + dangling images
docker builder prune -f      # remove build cache
```

### SparkyFitness database not initialising

```bash
docker logs sparkyfitness-db --tail 30
# If there are permission errors, reset the volume:
cd ~/SparkyFitness/docker
docker compose -f docker-compose.prod.yml -f docker-compose.sparky-build.yml --env-file .env down -v
docker compose -f docker-compose.prod.yml -f docker-compose.sparky-build.yml --env-file .env up -d --build
```

### `sparkyfitness-server` crash-loops with SyntaxError on recommendationRoutes

Symptom — `docker logs sparkyfitness-server` shows:
```
SyntaxError: The requested module './routes/recommendationRoutes.js'
  does not provide an export named 'default'
```

Cause: The upstream `CodeWithCJ/SparkyFitness` repo has broken class-based stub
files for the recommendation feature. The `cp` commands in Step 12 must overwrite
them with the correct function-based implementations.

Fix:

```bash
INTEG=~/AutoGym/entire_codebase/sparkyfitness-integration
SF=~/SparkyFitness

# Overwrite the broken upstream stubs with correct versions
cp $INTEG/SparkyFitnessServer/routes/recommendationRoutes.ts     $SF/SparkyFitnessServer/routes/
cp $INTEG/SparkyFitnessServer/services/recommendationService.ts  $SF/SparkyFitnessServer/services/
cp $INTEG/SparkyFitnessServer/models/recommendationRepository.ts $SF/SparkyFitnessServer/models/

# Verify last line of each file now shows export default
tail -1 $SF/SparkyFitnessServer/routes/recommendationRoutes.ts
tail -1 $SF/SparkyFitnessServer/services/recommendationService.ts
grep "export default" $SF/SparkyFitnessServer/models/recommendationRepository.ts

# Rebuild and restart
cd ~/SparkyFitness/docker
docker compose -f docker-compose.prod.yml -f docker-compose.sparky-build.yml \
  --env-file .env build --no-cache sparkyfitness-server
docker compose -f docker-compose.prod.yml -f docker-compose.sparky-build.yml \
  --env-file .env up -d sparkyfitness-server
sleep 10 && docker ps | grep sparkyfitness-server
```

---

### Login shows "almost there" but never completes

Cause: `SPARKY_FITNESS_FRONTEND_URL` in `.env` does not exactly match the floating
IP used in the browser. The auth system rejects the cookie/redirect.

Fix:

```bash
# Check what is currently set
grep FRONTEND_URL ~/SparkyFitness/docker/.env

# Compare with the IP you are actually using in the browser
# If they differ (e.g., .126 vs .226), fix it:
sed -i 's/OLD_IP/NEW_IP/g' ~/SparkyFitness/docker/.env

# Restart server to pick up the new value
cd ~/SparkyFitness/docker
docker compose -f docker-compose.prod.yml -f docker-compose.sparky-build.yml \
  --env-file .env up -d sparkyfitness-server

# Confirm the value the server actually loaded
docker logs sparkyfitness-server 2>&1 | grep FRONTEND_URL
```

---

## GitHub Actions CI/CD (Optional)

To enable automated weekly retraining, add these secrets in
**GitHub → repo → Settings → Secrets → Actions → New repository secret**:

| Secret | Value |
|---|---|
| `MLFLOW_TRACKING_URI` | `http://YOUR_IP:5000` |
| `POSTGRES_PASSWORD` | your ML DB password from Step 14 |
| `SERVING_URL` | `http://YOUR_IP:8000` |
| `OS_AUTH_URL` | `https://chi.tacc.chameleoncloud.org:5000/v3` |
| `OS_APPLICATION_CREDENTIAL_ID` | from Step 2 |
| `OS_APPLICATION_CREDENTIAL_SECRET` | from Step 2 |

The pipeline at `.github/workflows/retrain.yml` runs every Sunday at 2 AM UTC,
runs quality + fairness gates, auto-promotes if NDCG ≥ 0.55, and
auto-rolls back if the smoke test fails.
