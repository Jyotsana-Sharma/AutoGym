#!/usr/bin/env bash
# setup-sparkyfitness.sh
# Prepares the SparkyFitness submodule for use with the ML recommendation system.
# Run this once after cloning, before docker compose up.
#
# Usage:
#   bash scripts/setup-sparkyfitness.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPARKY_DIR="$REPO_ROOT/SparkyFitness"
INTEGRATION_DIR="$REPO_ROOT/sparkyfitness-integration"

echo ""
echo "========================================"
echo " SparkyFitness Integration Setup"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1 — Initialise the submodule
# ---------------------------------------------------------------------------
echo "[1/5] Initialising SparkyFitness submodule..."

if [ ! -f "$SPARKY_DIR/package.json" ]; then
    git -C "$REPO_ROOT" submodule update --init --recursive
    echo "      Submodule initialised."
else
    echo "      Submodule already present, pulling latest..."
    git -C "$SPARKY_DIR" pull origin main 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Step 2 — Create SparkyFitness .env from template
# ---------------------------------------------------------------------------
echo "[2/5] Setting up SparkyFitness environment variables..."

if [ ! -f "$SPARKY_DIR/.env" ]; then
    cp "$REPO_ROOT/.env.sparky.example" "$SPARKY_DIR/.env"
    echo "      Created SparkyFitness/.env from template."
    echo "      IMPORTANT: Review SparkyFitness/.env and set strong secrets for production."
else
    echo "      SparkyFitness/.env already exists — skipping."
fi

# ---------------------------------------------------------------------------
# Step 3 — Copy ML integration files into SparkyFitness
# ---------------------------------------------------------------------------
echo "[3/5] Copying ML recommendation integration files..."

# Backend
mkdir -p "$SPARKY_DIR/SparkyFitnessServer/schemas"
mkdir -p "$SPARKY_DIR/SparkyFitnessServer/models"
mkdir -p "$SPARKY_DIR/SparkyFitnessServer/services"
mkdir -p "$SPARKY_DIR/SparkyFitnessServer/routes"

cp "$INTEGRATION_DIR/SparkyFitnessServer/schemas/recommendationSchemas.ts" \
   "$SPARKY_DIR/SparkyFitnessServer/schemas/"

cp "$INTEGRATION_DIR/SparkyFitnessServer/models/recommendationRepository.ts" \
   "$SPARKY_DIR/SparkyFitnessServer/models/"

cp "$INTEGRATION_DIR/SparkyFitnessServer/services/recommendationService.ts" \
   "$SPARKY_DIR/SparkyFitnessServer/services/"

cp "$INTEGRATION_DIR/SparkyFitnessServer/routes/recommendationRoutes.ts" \
   "$SPARKY_DIR/SparkyFitnessServer/routes/"

# Frontend
mkdir -p "$SPARKY_DIR/SparkyFitnessFrontend/src/api"
mkdir -p "$SPARKY_DIR/SparkyFitnessFrontend/src/hooks/Foods"
mkdir -p "$SPARKY_DIR/SparkyFitnessFrontend/src/pages/Foods"

cp "$INTEGRATION_DIR/SparkyFitnessFrontend/src/api/recommendations.ts" \
   "$SPARKY_DIR/SparkyFitnessFrontend/src/api/"

cp "$INTEGRATION_DIR/SparkyFitnessFrontend/src/hooks/Foods/useRecommendations.ts" \
   "$SPARKY_DIR/SparkyFitnessFrontend/src/hooks/Foods/"

cp "$INTEGRATION_DIR/SparkyFitnessFrontend/src/pages/Foods/RecipeRecommendations.tsx" \
   "$SPARKY_DIR/SparkyFitnessFrontend/src/pages/Foods/"

echo "      Integration files copied."

# ---------------------------------------------------------------------------
# Step 4 — Patch SparkyFitnessServer.ts to mount the recommendation route
# ---------------------------------------------------------------------------
echo "[4/5] Patching SparkyFitnessServer.ts to register recommendation route..."

SERVER_FILE="$SPARKY_DIR/SparkyFitnessServer/SparkyFitnessServer.ts"

if [ ! -f "$SERVER_FILE" ]; then
    echo "      WARNING: $SERVER_FILE not found — skipping auto-patch."
    echo "      Manually add the following to SparkyFitnessServer.ts:"
    echo "        import { createRecommendationRouter } from './routes/recommendationRoutes';"
    echo "        app.use('/api', createRecommendationRouter(pool));"
else
    if grep -q "createRecommendationRouter" "$SERVER_FILE"; then
        echo "      Route already registered — skipping."
    else
        python3 - "$SERVER_FILE" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path, 'r') as f:
    content = f.read()

# 1. Insert import after the last existing router import line
import_line = 'import { createRecommendationRouter } from "./routes/recommendationRoutes";'
# Find the last "import { create...Router }" line and append after it
content = re.sub(
    r'(import \{ create\w+Router \} from "\./routes/\w+";\n)(?!import \{ create)',
    lambda m: m.group(0) + import_line + '\n',
    content,
    count=1
)

# 2. Insert route mount after the last existing app.use("/api", ...Router(...)) line
mount_line = '  app.use("/api", createRecommendationRouter(pool));'
content = re.sub(
    r'(  app\.use\("/api",\s+create\w+Router\([^)]*\)\);)',
    lambda m: m.group(0) + '\n' + mount_line,
    content,
    count=1
)

with open(path, 'w') as f:
    f.write(content)

print("      SparkyFitnessServer.ts patched successfully.")
PYEOF
    fi
fi

# ---------------------------------------------------------------------------
# Step 5 — Run the DB migration (requires sparkyfitness-db to be running)
# ---------------------------------------------------------------------------
echo "[5/5] Database migration..."
echo "      The recommendation tables will be created automatically when"
echo "      sparkyfitness-db starts via the init SQL in the integration files."
echo "      (Run: docker exec sparkyfitness-db psql -U sparky -d sparkyfitness_db"
echo "       -f /migration.sql  if you need to apply it manually.)"

echo ""
echo "========================================"
echo " Setup complete!"
echo " Next: docker compose --profile pipeline up -d"
echo "========================================"
echo ""
