"""Apply the SparkyFitness recommendation integration inside the app tree.

This script is intentionally idempotent.  The docker-compose setup service runs
it before the frontend/server containers start, and developers can run it
manually when refreshing the upstream SparkyFitness checkout.
"""

from __future__ import annotations

import shutil
import os
from pathlib import Path


APP = Path(os.environ.get("APP_DIR", "/app"))
INTEGRATION = Path(os.environ.get("INTEGRATION_DIR", "/integration"))
ENV_EXAMPLE = Path(os.environ.get("ENV_EXAMPLE", "/env.example"))


COPIES = [
    ("SparkyFitnessServer/schemas/recommendationSchemas.ts", "SparkyFitnessServer/schemas/recommendationSchemas.ts"),
    ("SparkyFitnessServer/models/recommendationRepository.ts", "SparkyFitnessServer/models/recommendationRepository.ts"),
    ("SparkyFitnessServer/services/recommendationService.ts", "SparkyFitnessServer/services/recommendationService.ts"),
    ("SparkyFitnessServer/routes/recommendationRoutes.ts", "SparkyFitnessServer/routes/recommendationRoutes.ts"),
    ("SparkyFitnessFrontend/src/api/recommendations.ts", "SparkyFitnessFrontend/src/api/recommendations.ts"),
    ("SparkyFitnessFrontend/src/hooks/Foods/useRecommendations.ts", "SparkyFitnessFrontend/src/hooks/Foods/useRecommendations.ts"),
    ("SparkyFitnessFrontend/src/pages/Foods/RecipeRecommendations.tsx", "SparkyFitnessFrontend/src/pages/Foods/RecipeRecommendations.tsx"),
]


def copy_files() -> None:
    print("[setup] Copying ML integration files...")
    for src_rel, dst_rel in COPIES:
        src = INTEGRATION / src_rel
        dst = APP / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    env_path = APP / ".env"
    if not env_path.exists() and ENV_EXAMPLE.exists():
        shutil.copy2(ENV_EXAMPLE, env_path)
        print("[setup] Created /app/.env from template")


def patch_server() -> None:
    path = APP / "SparkyFitnessServer/SparkyFitnessServer.ts"
    if not path.exists():
        print("[setup] WARNING: SparkyFitnessServer.ts not found; skipping route patch")
        return

    content = path.read_text()
    changed = False
    import_line = "import recommendationRoutes from './routes/recommendationRoutes.js';"
    mount_line = "app.use('/api/recommendations', recommendationRoutes);"

    if import_line not in content:
        anchor = "import foodRoutes from './routes/foodRoutes.js';"
        content = content.replace(anchor, f"{anchor}\n{import_line}")
        changed = True

    if mount_line not in content:
        anchor = "app.use('/api/foods', foodRoutes);"
        content = content.replace(anchor, f"{anchor}\n{mount_line}")
        changed = True

    if changed:
        path.write_text(content)
        print("[setup] Registered /api/recommendations route")
    else:
        print("[setup] Recommendation route already registered")


def patch_frontend() -> None:
    path = APP / "SparkyFitnessFrontend/src/pages/Foods/Foods.tsx"
    if not path.exists():
        print("[setup] WARNING: Foods.tsx not found; skipping component patch")
        return

    content = path.read_text()
    changed = False
    import_line = "import RecipeRecommendations from './RecipeRecommendations';"

    if import_line not in content:
        anchor = "import MealManagement from './MealManagement';"
        content = content.replace(anchor, f"{import_line}\n{anchor}")
        changed = True

    component_block = (
        "      {/* ML Recommendations Section */}\n"
        "      <Card>\n"
        "        <CardContent className=\"pt-6\">\n"
        "          <RecipeRecommendations limit={6} />\n"
        "        </CardContent>\n"
        "      </Card>\n\n"
    )
    if "<RecipeRecommendations" not in content:
        anchor = "      {/* Meal Management Section */}\n"
        content = content.replace(anchor, component_block + anchor)
        changed = True

    if changed:
        path.write_text(content)
        print("[setup] Added RecipeRecommendations to Foods page")
    else:
        print("[setup] RecipeRecommendations already present")


def main() -> None:
    copy_files()
    patch_server()
    patch_frontend()
    print("[setup] Done - SparkyFitness is ready to start")


if __name__ == "__main__":
    main()
