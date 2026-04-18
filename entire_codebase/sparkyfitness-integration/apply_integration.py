"""Apply the SparkyFitness recommendation integration inside the app tree.

This script is intentionally idempotent. Docker builds run it inside the image
so the upstream SparkyFitness checkout can stay clean. Developers can still run
the manual setup profile when they intentionally want a local host checkout
patched for debugging.
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


def patch_food_search_provider_selection() -> None:
    path = APP / "SparkyFitnessFrontend/src/components/FoodSearch/FoodSearch.tsx"
    if not path.exists():
        print("[setup] WARNING: FoodSearch.tsx not found; skipping provider patch")
        return

    content = path.read_text()
    changed = False

    supported_types = (
        "const SUPPORTED_ONLINE_FOOD_PROVIDER_TYPES = new Set([\n"
        "  'openfoodfacts',\n"
        "  'nutritionix',\n"
        "  'fatsecret',\n"
        "  'usda',\n"
        "  'mealie',\n"
        "  'tandoor',\n"
        "]);\n\n"
    )
    if "SUPPORTED_ONLINE_FOOD_PROVIDER_TYPES" not in content:
        anchor = "type FoodDataForBackend = Omit<CSVData, 'id'>;\n\n"
        content = content.replace(anchor, anchor + supported_types)
        changed = True

    old_selected_provider = (
        "  const selectedFoodDataProvider =\n"
        "    manualProviderId ||\n"
        "    defaultFoodDataProviderId ||\n"
        "    foodDataProviders[0]?.id ||\n"
        "    null;\n"
    )
    new_selected_provider = (
        "  const onlineFoodProviders = foodDataProviders.filter(\n"
        "    (provider) =>\n"
        "      provider.is_active &&\n"
        "      getProviderCategory(provider).includes('food') &&\n"
        "      SUPPORTED_ONLINE_FOOD_PROVIDER_TYPES.has(provider.provider_type)\n"
        "  );\n\n"
        "  const selectedFoodDataProvider =\n"
        "    (manualProviderId &&\n"
        "    onlineFoodProviders.some((provider) => provider.id === manualProviderId)\n"
        "      ? manualProviderId\n"
        "      : null) ||\n"
        "    (defaultFoodDataProviderId &&\n"
        "    onlineFoodProviders.some(\n"
        "      (provider) => provider.id === defaultFoodDataProviderId\n"
        "    )\n"
        "      ? defaultFoodDataProviderId\n"
        "      : null) ||\n"
        "    onlineFoodProviders[0]?.id ||\n"
        "    null;\n"
    )
    if old_selected_provider in content:
        content = content.replace(old_selected_provider, new_selected_provider)
        changed = True

    old_toast = (
        "          description: 'Provider not supported',\n"
    )
    new_toast = (
        "          description:\n"
        "            onlineFoodProviders.length === 0\n"
        "              ? 'No active online food provider is configured.'\n"
        "              : 'Provider not supported for food search.',\n"
    )
    if old_toast in content:
        content = content.replace(old_toast, new_toast)
        changed = True

    old_select_content = (
        "            <SelectContent>\n"
        "              {foodDataProviders\n"
        "                .filter(\n"
        "                  (provider) =>\n"
        "                    getProviderCategory(provider).includes('food') &&\n"
        "                    provider.is_active\n"
        "                )\n"
        "                .map((provider) => (\n"
        "                  <SelectItem key={provider.id} value={provider.id}>\n"
        "                    {' '}\n"
        "                    {provider.provider_name}{' '}\n"
        "                  </SelectItem>\n"
        "                ))}\n"
        "            </SelectContent>\n"
    )
    new_select_content = (
        "            <SelectContent>\n"
        "              {onlineFoodProviders.map((provider) => (\n"
        "                <SelectItem key={provider.id} value={provider.id}>\n"
        "                  {' '}\n"
        "                  {provider.provider_name}{' '}\n"
        "                </SelectItem>\n"
        "              ))}\n"
        "            </SelectContent>\n"
    )
    if old_select_content in content:
        content = content.replace(old_select_content, new_select_content)
        changed = True

    if changed:
        path.write_text(content)
        print("[setup] Restricted online food search to supported food providers")
    else:
        print("[setup] Online food provider selection already patched")


def main() -> None:
    copy_files()
    patch_server()
    patch_frontend()
    patch_food_search_provider_selection()
    print("[setup] Done - SparkyFitness is ready to start")


if __name__ == "__main__":
    main()
