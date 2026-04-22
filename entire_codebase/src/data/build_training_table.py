"""
build_training_table.py
=======================
Unified pipeline that:
  1. Loads RAW_recipes.csv + RAW_interactions.csv (Food.com)
  2. Enriches recipes  — exactly as in 01-data-preparation-foodcom.ipynb
       - avg_rating / n_reviews from interactions
       - filtering (minutes, n_reviews)
       - cuisine  via ETHNICITY_MAP
       - allergens via ALLERGEN_KEYWORDS
       - macro columns parsed from the nutrition list
  3. Builds interaction labels  (rating >= 4 → label = 1)
  4. Derives user features from cooking history
  5. Assembles the full training table:
       (user_id, recipe_id, full feature vector X, label y)
  6. Writes time-stratified train / val / test splits

Inputs  : RAW_recipes.csv, RAW_interactions.csv   (Kaggle Food.com dataset)
Outputs : enriched_recipes.csv
          training_table.csv
          train.csv  /  val.csv  /  test.csv
"""

import ast
import json
import pickle
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG  — change paths here if needed
# ─────────────────────────────────────────────────────────────

RAW_RECIPES_PATH      = "RAW_recipes.csv"
RAW_INTERACTIONS_PATH = "RAW_interactions.csv"
ENRICHED_RECIPES_PATH = "enriched_recipes.csv"
TRAINING_TABLE_PATH   = "training_table.csv"

POSITIVE_RATING_THRESHOLD = 4   # ratings >= this → label 1
MIN_USER_INTERACTIONS     = 5   # drop users with fewer interactions
HISTORY_PCA_COMPONENTS    = 6   # PCA dims for cooking history embedding
EMBEDDING_COMPONENTS      = 8   # low-dim ingredient/text embedding size
HISTORY_LOOKBACK_RECIPES  = 10  # recent positive interactions used for history
MINUTES_QUANTILE_UPPER    = 0.99
EMBEDDING_ARTIFACT_DIR    = "embedding_artifacts"


# ─────────────────────────────────────────────────────────────
# PART 1 — RECIPE ENRICHMENT  (from notebook)
# ─────────────────────────────────────────────────────────────

# ── 1a. Cuisine mapping (from notebook cell 53) ───────────────
ETHNICITY_MAP = {
    # Italian
    "italian": "italian", "sicilian": "italian", "tuscan": "italian",
    # American / North American
    "american": "american", "north-american": "american",
    "southern-united-states": "american", "southwestern-united-states": "american",
    "northeastern-united-states": "american", "midwestern": "american",
    "californian": "american", "pacific-northwest": "american",
    "native-american": "american",
    # Mexican / Latin
    "mexican": "mexican", "tex-mex": "mexican",
    "central-american": "latin_american", "brazilian": "latin_american",
    "argentine": "latin_american", "chilean": "latin_american",
    "peruvian": "latin_american", "ecuadorean": "latin_american",
    "venezuelan": "latin_american", "cuban": "latin_american",
    "caribbean": "latin_american", "baja": "latin_american",
    # European
    "european": "european", "greek": "greek", "french": "french",
    "spanish": "spanish", "german": "german", "scandinavian": "scandinavian",
    "irish": "irish", "english": "british", "scottish": "british",
    "swiss": "european", "dutch": "european", "portuguese": "european",
    "polish": "european", "hungarian": "european", "russian": "european",
    "austrian": "european", "belgian": "european", "czech": "european",
    "finnish": "european",
    # Asian
    "asian": "asian", "indian": "indian", "chinese": "chinese",
    "szechuan": "chinese", "cantonese": "chinese", "japanese": "japanese",
    "korean": "korean", "thai": "thai", "vietnamese": "asian",
    "indonesian": "asian", "malaysian": "asian", "filipino": "asian",
    "laotian": "asian", "mongolian": "asian", "cambodian": "asian",
    "pakistani": "asian", "hunan": "chinese", "beijing": "chinese",
    # Middle Eastern / North African
    "middle-eastern": "middle_eastern", "lebanese": "middle_eastern",
    "moroccan": "middle_eastern", "turkish": "middle_eastern",
    "saudi-arabian": "middle_eastern", "iranian-persian": "middle_eastern",
    "egyptian": "middle_eastern", "palestinian": "middle_eastern",
    # African
    "african": "african", "south-african": "african", "ethiopian": "african",
    "nigerian": "african", "angolan": "african", "somalian": "african",
    # Pacific
    "hawaiian": "pacific", "polynesian": "pacific",
    "micro-melanesia": "pacific", "new-zealand": "pacific",
    "australian": "pacific",
}

# ── 1b. Allergen keyword map (from notebook cell 55) ─────────
ALLERGEN_KEYWORDS = {
    "milk":      ["milk", "butter", "cream", "cheese", "yogurt", "mozzarella",
                  "parmesan", "gouda", "cheddar", "buttermilk", "whey"],
    "egg":       ["egg", "egg yolk", "egg white", "mayonnaise"],
    "nuts":      ["almond", "walnut", "hazelnut", "cashew", "pistachio",
                  "pecan", "macadamia", "brazil nut"],
    "peanut":    ["peanut", "peanut butter"],
    "fish":      ["salmon", "tuna", "cod", "trout", "halibut", "anchovy", "sardine"],
    "shellfish": ["shrimp", "prawn", "lobster", "crab", "scallop",
                  "mussel", "clam", "oyster"],
    "wheat":     ["wheat", "flour", "bread", "pasta", "noodles", "semolina"],
    "soy":       ["soy", "soybean", "tofu", "tempeh", "edamame", "soy sauce"],
    "sesame":    ["sesame", "tahini"],
}

# ── Helpers ───────────────────────────────────────────────────
def parse_tags(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []

def parse_ingredients(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []

def parse_nutrition(x):
    """Return nutrition list: [calories, fat_pdv, sugar_pdv,
    sodium_pdv, protein_pdv, sat_fat_pdv, carb_pdv]."""
    try:
        vals = ast.literal_eval(x) if isinstance(x, str) else x
    except (ValueError, SyntaxError):
        return [np.nan] * 7
    if not isinstance(vals, (list, tuple)) or len(vals) < 7:
        return [np.nan] * 7
    return list(vals)


def normalize_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_cuisine(tags_raw):
    """Map tags → cuisine using ETHNICITY_MAP (first match wins)."""
    tags = parse_tags(tags_raw)
    for t in tags:
        t_lower = t.lower()
        if t_lower in ETHNICITY_MAP:
            return ETHNICITY_MAP[t_lower]
    return "unknown"

def detect_allergens(ingredients_raw):
    """Return list of allergens detected in ingredient list."""
    ingredients = parse_ingredients(ingredients_raw)
    ingredients_lower = [ing.lower() for ing in ingredients]
    found = set()
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        for ing in ingredients_lower:
            if any(kw in ing for kw in keywords):
                found.add(allergen)
                break
    return list(found)


def build_ingredient_document(ingredients_raw):
    ingredients = parse_ingredients(ingredients_raw)
    return " ".join(filter(None, (normalize_text(ing) for ing in ingredients)))


def build_text_document(name, description):
    return " ".join(
        filter(None, [normalize_text(name), normalize_text(description)])
    ).strip()


def pad_embedding_matrix(matrix, target_components=EMBEDDING_COMPONENTS):
    if matrix.shape[1] >= target_components:
        return matrix[:, :target_components]
    pad_width = target_components - matrix.shape[1]
    return np.pad(matrix, ((0, 0), (0, pad_width)), mode="constant")


def fit_embedding_pipeline(documents, *, max_features):
    docs = pd.Series(documents).fillna("").astype(str)
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_features,
        )
        tfidf = vectorizer.fit_transform(docs)
    except ValueError:
        return {
            "vectorizer": None,
            "svd": None,
            "output_dim": 0,
            "target_dim": EMBEDDING_COMPONENTS,
        }, np.zeros((len(docs), EMBEDDING_COMPONENTS), dtype=float)

    max_rank = min(EMBEDDING_COMPONENTS, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if max_rank < 1:
        return {
            "vectorizer": vectorizer,
            "svd": None,
            "output_dim": 0,
            "target_dim": EMBEDDING_COMPONENTS,
        }, np.zeros((len(docs), EMBEDDING_COMPONENTS), dtype=float)

    svd = TruncatedSVD(n_components=max_rank, random_state=42)
    matrix = svd.fit_transform(tfidf)
    return {
        "vectorizer": vectorizer,
        "svd": svd,
        "output_dim": max_rank,
        "target_dim": EMBEDDING_COMPONENTS,
    }, pad_embedding_matrix(matrix)


def save_embedding_artifacts(ingredient_pipeline, text_pipeline):
    artifact_dir = Path(EMBEDDING_ARTIFACT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "ingredient_pipeline.pkl", "wb") as handle:
        pickle.dump(ingredient_pipeline, handle)
    with open(artifact_dir / "text_pipeline.pkl", "wb") as handle:
        pickle.dump(text_pipeline, handle)
    metadata = {
        "ingredient_dim": ingredient_pipeline.get("output_dim", 0),
        "text_dim": text_pipeline.get("output_dim", 0),
        "target_dim": EMBEDDING_COMPONENTS,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def add_recipe_embedding_features(recipes):
    recipes = recipes.copy()
    ingredient_docs = recipes["ingredients"].apply(build_ingredient_document)
    text_docs = recipes.apply(
        lambda row: build_text_document(row.get("name"), row.get("description")),
        axis=1,
    )

    ingredient_pipeline, ingredient_matrix = fit_embedding_pipeline(
        ingredient_docs, max_features=5000
    )
    text_pipeline, text_matrix = fit_embedding_pipeline(
        text_docs, max_features=10000
    )

    for idx in range(EMBEDDING_COMPONENTS):
        recipes[f"ingredient_emb_{idx+1}"] = ingredient_matrix[:, idx]
        recipes[f"text_emb_{idx+1}"] = text_matrix[:, idx]

    save_embedding_artifacts(ingredient_pipeline, text_pipeline)
    print(
        "  Added recipe embedding features:"
        f" ingredient_dim={ingredient_pipeline.get('output_dim', 0)}"
        f" text_dim={text_pipeline.get('output_dim', 0)}"
    )
    return recipes


def add_history_vector_columns(cooked, user_features, recipe_prefix, history_prefix):
    recipe_cols = [
        f"{recipe_prefix}_{idx}" for idx in range(1, EMBEDDING_COMPONENTS + 1)
        if f"{recipe_prefix}_{idx}" in cooked.columns
    ]
    if not recipe_cols:
        for idx in range(1, EMBEDDING_COMPONENTS + 1):
            user_features[f"{history_prefix}_{idx}"] = 0.0
        return user_features

    history_vectors = cooked[["user_id", *recipe_cols]].groupby("user_id", as_index=False).mean()
    history_vectors = history_vectors.rename(
        columns={
            f"{recipe_prefix}_{idx}": f"{history_prefix}_{idx}"
            for idx in range(1, EMBEDDING_COMPONENTS + 1)
            if f"{recipe_prefix}_{idx}" in history_vectors.columns
        }
    )
    user_features = user_features.merge(history_vectors, on="user_id", how="left")
    for idx in range(1, EMBEDDING_COMPONENTS + 1):
        col = f"{history_prefix}_{idx}"
        if col not in user_features.columns:
            user_features[col] = 0.0
        else:
            user_features[col] = (
                pd.to_numeric(user_features[col], errors="coerce")
                .fillna(0.0)
                .astype(float)
            )
    return user_features


def rowwise_cosine(frame, left_cols, right_cols):
    left = frame[left_cols].fillna(0.0).to_numpy(dtype=float)
    right = frame[right_cols].fillna(0.0).to_numpy(dtype=float)
    numerators = np.sum(left * right, axis=1)
    denominators = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerators,
        denominators,
        out=np.zeros(len(frame), dtype=float),
        where=denominators > 0,
    )


def add_similarity_features(training):
    training = training.copy()

    ingredient_cols = [f"ingredient_emb_{idx}" for idx in range(1, EMBEDDING_COMPONENTS + 1)]
    history_ingredient_cols = [
        f"history_ingredient_emb_{idx}" for idx in range(1, EMBEDDING_COMPONENTS + 1)
    ]
    text_cols = [f"text_emb_{idx}" for idx in range(1, EMBEDDING_COMPONENTS + 1)]
    history_text_cols = [f"history_text_emb_{idx}" for idx in range(1, EMBEDDING_COMPONENTS + 1)]

    for col in ingredient_cols + history_ingredient_cols + text_cols + history_text_cols:
        if col not in training.columns:
            training[col] = 0.0

    training["ingredient_embedding_cosine"] = rowwise_cosine(
        training, ingredient_cols, history_ingredient_cols
    )
    training["text_embedding_cosine"] = rowwise_cosine(
        training, text_cols, history_text_cols
    )
    macro_left = ["calories", "protein_g", "carbohydrate_g", "total_fat_g"]
    macro_right = [
        "daily_calorie_target",
        "protein_target_g",
        "carbs_target_g",
        "fat_target_g",
    ]
    for col in macro_left + macro_right:
        if col not in training.columns:
            training[col] = 0.0
    training["history_macro_similarity"] = rowwise_cosine(training, macro_left, macro_right)
    return training


def enrich_recipes(df_recipes, df_interact):
    """
    Reproduce every transformation from the notebook and return
    the fully enriched recipe DataFrame.
    """
    print("\n── STEP 1: Recipe enrichment ──────────────────────────")

    # Rename 'id' → 'recipe_id' to match interactions
    df_recipes = df_recipes.rename(columns={"id": "recipe_id"})

    cols = ["recipe_id", "name", "minutes", "tags", "ingredients",
            "n_ingredients", "n_steps", "nutrition", "description"]
    df = df_recipes[cols].copy()

    # ── avg_rating / n_reviews from interactions ────────────────────────────
    grouped = (
        df_interact
        .groupby("recipe_id")["rating"]
        .agg(avg_rating="mean", n_reviews="count")
        .reset_index()
    )
    df = df.merge(grouped, on="recipe_id", how="left")
    df["avg_rating"] = df["avg_rating"].fillna(0)
    df["n_reviews"]  = df["n_reviews"].fillna(0).astype(int)

    # ── Filtering (notebook cells 49 & 51) ──────────────────────────────────
    df = df[~df["name"].isna()]
    df["description"] = df["description"].fillna("")

    upper = df["minutes"].quantile(MINUTES_QUANTILE_UPPER)
    df = df[(df["minutes"] > 0) & (df["minutes"] <= upper) & (df["n_reviews"] >= 3)]
    print(f"  After filtering: {len(df):,} recipes")

    # ── Cuisine (notebook cell 54) ──────────────────────────────────────────
    df["cuisine"] = df["tags"].apply(extract_cuisine)
    print(f"  Cuisine distribution (top 5):\n"
          f"{df['cuisine'].value_counts().head(5).to_string()}")

    # ── Allergens (notebook cell 55) ────────────────────────────────────────
    df["allergens"] = df["ingredients"].apply(detect_allergens)

    # ── Nutrition → individual macro columns (notebook cell 57) ────────────
    # nutrition = [calories, total_fat(PDV), sugar(PDV), sodium(PDV),
    #              protein(PDV), saturated_fat(PDV), carbohydrate(PDV)]
    nutrition_parsed = df["nutrition"].apply(parse_nutrition)
    df["calories"]      = nutrition_parsed.apply(lambda v: v[0])
    df["total_fat"]     = nutrition_parsed.apply(lambda v: v[1])  # PDV %
    df["sugar"]         = nutrition_parsed.apply(lambda v: v[2])  # PDV %
    df["sodium"]        = nutrition_parsed.apply(lambda v: v[3])  # PDV %
    df["protein"]       = nutrition_parsed.apply(lambda v: v[4])  # PDV %
    df["saturated_fat"] = nutrition_parsed.apply(lambda v: v[5])  # PDV %
    df["carbohydrate"]  = nutrition_parsed.apply(lambda v: v[6])  # PDV %

    # Convert PDV % → absolute grams/mg (2000 kcal reference)
    PDV_TO_ABS = {
        "total_fat":     78,    # g
        "sugar":         50,    # g
        "sodium":        2300,  # mg
        "protein":       50,    # g
        "saturated_fat": 20,    # g
        "carbohydrate":  275,   # g
    }
    for col, factor in PDV_TO_ABS.items():
        df[col + "_g"] = (df[col] / 100) * factor

    df = df.drop(columns=["nutrition"])
    print(f"  Enriched columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────
# PART 2 — INTERACTION LABELS
# ─────────────────────────────────────────────────────────────

def build_labels(df_interact):
    """rating >= POSITIVE_RATING_THRESHOLD → label 1, else 0."""
    print("\n── STEP 2: Building labels ────────────────────────────")
    df = df_interact.copy()
    df["label"] = (df["rating"] >= POSITIVE_RATING_THRESHOLD).astype(int)
    pos = df["label"].sum()
    neg = len(df) - pos
    print(f"  Positive (label=1): {pos:,}  |  Negative (label=0): {neg:,}")
    return df


# ─────────────────────────────────────────────────────────────
# PART 3 — USER FEATURE DERIVATION
# ─────────────────────────────────────────────────────────────

def derive_user_features(interactions, recipes):
    """
    Derive user-level features from Food.com interaction history.
    Only positive (cooked/highly-rated) interactions are used so
    features reflect genuine user preferences.
    """
    print("\n── STEP 3: Deriving user features ─────────────────────")

    recipe_cols = [c for c in
                   ["recipe_id", "calories", "protein_g", "carbohydrate_g",
                    "total_fat_g", "cuisine", "tags"]
                   if c in recipes.columns]
    recipe_cols.extend(
        [
            c for c in recipes.columns
            if c.startswith("ingredient_emb_") or c.startswith("text_emb_")
        ]
    )

    history = interactions.merge(
        recipes[recipe_cols], on="recipe_id", how="left"
    )
    cooked = history[history["label"] == 1]
    if "date" in cooked.columns:
        cooked = cooked.copy()
        cooked["date"] = pd.to_datetime(cooked["date"], errors="coerce")
        cooked = (
            cooked.sort_values("date")
            .groupby("user_id", group_keys=False)
            .tail(HISTORY_LOOKBACK_RECIPES)
        )

    # ── Macro targets: average of what the user cooks ─────────────────────
    agg_kwargs = {}
    for src, tgt in [("calories", "daily_calorie_target"),
                     ("protein_g", "protein_target_g"),
                     ("carbohydrate_g", "carbs_target_g"),
                     ("total_fat_g", "fat_target_g")]:
        if src in cooked.columns:
            agg_kwargs[tgt] = (src, "mean")

    agg = cooked.groupby("user_id").agg(**agg_kwargs).reset_index()

    # ── Diet flags: 1 if ≥80% of cooked recipes carry the tag ────────────
    DIET_TAG_MAP = {
        "user_vegetarian":  "vegetarian",
        "user_vegan":       "vegan",
        "user_gluten_free": "gluten-free",
        "user_dairy_free":  "dairy-free",
        "user_low_sodium":  "low-sodium",
        "user_low_fat":     "low-fat",
    }

    if "tags" in cooked.columns:
        diet_records = []
        for uid, grp in cooked.groupby("user_id"):
            row = {"user_id": uid}
            for col, tag in DIET_TAG_MAP.items():
                share = grp["tags"].apply(
                    lambda x: tag in [t.lower() for t in parse_tags(x)]
                ).mean()
                row[col] = int(share >= 0.8)
            diet_records.append(row)
        diet_flags = pd.DataFrame(diet_records)
        agg = agg.merge(diet_flags, on="user_id", how="left")

    # ── Cooking history embeddings via PCA on cuisine one-hots ────────────
    agg = add_history_embeddings(cooked, agg)
    agg = add_history_vector_columns(cooked, agg, "ingredient_emb", "history_ingredient_emb")
    agg = add_history_vector_columns(cooked, agg, "text_emb", "history_text_emb")

    # ── Drop users with too few interactions ──────────────────────────────
    user_counts = interactions.groupby("user_id").size()
    active = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    agg = agg[agg["user_id"].isin(active)]

    print(f"  User features built for {len(agg):,} users  "
          f"({agg.shape[1] - 1} features per user)")
    return agg


def add_history_embeddings(cooked, user_features):
    """
    Summarise each user's cuisine preference as a 6-dim PCA vector
    derived from the cuisine one-hot distribution of their cooked recipes.
    """
    if "cuisine" not in cooked.columns:
        return user_features

    cuisine_ohe = pd.get_dummies(cooked["cuisine"], prefix="cuisine")
    cuisine_ohe["user_id"] = cooked["user_id"].values
    user_cuisine = cuisine_ohe.groupby("user_id").mean().reset_index()

    feature_cols = [c for c in user_cuisine.columns if c.startswith("cuisine_")]
    if not feature_cols:
        return user_features

    n_components = min(HISTORY_PCA_COMPONENTS, len(feature_cols),
                       len(user_cuisine))
    pca = PCA(n_components=n_components)
    matrix = user_cuisine[feature_cols].fillna(0).values
    pca_result = pca.fit_transform(matrix)

    pc_cols = [f"history_pc{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_result, columns=pc_cols)
    pca_df["user_id"] = user_cuisine["user_id"].values

    user_features = user_features.merge(pca_df, on="user_id", how="left")
    for idx in range(1, HISTORY_PCA_COMPONENTS + 1):
        col = f"history_pc{idx}"
        if col not in user_features.columns:
            user_features[col] = 0.0
        else:
            user_features[col] = pd.to_numeric(user_features[col], errors="coerce").fillna(0.0)
    print(f"  Added {n_components} history PCA components")
    return user_features


# ─────────────────────────────────────────────────────────────
# PART 4 — ASSEMBLE TRAINING TABLE
# ─────────────────────────────────────────────────────────────

def assemble_training_table(interactions, recipes, user_features):
    """
    Final join:
      interactions  (user_id, recipe_id, date, rating, label)
        ↕ recipe features  (macros, cuisine, allergen flags, …)
        ↕ user features    (targets, diet flags, history PCA)
    → one row per (user, recipe) with full X + y.
    """
    print("\n── STEP 4: Assembling training table ──────────────────")

    # Explode allergens list → binary columns on the recipe side
    all_allergens = sorted(ALLERGEN_KEYWORDS.keys())
    for allergen in all_allergens:
        col = f"has_{allergen}"
        recipes[col] = recipes["allergens"].apply(
            lambda a: int(allergen in a) if isinstance(a, list) else 0
        )

    # Drop raw string columns before join
    drop_recipe_cols = {"tags", "ingredients", "description", "allergens"}
    recipe_feature_cols = [c for c in recipes.columns
                           if c not in drop_recipe_cols]

    training = (
        interactions[["user_id", "recipe_id", "date", "rating", "label"]]
        .merge(recipes[recipe_feature_cols], on="recipe_id", how="inner")
        .merge(user_features,                on="user_id",    how="inner")
    )
    training = add_similarity_features(training)

    print(f"  Shape         : {training.shape}")
    print(f"  Users         : {training['user_id'].nunique():,}")
    print(f"  Recipes       : {training['recipe_id'].nunique():,}")
    print(f"  Label balance : {training['label'].mean():.1%} positive")
    return training


# ─────────────────────────────────────────────────────────────
# PART 5 — TIME-BASED SPLIT
# ─────────────────────────────────────────────────────────────

def split_by_time(training, train_frac=0.80, val_frac=0.10):
    """
    Percentage-based temporal split — sorted by date to prevent leakage.
      Train : first 80% of interactions (chronologically)
      Val   : next 10%
      Test  : last 10%  (held-out, never seen during training)

    Using percentages instead of fixed day windows guarantees val and test
    always have enough rows regardless of the dataset's time range.
    """
    print("\n── STEP 5: Time-based split ───────────────────────────")
    training = training.sort_values("date").reset_index(drop=True)
    n = len(training)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    train = training.iloc[:train_end]
    val   = training.iloc[train_end:val_end]
    test  = training.iloc[val_end:]

    print(f"  Train : {len(train):,} rows  (dates: {training['date'].iloc[0]} → {training['date'].iloc[train_end-1]})")
    print(f"  Val   : {len(val):,} rows")
    print(f"  Test  : {len(test):,} rows  ← held-out, never seen in training")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SparkyFitness — Unified Data Pipeline")
    print("=" * 60)

    # Load raw files
    print(f"\nLoading {RAW_INTERACTIONS_PATH} ...")
    df_interact = pd.read_csv(RAW_INTERACTIONS_PATH)
    df_interact.columns = df_interact.columns.str.strip().str.lower()
    print(f"  Shape: {df_interact.shape}")

    print(f"Loading {RAW_RECIPES_PATH} ...")
    df_recipes = pd.read_csv(RAW_RECIPES_PATH)
    df_recipes.columns = df_recipes.columns.str.strip().str.lower()
    print(f"  Shape: {df_recipes.shape}")

    # Step 1: Enrich recipes (notebook logic)
    enriched = enrich_recipes(df_recipes, df_interact)
    enriched = add_recipe_embedding_features(enriched)
    enriched.to_csv(ENRICHED_RECIPES_PATH, index=False)
    print(f"\n  Saved → {ENRICHED_RECIPES_PATH}  ({len(enriched):,} rows)")

    # Step 2: Build interaction labels
    interactions = build_labels(df_interact)

    # Keep only interactions whose recipe_id survived enrichment filtering
    valid_ids = set(enriched["recipe_id"])
    interactions = interactions[interactions["recipe_id"].isin(valid_ids)]
    print(f"  Interactions after recipe filter: {len(interactions):,}")

    # Step 3: Derive user features
    user_features = derive_user_features(interactions, enriched)

    # Step 4: Assemble full training table
    training = assemble_training_table(interactions, enriched, user_features)
    training.to_csv(TRAINING_TABLE_PATH, index=False)
    print(f"\n  Saved → {TRAINING_TABLE_PATH}  ({len(training):,} rows)")

    # Step 5: Time-based split
    train, val, test = split_by_time(training)
    train.to_csv("train.csv", index=False)
    val.to_csv("val.csv",     index=False)
    test.to_csv("test.csv",   index=False)

    # Summary
    print("\n" + "=" * 60)
    print("✅  Pipeline complete. Output files:")
    print(f"   {ENRICHED_RECIPES_PATH:<30} {len(enriched):>8,} rows")
    print(f"   {TRAINING_TABLE_PATH:<30} {len(training):>8,} rows  "
          f"({training.shape[1]} columns)")
    print(f"   {'train.csv':<30} {len(train):>8,} rows")
    print(f"   {'val.csv':<30} {len(val):>8,} rows")
    print(f"   {'test.csv':<30} {len(test):>8,} rows")
    print("=" * 60)

    # Print a sample row to verify structure
    print("\nSample training row (key columns):")
    sample_cols = (
        ["user_id", "recipe_id", "label", "rating",
         "calories", "protein_g", "carbohydrate_g", "total_fat_g",
         "cuisine", "minutes"]
        + [c for c in training.columns
           if c.startswith("daily_") or c.startswith("user_")
           or c.startswith("history_")][:8]
    )
    sample_cols = [c for c in sample_cols if c in training.columns]
    print(training[sample_cols].head(2).to_string(index=False))


if __name__ == "__main__":
    main()
