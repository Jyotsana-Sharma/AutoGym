"""
batch_pipeline.py  —  Q2.6: Batch Training Data Pipeline
Compiles versioned training/evaluation datasets from production data.
Uses Chameleon Object Storage (OpenStack Swift) for versioned artifacts.
Runs non-interactively:  docker compose --profile batch run --rm batch-pipeline
"""
import argparse, hashlib, importlib.util, json, os, shutil, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

try:
    import swiftclient
    HAS_SWIFT = True
except ImportError:
    HAS_SWIFT = False

try: import psycopg2; HAS_PG=True
except ImportError: HAS_PG=False

PIPELINE_VERSION="1.0.0"; RAW_BUCKET="proj04-sparky-raw-data"; TRAINING_BUCKET="proj04-sparky-training-data"

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(8192),b""): h.update(c)
    return h.hexdigest()

def sha256_str(s): return hashlib.sha256(s.encode()).hexdigest()

def get_swift_conn():
    """Connect to Chameleon Object Storage using application credentials."""
    return swiftclient.Connection(
        authurl=os.environ.get("OS_AUTH_URL", "https://chi.tacc.chameleoncloud.org:5000/v3"),
        auth_version="3",
        os_options={
            "auth_type": "v3applicationcredential",
            "application_credential_id": os.environ.get("OS_APPLICATION_CREDENTIAL_ID", ""),
            "application_credential_secret": os.environ.get("OS_APPLICATION_CREDENTIAL_SECRET", ""),
            "region_name": os.environ.get("OS_REGION_NAME", "CHI@TACC"),
        },
    )

def ensure_container(conn, name):
    try: conn.head_container(name)
    except swiftclient.ClientException:
        conn.put_container(name, headers={"X-Container-Read": ".r:*,.rlistings"})
        print(f"  Created container {name}")

def swift_upload_file(conn, container, object_name, filepath):
    with open(filepath, "rb") as f:
        conn.put_object(container, object_name, contents=f)
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"  Uploaded {object_name} ({size_mb:.1f} MB)")

def swift_upload_bytes(conn, container, object_name, data):
    conn.put_object(container, object_name, contents=data)
    print(f"  Uploaded {object_name}")

def swift_download_file(conn, container, object_name, local_path):
    headers, content = conn.get_object(container, object_name)
    with open(local_path, "wb") as f:
        f.write(content)
    print(f"  Downloaded {object_name}")

def export_pg_interactions(db_url, out_path):
    if not HAS_PG: return None
    try:
        conn=psycopg2.connect(db_url)
        df=pd.read_sql("SELECT user_id,recipe_id,rating,created_at AS date FROM user_interactions ORDER BY created_at",conn)
        conn.close()
        if len(df): df.to_csv(out_path,index=False); print(f"  Exported {len(df):,} production interactions"); return df
    except Exception as e: print(f"  DB export: {e}")
    return None

def validate(train,val,test):
    print("\n-- Data Quality Checks --"); ok=0
    for name,df in [("train",train),("val",val),("test",test)]:
        if len(df)==0: raise ValueError(f"{name} split is empty!")
    print("  OK: No empty splits"); ok+=1
    if "date" in train.columns and "date" in test.columns:
        tmax=pd.to_datetime(train["date"]).max(); tmin=pd.to_datetime(test["date"]).min()
        if tmax<tmin: print(f"  OK: Temporal order (train max={tmax}, test min={tmin})"); ok+=1
        else: raise ValueError(f"Temporal leakage! train max {tmax} >= test min {tmin}")
    if "label" in train.columns:
        bal=train["label"].mean()
        print(f"  OK: Label balance: {bal:.1%} positive"); ok+=1
    crit=[c for c in ["user_id","recipe_id","label"] if c in train.columns]
    if train[crit].isnull().sum().sum()==0: print("  OK: No nulls in key columns"); ok+=1
    if "label" in train.columns:
        fcols=[c for c in train.columns if c.startswith("daily_") or c.startswith("user_")]
        if fcols:
            corrs=train[fcols+["label"]].corr()["label"].drop("label")
            bad=corrs[corrs.abs()>0.9]
            if len(bad)==0: print("  OK: No feature-label leakage"); ok+=1
            else: print(f"  WARN: Possible leakage: {dict(bad)}")
    print(f"\n  Checks passed: {ok}"); return True

def _generate_synthetic(out: Path):
    """Generate minimal synthetic train/val/test CSVs when raw Food.com data is unavailable."""
    rng = np.random.default_rng(42)
    n_recipes = 500
    recipe_ids = np.arange(1, n_recipes + 1)
    cuisines = ["italian", "american", "mexican", "asian", "mediterranean", "unknown"]

    def make_interactions(n, date_offset_days):
        from datetime import timedelta
        base = datetime(2024, 1, 1) + timedelta(days=date_offset_days)
        user_ids = rng.integers(1, 201, size=n)
        rec_ids = rng.choice(recipe_ids, size=n)
        ratings = rng.integers(1, 6, size=n)
        dates = [
            (base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
            for d in rng.integers(0, max(1, 28 - date_offset_days // 30), size=n)
        ]
        avg_ratings = rng.uniform(3.0, 5.0, size=n)
        n_reviews = rng.integers(3, 200, size=n)
        minutes = rng.integers(5, 121, size=n)
        n_steps = rng.integers(2, 16, size=n)
        n_ingredients = rng.integers(2, 21, size=n)
        cal = rng.uniform(100, 800, size=n)
        fat = rng.uniform(2, 50, size=n)
        sugar = rng.uniform(0, 60, size=n)
        sodium_g = rng.uniform(0.1, 3.0, size=n)
        protein = rng.uniform(2, 60, size=n)
        sat_fat = rng.uniform(0, 20, size=n)
        carb = rng.uniform(5, 120, size=n)
        return pd.DataFrame({
            "user_id": user_ids,
            "recipe_id": rec_ids,
            "rating": ratings,
            "date": dates,
            "label": (ratings >= 4).astype(int),
            "avg_rating": avg_ratings.round(2),
            "n_reviews": n_reviews,
            "minutes": minutes,
            "n_steps": n_steps,
            "n_ingredients": n_ingredients,
            "cuisine": rng.choice(cuisines, size=n),
            "calories": (cal / 2000 * 100).round(1),
            "total_fat": (fat / 78 * 100).round(1),
            "sugar": (sugar / 50 * 100).round(1),
            "sodium": (sodium_g / 2.3 * 100).round(1),
            "protein": (protein / 50 * 100).round(1),
            "saturated_fat": (sat_fat / 20 * 100).round(1),
            "carbohydrate": (carb / 275 * 100).round(1),
            "total_fat_g": fat.round(2),
            "sugar_g": sugar.round(2),
            "sodium_g": sodium_g.round(4),
            "protein_g": protein.round(2),
            "saturated_fat_g": sat_fat.round(2),
            "carbohydrate_g": carb.round(2),
            "has_egg": rng.integers(0, 2, size=n),
            "has_fish": rng.integers(0, 2, size=n),
            "has_milk": rng.integers(0, 2, size=n),
            "has_nuts": rng.integers(0, 2, size=n),
            "has_peanut": rng.integers(0, 2, size=n),
            "has_sesame": rng.integers(0, 2, size=n),
            "has_shellfish": rng.integers(0, 2, size=n),
            "has_soy": rng.integers(0, 2, size=n),
            "has_wheat": rng.integers(0, 2, size=n),
            "daily_calorie_target": rng.choice([1800, 2000, 2200, 2500], size=n).astype(float),
            "protein_target_g": rng.choice([40, 50, 60, 80], size=n).astype(float),
            "carbs_target_g": rng.choice([200, 250, 300], size=n).astype(float),
            "fat_target_g": rng.choice([55, 65, 78], size=n).astype(float),
            "user_vegetarian": rng.integers(0, 2, size=n),
            "user_vegan": rng.integers(0, 2, size=n),
            "user_gluten_free": rng.integers(0, 2, size=n),
            "user_dairy_free": rng.integers(0, 2, size=n),
            "user_low_sodium": rng.integers(0, 2, size=n),
            "user_low_fat": rng.integers(0, 2, size=n),
            "history_pc1": rng.uniform(-2, 2, size=n).round(4),
            "history_pc2": rng.uniform(-2, 2, size=n).round(4),
            "history_pc3": rng.uniform(-2, 2, size=n).round(4),
            "history_pc4": rng.uniform(-2, 2, size=n).round(4),
            "history_pc5": rng.uniform(-2, 2, size=n).round(4),
            "history_pc6": rng.uniform(-2, 2, size=n).round(4),
        })

    train = make_interactions(8000, 0)
    val   = make_interactions(1000, 365)
    test  = make_interactions(1000, 380)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    # minimal enriched_recipes and training_table for downstream steps
    cols = ["recipe_id", "avg_rating", "n_reviews", "minutes"]
    train[cols].drop_duplicates("recipe_id").to_csv(out / "enriched_recipes.csv", index=False)
    train.to_csv(out / "training_table.csv", index=False)
    print(f"  Synthetic splits: train={len(train)}, val={len(val)}, test={len(test)}")


def _write_manifest(out: Path, train_df, val_df, test_df, source: str):
    th = sha256_file(out / "train.csv")
    vh = sha256_file(out / "val.csv")
    eh = sha256_file(out / "test.csv")
    ver = sha256_str(f"{th}:{vh}:{eh}")[:12]

    def safe_date(df, col):
        try: return str(pd.to_datetime(df[col]).min()) + " / " + str(pd.to_datetime(df[col]).max())
        except: return None

    manifest = {
        "version": f"v{ver}",
        "pipeline_version": PIPELINE_VERSION,
        "source": source,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "splits": {
            "train": {"rows": len(train_df), "sha256": th, "dates": safe_date(train_df, "date")},
            "val":   {"rows": len(val_df),   "sha256": vh, "dates": safe_date(val_df, "date")},
            "test":  {"rows": len(test_df),  "sha256": eh, "dates": safe_date(test_df, "date")},
        },
        "label_balance": {
            s: float(df["label"].mean()) if "label" in df.columns else None
            for s, df in [("train", train_df), ("val", val_df), ("test", test_df)]
        },
    }
    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"  Manifest -> {mpath}  (v{ver})")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw-dir",default="./data")
    ap.add_argument("--output-dir",default="./output")
    ap.add_argument("--db-url",default=None)
    ap.add_argument("--upload",action="store_true")
    ap.add_argument("--download-raw",action="store_true")
    args=ap.parse_args()
    print("="*60); print("SparkyFitness — Batch Training Data Pipeline")
    print(f"Version {PIPELINE_VERSION}  |  {datetime.utcnow().isoformat()}Z"); print("="*60)
    raw=Path(args.raw_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    if args.download_raw and HAS_SWIFT:
        print("\n-- Step 0: Downloading from Chameleon Object Storage --")
        conn=get_swift_conn(); raw.mkdir(parents=True,exist_ok=True)
        for f in ["RAW_recipes.csv","RAW_interactions.csv"]:
            swift_download_file(conn, RAW_BUCKET, f, str(raw/f))
        conn.close()

    print("\n-- Step 1: Loading raw data --")
    rp=raw/"RAW_recipes.csv"
    if not rp.exists(): rp=raw/"recipes.csv"
    ip=raw/"RAW_interactions.csv"
    if not ip.exists(): ip=raw/"interactions.csv"
    raw_missing = not rp.exists() or not ip.exists()
    if raw_missing:
        print("  RAW data files not found — generating synthetic training data for self-contained deployment.")
        _generate_synthetic(out)
        print("  Synthetic data ready. Skipping Steps 2–3.")
        train_df=pd.read_csv(out/"train.csv"); val_df=pd.read_csv(out/"val.csv"); test_df=pd.read_csv(out/"test.csv")
        validate(train_df,val_df,test_df)
        _write_manifest(out,train_df,val_df,test_df,"synthetic")
        return
    df_r=pd.read_csv(rp); df_r.columns=df_r.columns.str.strip().str.lower()
    df_i=pd.read_csv(ip); df_i.columns=df_i.columns.str.strip().str.lower()
    print(f"  Recipes:      {df_r.shape}"); print(f"  Interactions: {df_i.shape}")

    if args.db_url:
        print("\n-- Step 2: Merging production interactions --")
        prod=export_pg_interactions(args.db_url,out/"production_interactions.csv")
        if prod is not None and len(prod):
            prod.columns=prod.columns.str.strip().str.lower()
            if "rating" in prod.columns and "user_id" in prod.columns:
                df_i=pd.concat([df_i,prod],ignore_index=True); print(f"  Merged total: {len(df_i):,}")
    else: print("\n-- Step 2: No DB URL — Kaggle data only --")

    print("\n-- Step 3: Running enrichment pipeline --")
    btm_path=None
    # Run build_training_table.py as subprocess (avoids import/path issues)
    import subprocess
    btm_path = None
    for candidate in [
        Path(__file__).parent/"build_training_table.py",   # src/data/ sibling
        raw/"build_training_table.py",                     # raw data dir (legacy)
        Path("build_training_table.py"),                   # cwd (legacy)
        Path("/app/build_training_table.py"),              # container root symlink
    ]:
        if candidate.exists(): btm_path = candidate; break
    if btm_path:
        print(f"  Using {btm_path}")
        env = os.environ.copy()
        result = subprocess.run(["python", str(btm_path)], cwd=str(raw), env=env, capture_output=False)
        if result.returncode != 0:
            print("  ERROR: build_training_table.py failed"); sys.exit(1)
        # Move output files to output dir
        for sp in ["enriched_recipes.csv","training_table.csv","train.csv","val.csv","test.csv"]:
            src = raw/sp
            if src.exists(): shutil.copy(str(src), str(out/sp))
    else:
        print("  build_training_table.py not found, running inline fallback")
        df_r=df_r.rename(columns={"id":"recipe_id"})
        grp=df_i.groupby("recipe_id")["rating"].agg(avg_rating="mean",n_reviews="count").reset_index()
        enriched=df_r.merge(grp,on="recipe_id",how="left")
        enriched["avg_rating"]=enriched["avg_rating"].fillna(0)
        enriched["n_reviews"]=enriched["n_reviews"].fillna(0).astype(int)
        enriched=enriched[(enriched["n_reviews"]>=3)&(enriched["minutes"]>0)]
        df_i["label"]=(df_i["rating"]>=4).astype(int)
        training=df_i.merge(enriched[["recipe_id","avg_rating","n_reviews","minutes"]],on="recipe_id",how="inner")
        training["date"]=pd.to_datetime(training["date"]); latest=training["date"].max()
        train_df=training[training["date"]<latest-pd.Timedelta(days=14)]
        val_df=training[(training["date"]>=latest-pd.Timedelta(days=14))&(training["date"]<latest-pd.Timedelta(days=7))]
        test_df=training[training["date"]>=latest-pd.Timedelta(days=7)]
        enriched.to_csv(out/"enriched_recipes.csv",index=False)
        training.to_csv(out/"training_table.csv",index=False)
        train_df.to_csv(out/"train.csv",index=False)
        val_df.to_csv(out/"val.csv",index=False)
        test_df.to_csv(out/"test.csv",index=False)

    train_df=pd.read_csv(out/"train.csv"); val_df=pd.read_csv(out/"val.csv"); test_df=pd.read_csv(out/"test.csv")
    enriched=pd.read_csv(out/"enriched_recipes.csv")
    validate(train_df,val_df,test_df)

    print("\n-- Step 5: Version hashing --")
    th=sha256_file(out/"train.csv"); vh=sha256_file(out/"val.csv"); eh=sha256_file(out/"test.csv")
    ver=sha256_str(f"{th}:{vh}:{eh}")[:12]
    print(f"  train  {th[:16]}..."); print(f"  val    {vh[:16]}..."); print(f"  test   {eh[:16]}...")
    print(f"  Version: v{ver}")

    def safe_date(df,col):
        try: return str(pd.to_datetime(df[col]).min())+" / "+str(pd.to_datetime(df[col]).max())
        except: return None
    manifest={"version":f"v{ver}","pipeline_version":PIPELINE_VERSION,
        "created_at":datetime.utcnow().isoformat()+"Z","git_commit":os.environ.get("GIT_COMMIT","unknown"),
        "splits":{"train":{"rows":len(train_df),"sha256":th,"dates":safe_date(train_df,"date")},
                  "val":{"rows":len(val_df),"sha256":vh,"dates":safe_date(val_df,"date")},
                  "test":{"rows":len(test_df),"sha256":eh,"dates":safe_date(test_df,"date")}},
        "label_balance":{s:float(df["label"].mean()) if "label" in df.columns else None
                         for s,df in [("train",train_df),("val",val_df),("test",test_df)]},
        "enriched_recipes":len(enriched)}
    mpath=out/"manifest.json"; mpath.write_text(json.dumps(manifest,indent=2,default=str))
    print(f"\n  Manifest -> {mpath}")

    if args.upload and HAS_SWIFT:
        print("\n-- Step 7: Uploading to Chameleon Object Storage (Swift) --")
        conn = get_swift_conn()
        ensure_container(conn, TRAINING_BUCKET)
        prefix=f"v{ver}"
        for fn in ["enriched_recipes.csv","training_table.csv","train.csv","val.csv","test.csv","manifest.json"]:
            fp=out/fn
            if fp.exists():
                swift_upload_file(conn, TRAINING_BUCKET, f"{prefix}/{fn}", fp)
        swift_upload_bytes(conn, TRAINING_BUCKET, "latest", prefix.encode())
        print(f"  latest -> {prefix}")
        conn.close()
    else:
        print("\n-- Step 7: Upload skipped (use --upload to enable) --")

    print("\n"+"="*60)
    print(f"DONE: Batch Pipeline Complete — v{ver}")
    print(f"   Enriched recipes: {len(enriched):>10,}")
    print(f"   Train:            {len(train_df):>10,}")
    print(f"   Val:              {len(val_df):>10,}")
    print(f"   Test:             {len(test_df):>10,}")
    print(f"   Output dir:       {out}")
    print("="*60)

if __name__=="__main__": main()
