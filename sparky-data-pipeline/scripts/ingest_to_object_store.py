"""
ingest_to_object_store.py  —  Q2.3: External Data Ingestion
Runs non-interactively: docker compose --profile ingest run --rm ingest
"""
import argparse, ast, hashlib, json, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

try:
    import swiftclient
    HAS_SWIFT = True
except ImportError:
    HAS_SWIFT = False
    print("warn: python-swiftclient not installed — dry-run mode")

RAW_BUCKET = "proj04-sparky-raw-data"
NOISE_STD = 0.05; EXPAND_FACTOR = 2; MAX_SIZE_GB = 5
EXPECTED_RECIPE_COLS = ["name","id","minutes","tags","nutrition","n_steps","n_ingredients","ingredients","description"]
EXPECTED_INTER_COLS  = ["user_id","recipe_id","date","rating","review"]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()

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
        print(f"  Created container {name} (public read)")

def upload_file(conn, container, object_name, filepath, metadata=None):
    headers = {}
    if metadata:
        for k, v in metadata.items():
            headers[f"X-Object-Meta-{k}"] = str(v)
    with open(filepath, "rb") as f:
        conn.put_object(container, object_name, contents=f, headers=headers)
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"  Uploaded {object_name} ({size_mb:.1f} MB)")

def upload_bytes(conn, container, object_name, data, metadata=None):
    headers = {}
    if metadata:
        for k, v in metadata.items():
            headers[f"X-Object-Meta-{k}"] = str(v)
    conn.put_object(container, object_name, contents=data, headers=headers)
    print(f"  Uploaded {object_name}")

def validate_recipes(df):
    print("\n-- Validating recipes --")
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in EXPECTED_RECIPE_COLS if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    assert df["id"].nunique() == len(df), "Duplicate recipe IDs"
    assert df["id"].notna().all(), "Null recipe IDs"
    print(f"  OK: {len(df):,} recipes"); return df

def validate_interactions(df):
    print("\n-- Validating interactions --")
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in EXPECTED_INTER_COLS if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    assert df["user_id"].notna().all(), "Null user_ids"
    assert df["recipe_id"].notna().all(), "Null recipe_ids"
    print(f"  OK: {len(df):,} interactions"); return df

def expand_recipes(df):
    n = int(len(df) * (EXPAND_FACTOR - 1))
    print(f"\n-- Synthetic expansion: generating {n:,} rows --")
    syn = df.sample(n=n, replace=True, random_state=42).reset_index(drop=True)
    for col in ["minutes","n_steps","n_ingredients"]:
        if col in syn.columns:
            noise = np.random.normal(0, NOISE_STD * syn[col].std(), size=n)
            syn[col] = np.maximum(1, (syn[col] + noise).round().astype(int))
    if "nutrition" in syn.columns:
        def _perturb(x):
            try:
                vals = ast.literal_eval(x) if isinstance(x,str) else x
                if isinstance(vals,(list,tuple)) and len(vals)>=7:
                    return str([max(0, v*(1+np.random.normal(0,NOISE_STD))) for v in vals])
            except Exception: pass
            return x
        syn["nutrition"] = syn["nutrition"].apply(_perturb)
    max_id = df["id"].max() if "id" in df.columns else 0
    syn["id"] = range(max_id+1, max_id+1+n)
    syn["name"] = syn["name"].apply(lambda x: f"[Synthetic] {x}" if isinstance(x,str) else x)
    df["is_synthetic"] = False; syn["is_synthetic"] = True
    combined = pd.concat([df, syn], ignore_index=True)
    print(f"  {len(df):,} -> {len(combined):,} recipes"); return combined

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default="./data")
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print("="*60); print("SparkyFitness — External Data Ingestion Pipeline"); print("="*60)
    d = Path(args.local_dir)
    rp = d/"RAW_recipes.csv"
    if not rp.exists(): rp = d/"recipes.csv"
    ip = d/"RAW_interactions.csv"
    if not ip.exists(): ip = d/"interactions.csv"
    for p in [rp, ip]:
        if not p.exists(): print(f"  ERROR: {p} not found"); sys.exit(1)
    df_r = validate_recipes(pd.read_csv(rp))
    df_i = validate_interactions(pd.read_csv(ip))
    print("\n-- Computing SHA-256 hashes --")
    rh = sha256_file(rp); ih = sha256_file(ip)
    print(f"  recipes      {rh[:16]}..."); print(f"  interactions {ih[:16]}...")
    size_gb = (os.path.getsize(rp) + os.path.getsize(ip)) / 1e9
    print(f"  Total size: {size_gb:.2f} GB")
    expanded_path = None
    if args.expand and size_gb < MAX_SIZE_GB:
        df_exp = expand_recipes(df_r)
        expanded_path = d / "RAW_recipes_expanded.csv"
        df_exp.to_csv(expanded_path, index=False)
    elif args.expand:
        print(f"  Size {size_gb:.2f} GB >= {MAX_SIZE_GB} GB, skip expansion")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    meta_r = {"source":"kaggle-food-com","download-date":ts,"sha256":rh,"row-count":str(len(df_r))}
    meta_i = {"source":"kaggle-food-com","download-date":ts,"sha256":ih,"row-count":str(len(df_i))}

    if args.dry_run or not HAS_SWIFT:
        print("\n-- DRY RUN — would upload to container:", RAW_BUCKET)
    else:
        print("\n-- Uploading to Chameleon Object Storage (Swift) --")
        conn = get_swift_conn()
        ensure_container(conn, RAW_BUCKET)
        upload_file(conn, RAW_BUCKET, "RAW_recipes.csv", rp, meta_r)
        upload_file(conn, RAW_BUCKET, "RAW_interactions.csv", ip, meta_i)
        if expanded_path and expanded_path.exists():
            upload_file(conn, RAW_BUCKET, "RAW_recipes_expanded.csv",
                        expanded_path, {**meta_r, "synthetic": "true"})
        manifest = {"pipeline":"ingest_to_object_store.py","timestamp":ts,
                     "files":{"RAW_recipes.csv":{"sha256":rh,"rows":len(df_r)},
                              "RAW_interactions.csv":{"sha256":ih,"rows":len(df_i)}},
                     "expanded": expanded_path is not None}
        upload_bytes(conn, RAW_BUCKET, "manifest.json",
                     json.dumps(manifest, indent=2).encode())
        conn.close()

    print("\n"+"="*60)
    print("DONE: Ingestion complete")
    print(f"   Recipes:      {len(df_r):>10,} rows")
    print(f"   Interactions: {len(df_i):>10,} rows")
    print("="*60)

if __name__ == "__main__":
    main()
