"""
batch_pipeline.py  —  Q2.6: Batch Training Data Pipeline
Compiles versioned training/evaluation datasets from production data.
Uses Chameleon Object Storage (OpenStack Swift) for versioned artifacts.
Runs non-interactively:  docker compose --profile batch run --rm batch-pipeline
"""
import argparse, hashlib, importlib.util, json, os, shutil, sys, time
from datetime import datetime
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import swiftclient
    HAS_SWIFT = True
except ImportError:
    HAS_SWIFT = False

try: import psycopg2; HAS_PG=True
except ImportError: HAS_PG=False

try:
    from src.serving.feature_contract import FEATURE_COLUMNS
except Exception:
    FEATURE_COLUMNS = []

PIPELINE_VERSION="1.0.0"; RAW_BUCKET="proj04-sparky-raw-data"; TRAINING_BUCKET="proj04-sparky-training-data"
ONLINE_UPSAMPLE_FACTOR = 5
ACTION_LABEL_MAP = {"logged": 1, "saved": 1, "dismissed": 0, "viewed": 0}
ACTION_PRIORITY = {"logged": 3, "saved": 3, "dismissed": 2, "viewed": 1}
POSITIVE_RATING_THRESHOLD = 4.0
NEGATIVE_RATING_THRESHOLD = 2.0

# Quality gate: only keep feedback from the last N days (stale labels distort new model)
FEEDBACK_FRESHNESS_DAYS = int(os.environ.get("FEEDBACK_FRESHNESS_DAYS", "14"))
# Quality gate: exclude feedback logged within this many seconds of a drift event
DRIFT_EXCLUSION_WINDOW_SEC = int(os.environ.get("DRIFT_EXCLUSION_WINDOW_SEC", "3600"))

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
            "project_domain_name": os.environ.get("OS_PROJECT_DOMAIN_NAME", "chameleon"),
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
        df=pd.read_sql(
            """
            SELECT user_id, recipe_id, rating, created_at AS date
            FROM user_interactions
            WHERE rating IS NOT NULL
            ORDER BY created_at
            """,
            conn
        )
        conn.close()
        if len(df): df.to_csv(out_path,index=False); print(f"  Exported {len(df):,} production interactions"); return df
    except Exception as e: print(f"  DB export: {e}")
    return None


def feature_default(name):
    return "unknown" if name == "cuisine" else 0.0


def infer_online_label(record):
    action = (record.get("action") or "").strip().lower()
    if action in ACTION_LABEL_MAP:
        return ACTION_LABEL_MAP[action]

    rating = record.get("rating")
    if rating is None or rating == "":
        return None

    try:
        rating_value = float(rating)
    except (TypeError, ValueError):
        return None

    if rating_value >= POSITIVE_RATING_THRESHOLD:
        return 1
    if rating_value <= NEGATIVE_RATING_THRESHOLD:
        return 0
    return None


def online_feedback_priority(record, label):
    action = (record.get("action") or "").strip().lower()
    if action in ACTION_PRIORITY:
        return ACTION_PRIORITY[action]
    return 3 if label == 1 else 2


def online_feedback_key(record):
    recommendation_id = record.get("recommendation_id")
    if recommendation_id:
        return str(recommendation_id)
    return ":".join(
        str(record.get(key, ""))
        for key in ("request_id", "inf_user_id", "inf_recipe_id")
    )


def choose_online_training_records(records):
    chosen = {}
    for record in records:
        label = infer_online_label(record)
        if label is None:
            continue
        key = online_feedback_key(record)
        priority = online_feedback_priority(record, label)
        current = chosen.get(key)
        if current is None or priority >= current[0]:
            chosen[key] = (priority, label, record)
    return [(label, record) for _, label, record in chosen.values()]


def filter_uniform_feedback_users(chosen):
    """Drop users whose entire feedback history is a single label (all saved or all dismissed).
    These are likely disengaged users or bots — their labels add no signal and can dominate
    a small online batch.  Users with only one feedback record are kept (can't detect variance).
    """
    from collections import defaultdict
    user_labels = defaultdict(set)
    for label, record in chosen:
        user_labels[str(record.get("inf_user_id", ""))].add(label)
    valid_users = {uid for uid, label_set in user_labels.items() if len(label_set) != 1 or len(user_labels) == 1}
    kept = [(label, r) for label, r in chosen if str(r.get("inf_user_id", "")) in valid_users]
    dropped = len(chosen) - len(kept)
    if dropped:
        print(f"  Quality filter: dropped {dropped} rows from {len(user_labels) - len(valid_users)} uniform-feedback user(s)")
    return kept


def build_online_training_rows(records, upsample_factor=ONLINE_UPSAMPLE_FACTOR):
    rows = []
    for label, record in filter_uniform_feedback_users(choose_online_training_records(records)):
        inf_user_id = int(record["inf_user_id"])
        inf_recipe_id = int(record["inf_recipe_id"])
        fb_user_id = int(record["fb_user_id"])
        fb_recipe_id = int(record["fb_recipe_id"])
        if inf_user_id != fb_user_id or inf_recipe_id != fb_recipe_id:
            continue

        payload = record.get("features_json")
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            continue

        row = {
            "user_id": inf_user_id,
            "recipe_id": inf_recipe_id,
            "date": record.get("date"),
            "label": label,
            "data_source": "online",
            "request_id": record.get("request_id"),
            "recommendation_id": record.get("recommendation_id"),
        }
        for col in FEATURE_COLUMNS:
            row[col] = payload.get(col, feature_default(col))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    if upsample_factor > 1:
        frame = pd.concat([frame] * upsample_factor, ignore_index=True)
    return frame


def export_online_training_rows(db_url, out_path):
    if not HAS_PG:
        return pd.DataFrame()
    try:
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            # Fix: freshness gate — only use feedback from the last FEEDBACK_FRESHNESS_DAYS days.
            # Stale labels (logged weeks after the prediction) describe a user preference that
            # may no longer be current and distort the new model's targets.
            #
            # Fix: drift exclusion — skip feedback recorded within DRIFT_EXCLUSION_WINDOW_SEC
            # of a detected drift event. During drift the production distribution differs from
            # training, so labels from that window are less reliable.
            cur.execute(
                """
                SELECT
                    inf.request_id,
                    inf.recommendation_id,
                    inf.user_id        AS inf_user_id,
                    inf.recipe_id      AS inf_recipe_id,
                    inf.features::text AS features_json,
                    fb.user_id         AS fb_user_id,
                    fb.recipe_id       AS fb_recipe_id,
                    fb.rating,
                    fb.action,
                    COALESCE(fb.feedback_at, inf.captured_at) AS date
                FROM inference_features inf
                JOIN user_feedback fb
                  ON inf.request_id        = fb.request_id
                 AND inf.recommendation_id = fb.recommendation_id
                WHERE (fb.action IN ('logged', 'saved', 'dismissed', 'viewed')
                       OR fb.rating IS NOT NULL)
                  AND COALESCE(fb.feedback_at, inf.captured_at)
                        >= NOW() - INTERVAL '1 day' * %s
                  AND NOT EXISTS (
                        SELECT 1 FROM drift_log dl
                        WHERE dl.drift_detected = TRUE
                          AND ABS(EXTRACT(EPOCH FROM (
                                COALESCE(fb.feedback_at, inf.captured_at) - dl.run_at
                              ))) < %s
                      )
                ORDER BY COALESCE(fb.feedback_at, inf.captured_at)
                """,
                (FEEDBACK_FRESHNESS_DAYS, DRIFT_EXCLUSION_WINDOW_SEC),
            )
            cols = [desc[0] for desc in cur.description]
            records = [dict(zip(cols, row)) for row in cur.fetchall()]
        conn.close()
        print(f"  Freshness gate: last {FEEDBACK_FRESHNESS_DAYS} days | "
              f"Drift exclusion window: {DRIFT_EXCLUSION_WINDOW_SEC}s")
        frame = build_online_training_rows(records)
        if len(frame):
            frame.to_csv(out_path, index=False)
            print(f"  Exported {len(frame):,} upsampled online training rows")
        return frame
    except Exception as e:
        print(f"  Online feedback export: {e}")
        return pd.DataFrame()


def _temporal_split(frame, train_frac=0.80, val_frac=0.10):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame.sort_values("date").reset_index(drop=True)
    n = len(frame)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:val_end].copy(),
        frame.iloc[val_end:].copy(),
    )


def split_training_frame(training, train_frac=0.80, val_frac=0.10):
    """Split training rows while preserving live feedback in the train split.

    Online feedback is newer than the static Food.com corpus. A single global
    temporal split pushes most or all online rows into validation/test, which
    makes retraining evaluate on feedback the model never learned from. Split
    each source independently so live user feedback is represented in training
    while still keeping a live holdout when enough rows are available.
    """
    training = training.copy()
    if "data_source" not in training.columns:
        return _temporal_split(training, train_frac=train_frac, val_frac=val_frac)

    training["data_source"] = training["data_source"].fillna("offline").astype(str)
    split_parts = {"train": [], "val": [], "test": []}
    for _, source_frame in training.groupby("data_source", sort=False):
        train_part, val_part, test_part = _temporal_split(
            source_frame,
            train_frac=train_frac,
            val_frac=val_frac,
        )
        split_parts["train"].append(train_part)
        split_parts["val"].append(val_part)
        split_parts["test"].append(test_part)

    def combine(parts):
        if not parts:
            return pd.DataFrame(columns=training.columns)
        return pd.concat(parts, ignore_index=True, sort=False).sort_values("date").reset_index(drop=True)

    return (
        combine(split_parts["train"]),
        combine(split_parts["val"]),
        combine(split_parts["test"]),
    )

def validate(train,val,test):
    print("\n-- Data Quality Checks --"); ok=0
    for name,df in [("train",train),("val",val),("test",test)]:
        if len(df)==0: raise ValueError(f"{name} split is empty!")
    print("  OK: No empty splits"); ok+=1
    if "date" in train.columns and "date" in test.columns:
        source_col = "data_source" if "data_source" in train.columns and "data_source" in test.columns else None
        sources = sorted(set(train[source_col].dropna().astype(str)) | set(test[source_col].dropna().astype(str))) if source_col else [None]
        for source in sources:
            train_part = train[train[source_col].astype(str) == source] if source_col else train
            test_part = test[test[source_col].astype(str) == source] if source_col else test
            if train_part.empty or test_part.empty:
                continue
            tmax=pd.to_datetime(train_part["date"], errors="coerce", format="mixed", utc=True).max(); tmin=pd.to_datetime(test_part["date"], errors="coerce", format="mixed", utc=True).min()
            label = f" for {source}" if source_col else ""
            if tmax<tmin: print(f"  OK: Temporal order{label} (train max={tmax}, test min={tmin})"); ok+=1
            else: raise ValueError(f"Temporal leakage{label}! train max {tmax} >= test min {tmin}")
    if "label" in train.columns:
        bal=train["label"].mean()
        print(f"  OK: Label balance: {bal:.1%} positive"); ok+=1
    crit=[c for c in ["user_id","recipe_id","label"] if c in train.columns]
    if train[crit].isnull().sum().sum()==0: print("  OK: No nulls in key columns"); ok+=1
    if "label" in train.columns:
        fcols=[
            c for c in train.columns
            if (c.startswith("daily_") or c.startswith("user_"))
            and c not in {"user_id", "label"}
        ]
        if fcols:
            corrs=train[fcols+["label"]].corr()["label"].drop("label")
            bad=corrs[corrs.abs()>0.9]
            if len(bad)==0: print("  OK: No feature-label leakage"); ok+=1
            else: print(f"  WARN: Possible leakage: {dict(bad)}")
    print(f"\n  Checks passed: {ok}"); return True


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
        print("  ERROR: RAW_recipes.csv or RAW_interactions.csv not found in", raw)
        print("  Either place the Food.com raw files in the data/ directory,")
        print("  or provide Swift credentials to download them automatically.")
        sys.exit(1)
    df_r=pd.read_csv(rp); df_r.columns=df_r.columns.str.strip().str.lower()
    df_i=pd.read_csv(ip); df_i.columns=df_i.columns.str.strip().str.lower()
    print(f"  Recipes:      {df_r.shape}"); print(f"  Interactions: {df_i.shape}")

    online_rows = pd.DataFrame()
    if args.db_url:
        print("\n-- Step 2: Merging production interactions --")
        prod=export_pg_interactions(args.db_url,out/"production_interactions.csv")
        if prod is not None and len(prod):
            prod.columns=prod.columns.str.strip().str.lower()
            if "rating" in prod.columns and "user_id" in prod.columns:
                df_i=pd.concat([df_i,prod],ignore_index=True); print(f"  Merged total: {len(df_i):,}")
        online_rows = export_online_training_rows(args.db_url, out/"online_training_rows.csv")
    else: print("\n-- Step 2: No DB URL — Kaggle data only --")

    print("\n-- Step 3: Running enrichment pipeline --")
    btm_path=None
    # Run build_training_table.py as subprocess (avoids import/path issues)
    import subprocess
    btm_path = None
    for candidate in [
        Path(__file__).resolve().parent/"build_training_table.py",  # src/data/ sibling
        raw/"build_training_table.py",                     # raw data dir (legacy)
        Path("build_training_table.py"),                   # cwd (legacy)
        Path("/app/build_training_table.py"),              # container root symlink
    ]:
        if candidate.exists(): btm_path = candidate; break
    if btm_path:
        print(f"  Using {btm_path}")
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{PROJECT_ROOT}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(PROJECT_ROOT)
        )
        result = subprocess.run(
            [sys.executable, str(btm_path.resolve())],
            cwd=str(raw),
            env=env,
            capture_output=False,
        )
        if result.returncode != 0:
            print("  ERROR: build_training_table.py failed"); sys.exit(1)
        # Move output files to output dir
        for sp in ["enriched_recipes.csv","training_table.csv","train.csv","val.csv","test.csv"]:
            src = raw/sp
            if src.exists(): shutil.copy(str(src), str(out/sp))
        if (raw / "embedding_artifacts").exists():
            shutil.copytree(raw / "embedding_artifacts", out / "embedding_artifacts", dirs_exist_ok=True)
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
        training=training.sort_values("date").reset_index(drop=True)
        n=len(training); train_end=int(n*0.80); val_end=int(n*0.90)
        train_df=training.iloc[:train_end]
        val_df=training.iloc[train_end:val_end]
        test_df=training.iloc[val_end:]
        enriched.to_csv(out/"enriched_recipes.csv",index=False)
        training.to_csv(out/"training_table.csv",index=False)
        train_df.to_csv(out/"train.csv",index=False)
        val_df.to_csv(out/"val.csv",index=False)
        test_df.to_csv(out/"test.csv",index=False)

    if len(online_rows):
        print("\n-- Step 4: Appending online feedback rows --")
        training_df = pd.read_csv(out/"training_table.csv")
        if "data_source" not in training_df.columns:
            training_df["data_source"] = "offline"
        else:
            training_df["data_source"] = training_df["data_source"].fillna("offline")
        merged_training = pd.concat([training_df, online_rows], ignore_index=True, sort=False)
        train_df, val_df, test_df = split_training_frame(merged_training)
        merged_training.to_csv(out/"training_table.csv", index=False)
        train_df.to_csv(out/"train.csv", index=False)
        val_df.to_csv(out/"val.csv", index=False)
        test_df.to_csv(out/"test.csv", index=False)
        print(f"  Offline rows: {len(training_df):,}")
        print(f"  Online rows:  {len(online_rows):,}")
        print(f"  Total rows:   {len(merged_training):,}")

    train_df = pd.read_csv(out/"train.csv")
    val_df = pd.read_csv(out/"val.csv")
    test_df = pd.read_csv(out/"test.csv")

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
        artifact_dir = out / "embedding_artifacts"
        if artifact_dir.exists():
            for fp in artifact_dir.iterdir():
                if fp.is_file():
                    swift_upload_file(conn, TRAINING_BUCKET, f"{prefix}/embedding_artifacts/{fp.name}", fp)
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
