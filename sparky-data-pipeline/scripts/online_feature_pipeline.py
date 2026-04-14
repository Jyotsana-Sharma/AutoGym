"""
online_feature_pipeline.py  —  Q2.5: Online Feature Computation
Runs non-interactively: docker compose --profile online-demo run --rm online-feature-demo
Integrate-able: from online_feature_pipeline import OnlineFeaturePipeline
"""
import argparse, os, time
from typing import Dict, Optional, Tuple
import numpy as np, pandas as pd
try: import psycopg2, psycopg2.extras; HAS_PG = True
except ImportError: HAS_PG = False

ALLERGEN_COLS = [f"has_{a}" for a in ["milk","egg","nuts","peanut","fish","shellfish","wheat","soy","sesame"]]
USER_FEAT_COLS = (["daily_calorie_target","protein_target_g","carbs_target_g","fat_target_g"]
    +[f"user_{d}" for d in ["vegetarian","vegan","gluten_free","dairy_free","low_sodium","low_fat"]]
    +[f"history_pc{i}" for i in range(1,7)])
RECIPE_FEAT_COLS = (["calories","protein_g","carbohydrate_g","total_fat_g","sugar_g","sodium_mg",
    "saturated_fat_g","n_ingredients","n_steps","minutes","avg_rating","n_reviews"]+ALLERGEN_COLS)

class OnlineFeaturePipeline:
    def __init__(self, recipes_path, db_url=None, user_features_csv=None):
        print("-- Initializing OnlineFeaturePipeline --")
        self.db_conn = None
        t0 = time.time()
        self.recipes = pd.read_csv(recipes_path)
        self.recipes.columns = self.recipes.columns.str.strip().str.lower()
        if "recipe_id" not in self.recipes.columns and "id" in self.recipes.columns:
            self.recipes.rename(columns={"id":"recipe_id"}, inplace=True)
        self.recipe_lookup = {int(r["recipe_id"]): r.to_dict() for _,r in self.recipes.iterrows()}
        ms = (time.time()-t0)*1000; mb = self.recipes.memory_usage(deep=True).sum()/1e6
        print(f"  Loaded {len(self.recipes):,} recipes ({ms:.0f}ms, {mb:.1f} MB)")
        self.user_cache = {}
        if user_features_csv and os.path.exists(user_features_csv):
            udf = pd.read_csv(user_features_csv)
            self.user_cache = {int(r["user_id"]): r.to_dict() for _,r in udf.iterrows()}
            print(f"  Loaded {len(self.user_cache):,} cached user features")
        if db_url and HAS_PG:
            try: self.db_conn = psycopg2.connect(db_url); print("  Connected to PostgreSQL")
            except Exception as e: print(f"  DB: {e}")
        print("  Pipeline ready\n")

    def get_user_features(self, user_id):
        if user_id in self.user_cache: return self.user_cache[user_id]
        if self.db_conn:
            try:
                cur = self.db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("SELECT * FROM user_features_cache WHERE user_id=%s",(user_id,))
                row = cur.fetchone(); cur.close()
                if row: self.user_cache[user_id]=dict(row); return dict(row)
            except Exception: pass
            try:
                cur = self.db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("SELECT ui.recipe_id, r.calories, r.protein_g, r.carbohydrate_g, r.total_fat_g "
                    "FROM user_interactions ui JOIN recipes r ON ui.recipe_id=r.recipe_id "
                    "WHERE ui.user_id=%s AND ui.rating>=4 ORDER BY ui.created_at DESC LIMIT 100",(user_id,))
                rows = cur.fetchall(); cur.close()
                if len(rows)>=3:
                    df=pd.DataFrame(rows)
                    feats={"user_id":user_id,"daily_calorie_target":float(df["calories"].mean()),
                           "protein_target_g":float(df["protein_g"].mean()),
                           "carbs_target_g":float(df["carbohydrate_g"].mean()),
                           "fat_target_g":float(df["total_fat_g"].mean())}
                    for c in USER_FEAT_COLS: feats.setdefault(c,0)
                    self.user_cache[user_id]=feats; return feats
            except Exception: pass
        return dict(user_id=user_id,daily_calorie_target=2000,protein_target_g=50,
                    carbs_target_g=275,fat_target_g=65,
                    **{c:0 for c in USER_FEAT_COLS if c not in ["daily_calorie_target","protein_target_g","carbs_target_g","fat_target_g"]})

    def rule_filter(self, ctx):
        t0=time.time(); df=self.recipes.copy()
        col_map={a:f"has_{a}" for a in ["milk","egg","nuts","peanut","fish","shellfish","wheat","soy","sesame"]}
        for a in ctx.get("allergens",[]):
            c=col_map.get(a)
            if c and c in df.columns: df=df[df[c]==0]
        cal=ctx.get("calorie_target",600)
        if "calories" in df.columns: df=df[(df["calories"]>=cal*0.5)&(df["calories"]<=cal*1.5)]
        tlim=ctx.get("time_limit",120)
        if "minutes" in df.columns: df=df[df["minutes"]<=tlim]
        return df,(time.time()-t0)*1000

    def compute_features(self, user_id, ctx):
        t_total=time.time(); timings={}
        t0=time.time(); uf=self.get_user_features(user_id); timings["user_features_ms"]=(time.time()-t0)*1000
        cands,fms=self.rule_filter(ctx); timings["rule_filter_ms"]=fms
        if len(cands)==0:
            return pd.DataFrame(),{"candidates":0,"timings":timings,"total_ms":(time.time()-t_total)*1000}
        t0=time.time()
        rcols=[c for c in RECIPE_FEAT_COLS if c in cands.columns]
        fm=cands[["recipe_id"]+rcols].copy()
        for c in USER_FEAT_COLS: fm[c]=uf.get(c,0)
        meal=ctx.get("meal_type","dinner")
        for m in ["breakfast","lunch","dinner","snack","dessert"]: fm[f"meal_{m}"]=1 if m==meal else 0
        tod=ctx.get("time_of_day","evening")
        fm["time_morning"]=1 if tod=="morning" else 0; fm["time_evening"]=1 if tod=="evening" else 0
        if "cuisine" in cands.columns:
            dum=pd.get_dummies(cands["cuisine"],prefix="cuisine")
            for c in dum.columns: fm[c]=dum[c].values
        fm=fm.fillna(0); timings["matrix_build_ms"]=(time.time()-t0)*1000
        timings["total_ms"]=(time.time()-t_total)*1000
        return fm,{"candidates":len(fm),"features":len(fm.columns)-1,"timings":timings,"total_ms":timings["total_ms"]}

def run_demo(recipes_path, user_csv=None, db_url=None):
    print("="*60); print("SparkyFitness — Online Feature Computation Demo"); print("="*60)
    pipe=OnlineFeaturePipeline(recipes_path,db_url=db_url,user_features_csv=user_csv)
    ctx={"user_id":12345,"meal_type":"dinner","calorie_target":700,"time_limit":60,
         "allergens":["nuts","shellfish"],"dietary_restrictions":{},"time_of_day":"evening"}
    print("-- Demo Request --")
    for k,v in ctx.items(): print(f"  {k:<25} {v}")
    fm,meta=pipe.compute_features(ctx["user_id"],ctx)
    print(f"\n-- Results --")
    print(f"  Candidates after filter: {meta['candidates']}")
    print(f"  Feature dimensions:      {meta.get('features','n/a')}")
    print(f"\n-- Timing Breakdown --")
    for step,ms in meta["timings"].items(): print(f"  {step:<25} {ms:>8.1f} ms")
    if len(fm)>0:
        print(f"\n-- Sample rows (first 3) --")
        show=[c for c in ["recipe_id","calories","protein_g","minutes","daily_calorie_target","meal_dinner"] if c in fm.columns]
        print(fm[show].head(3).to_string(index=False))
        print(f"\n-- Simulated XGBoost ranking (top 5) --")
        fm["score"]=np.random.random(len(fm))
        for _,row in fm.nlargest(5,"score").iterrows():
            rec=pipe.recipe_lookup.get(int(row["recipe_id"]),{})
            print(f"    #{int(row['recipe_id']):<8} {str(rec.get('name',''))[:35]:<35} {rec.get('calories',0):>6.0f} kcal  score={row['score']:.4f}")
    print(f"\nDONE: Total latency: {meta['total_ms']:.1f}ms")
    print("="*60)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--recipes-path",default="./data/enriched_recipes.csv")
    ap.add_argument("--user-features-csv",default=None)
    ap.add_argument("--db-url",default=None)
    ap.add_argument("--demo",action="store_true")
    args=ap.parse_args()
    if args.demo: run_demo(args.recipes_path,args.user_features_csv,args.db_url)
    else: print("Use --demo for end-to-end demonstration.")

if __name__=="__main__": 
    main()
