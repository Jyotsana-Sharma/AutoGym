"""
data_generator.py  —  Q2.4: Data Generator
Hits the (hypothetical) /recommend_meals endpoint with synthetic data.
Replays held-out Food.com interactions day-by-day.
Runs non-interactively:  docker compose --profile generate run --rm data-generator
"""
import argparse, json, os, random, sys, time
from collections import defaultdict
from datetime import datetime
import numpy as np, pandas as pd
try: import requests as req_lib; HAS_REQUESTS = True
except ImportError: HAS_REQUESTS = False
try: import psycopg2; HAS_PG = True
except ImportError: HAS_PG = False

MEAL_CAL = {"breakfast":(300,600),"lunch":(500,900),"dinner":(600,1200),"snack":(100,300),"dessert":(200,500)}
TIME_LIMITS = [15,30,45,60,90,120]
ALLERGEN_OPT = [[],["milk"],["egg"],["nuts"],["wheat"],["milk","egg"],["shellfish"],[]]
DIET_OPT = [{},{"vegetarian":True},{"vegan":True},{"gluten_free":True},{"low_fat":True}]

def make_profile(uid, hist=None):
    '''
    generates a synthetic user (random macro targets, allergens, diet flags)
    '''
    return {"user_id":int(uid),"allergens":random.choice(ALLERGEN_OPT),
            "dietary_restrictions":random.choice(DIET_OPT),
            "macro_targets":{"calories":int(np.random.normal(2000,300)),
                             "protein_g":int(np.random.normal(60,15)),
                             "carbs_g":int(np.random.normal(250,50)),
                             "fat_g":int(np.random.normal(65,15))}}

def make_context(profile):
    '''
    picks meal_type, calorie_target, time_limit randomly from realistic ranges (e.g. dinner: 600–1200 kcal)
    '''
    hour = random.choices(range(24),weights=[0]*5+[1,2,3,4,3]+[2,3,4,3,2]+[1,3,4,3,2]+[1,1,0,0],k=1)[0]
    if hour<10: mt="breakfast"
    elif hour<14: mt="lunch"
    elif hour<16: mt="snack"
    elif hour<20: mt="dinner"
    else: mt=random.choice(["snack","dessert"])
    lo,hi = MEAL_CAL[mt]
    return {"user_id":profile["user_id"],"meal_type":mt,"calorie_target":random.randint(lo,hi),
            "time_limit":random.choice(TIME_LIMITS),"allergens":profile["allergens"],
            "dietary_restrictions":profile["dietary_restrictions"],
            "time_of_day":"morning" if hour<12 else "evening"}

def simulate_action(rating):
    '''
    maps predicted rating to action:
            4–5 → cook
            3–4 → view
            ≤2 → skip`
    '''
    if rating>=4: return {"action":"cook","rating":random.choice([4,5])}
    if rating>=3: return {"action":random.choice(["cook","view","view"]),"rating":random.choice([3,4])}
    if rating>=2: return {"action":random.choice(["view","skip","skip"]),"rating":random.choice([2,3])}
    return {"action":"skip","rating":random.choice([1,2])}

def call_api(url, ctx):
    '''
    POSTs to /recommend_meals endpoint (if online)
    '''
    try:
        t0=time.time(); r=req_lib.post(f"{url}/recommend_meals",json=ctx,timeout=10)
        return (r.json() if r.status_code==200 else None),(time.time()-t0)*1000
    except Exception: return None,0

def write_to_pg(conn,uid,rid,rating,action):
    '''
    inserts to user_interactions table
    Key insight: This creates the retraining flywheel. Actions logged here get merged in the next batch pipeline run via export_pg_interactions().
    '''
    try:
        cur=conn.cursor()
        cur.execute("INSERT INTO user_interactions (user_id,recipe_id,rating,action,created_at) VALUES (%s,%s,%s,%s,NOW())",(uid,rid,rating,action))
        conn.commit(); cur.close()
    except Exception as e: conn.rollback()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions-path", default="./data/RAW_interactions.csv")
    ap.add_argument("--recipes-path", default="./data/enriched_recipes.csv")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--rate", type=float, default=10)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--log-file", default="./output/generated_interactions.csv")
    args = ap.parse_args()
    print("="*60); print("SparkyFitness — Data Generator"); print("="*60)
    df = pd.read_csv(args.interactions_path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Loaded {len(df):,} interactions")
    cutoff = df["date"].max() - pd.Timedelta(days=args.days)
    prod = df[df["date"]>cutoff].copy()
    prod["day"] = prod["date"].dt.date.astype(str)
    days = sorted(prod["day"].unique())
    print(f"  Production partition: {len(prod):,} rows, {len(days)} days\n")
    conn = None
    if args.db_url and HAS_PG:
        try: conn=psycopg2.connect(args.db_url); print("  Connected to PostgreSQL")
        except Exception as e: print(f"  DB connect failed: {e}")
    profiles = {uid: make_profile(uid) for uid in prod["user_id"].unique()}
    print(f"  Generated {len(profiles):,} user profiles")
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    with open(args.log_file,"w") as f: f.write("user_id,recipe_id,rating,action,timestamp\n")
    met = defaultdict(int); met["latency_ms"]=0.0
    delay = 1/args.rate if args.rate>0 else 0
    for di,day in enumerate(days):
        rows = prod[prod["day"]==day]
        print(f"\n  Day {di+1}/{len(days)}: {day}  ({len(rows):,} interactions)")
        for _,r in rows.iterrows():
            uid,rid,rating = int(r["user_id"]),int(r["recipe_id"]),int(r["rating"])
            ctx = make_context(profiles.get(uid, make_profile(uid)))
            met["requests"]+=1
            if not args.offline and HAS_REQUESTS:
                resp,ms = call_api(args.api_url,ctx); met["latency_ms"]+=ms
                met["ok" if resp else "fail"]+=1
            else: met["ok"]+=1
            act = simulate_action(rating); met[act["action"]]+=1
            ts = datetime.utcnow().isoformat()
            with open(args.log_file,"a") as f: f.write(f"{uid},{rid},{act['rating']},{act['action']},{ts}\n")
            if conn: write_to_pg(conn,uid,rid,act["rating"],act["action"])
            met["logged"]+=1
            if delay: time.sleep(delay)
        print(f"    cumulative: {met['requests']} req, {met['ok']} ok, {met['logged']} logged")
    if conn: conn.close()
    print("\n"+"="*60)
    print("DONE: Data Generator Complete")
    print(f"   Requests:  {met['requests']:,}")
    print(f"   Logged:    {met['logged']:,}")
    print(f"   Actions:   cook={met['cook']}, view={met['view']}, skip={met['skip']}")
    print(f"   Log file:  {args.log_file}")
    print("="*60)

if __name__ == "__main__": main()
