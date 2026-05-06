"""
data_pull.py
------------
Pulls all data sources required for the NBA shooting quality model.

Sources:
    1. leaguedashplayerstats  (Base + Advanced)  → player_stats.csv
    2. leaguedashplayershotlocations             → shot_locations.csv
    3. leaguedashplayerptshot (0 dribbles)       → catch_shoot.csv
    4. leaguedashplayerptshot (2+ dribbles)      → pull_up.csv

Run: python3 data_pull.py
Next: python3 ppsa.py
      python3 sqs.py
"""

import time
import pandas as pd
from pathlib import Path
from curl_cffi import requests as curl_requests

ROOT = Path(__file__).resolve().parent
RAW  = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SEASON      = "2023-24"
DELAY       = 5.0
RETRY_WAIT  = 15.0
MAX_RETRIES = 4

HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}


# ── core fetch ─────────────────────────────────────────────────────────────────

def fetch(url: str, params: dict) -> pd.DataFrame:
    headers = {k: v for k, v in HEADERS.items() if k != "Host"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = curl_requests.get(
                url, params=params, headers=headers,
                impersonate="chrome110", timeout=90
            )
            if r.status_code != 200:
                print(f"    status {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
            data = r.json()
            cols = data["resultSets"][0]["headers"]
            rows = data["resultSets"][0]["rowSet"]
            return pd.DataFrame(rows, columns=cols)
        except Exception as e:
            print(f"    [attempt {attempt}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)
            else:
                raise


# ── source 1: base + advanced player stats ────────────────────────────────────

def pull_player_stats():
    print(f"\n── Player stats (Base + Advanced)  {SEASON}")
    url = "https://stats.nba.com/stats/leaguedashplayerstats"
    base_params = {
        "LastNGames": "0", "Month": "0", "OpponentTeamID": "0",
        "PaceAdjust": "N", "PerMode": "PerGame", "Period": "0",
        "PlusMinus": "N", "Rank": "N", "Season": SEASON,
        "SeasonType": "Regular Season", "LeagueID": "00",
    }

    trad = fetch(url, {**base_params, "MeasureType": "Base"})
    print(f"  ✓ Base: {len(trad)} players")
    time.sleep(DELAY)

    adv = fetch(url, {**base_params, "MeasureType": "Advanced"})
    print(f"  ✓ Advanced: {len(adv)} players")
    time.sleep(DELAY)

    overlap  = set(trad.columns) & set(adv.columns)
    adv_trim = adv.drop(columns=[c for c in overlap if c != "PLAYER_ID"])
    df = trad.merge(adv_trim, on="PLAYER_ID", how="left")
    df.columns = [c.lower() for c in df.columns]

    skip = {"player_name", "team_abbreviation"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["gp"] >= 20) & (df["min"] >= 15)].copy()
    path = RAW / "player_stats.csv"
    df.to_csv(path, index=False)
    print(f"  ✓ Saved: {path}  ({len(df)} qualifying players)")
    return df


# ── source 2: shot locations by zone ─────────────────────────────────────────

def pull_shot_locations():
    print(f"\n── Shot locations by zone  {SEASON}")
    url = "https://stats.nba.com/stats/leaguedashplayershotlocations"
    params = {
        "College": "", "Conference": "", "Country": "", "DateFrom": "",
        "DateTo": "", "DistanceRange": "By Zone", "Division": "",
        "DraftPick": "", "DraftYear": "", "GameScope": "", "GameSegment": "",
        "Height": "", "LastNGames": "0", "LeagueID": "",
        "Location": "", "MeasureType": "Base", "Month": "0",
        "OpponentTeamID": "0", "Outcome": "", "PORound": "",
        "PaceAdjust": "N", "PerMode": "Totals", "Period": "0",
        "PlayerExperience": "", "PlayerPosition": "", "PlusMinus": "N",
        "Rank": "N", "Season": SEASON, "SeasonSegment": "",
        "SeasonType": "Regular Season", "ShotClockRange": "",
        "StarterBench": "", "TeamID": "", "VsConference": "",
        "VsDivision": "", "Weight": "",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = curl_requests.get(
                url, params=params, headers=HEADERS,
                impersonate="chrome110", timeout=90
            )
            r.raise_for_status()
            data = r.json()

            # resultSets is a dict here, not a list
            rows = data["resultSets"]["rowSet"]

            id_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID",
                       "TEAM_ABBREVIATION", "AGE", "NICKNAME"]
            zones   = ["ra", "paint", "mid", "lc3", "rc3", "ab3", "bc", "corner3"]
            renamed = id_cols.copy()
            for z in zones:
                renamed += [f"{z}_fgm", f"{z}_fga", f"{z}_fg_pct"]

            df = pd.DataFrame(rows, columns=renamed)

            for col in df.columns[6:]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce")

            path = RAW / "shot_locations.csv"
            df.to_csv(path, index=False)
            print(f"  ✓ Saved: {path}  ({len(df)} players)")
            time.sleep(DELAY)
            return df

        except Exception as e:
            print(f"    [attempt {attempt}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)
            else:
                raise


# ── source 3: catch-and-shoot splits ─────────────────────────────────────────
    
def pull_creation_splits():
    print(f"\n── Shot creation splits  {SEASON}")
    url = "https://stats.nba.com/stats/leaguedashplayerptshot"
    base_params = {
        "LeagueID": "00",
        "PerMode": "Totals",
        "Season": SEASON,
        "SeasonType": "Regular Season",
        "DateFrom": "",
        "DateTo": "",
    }

    for dribble_range, label, fname in [
        ("0 Dribbles",  "cs", "catch_shoot.csv"),
        ("2 Dribbles", "pu", "pull_up.csv"),
    ]:
        print(f"  Pulling {dribble_range}...")
        params = {**base_params, "DribbleRange": dribble_range}
        df = fetch(url, params)
        time.sleep(DELAY)

        df = df.rename(columns={
            "FGM":  f"{label}_fgm",
            "FGA":  f"{label}_fga",
            "FG2M": f"{label}_fg2m",
            "FG2A": f"{label}_fg2a",
            "FG3M": f"{label}_fg3m",
            "FG3A": f"{label}_fg3a",
        })

        keep = ["PLAYER_ID", "PLAYER_NAME"] + [c for c in df.columns if c.startswith(label)]
        df   = df[[c for c in keep if c in df.columns]].copy()
        df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce")

        for col in df.columns[2:]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        path = RAW / fname
        df.to_csv(path, index=False)
        print(f"  ✓ Saved: {path}  ({len(df)} players)")




# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Pulling NBA data  ·  Season: {SEASON}")

    pull_player_stats()
    pull_shot_locations()
    pull_creation_splits()

    print("\n✓ All sources saved to data/raw/")
    print("\nNext steps:")
    print("  python3 ppsa.py")
    print("  python3 sqs.py")
