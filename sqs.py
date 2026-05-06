"""
sqs.py
------
Shooting Quality Score (SQS) — combines two components to evaluate
shooting efficiency adjusted for shot quality.

    PPSA_adj  — difficulty-weighted rate stat
                How good a shooter is this player, given how hard
                their shots are to make?

    PAE       — Points Above Expected (counting stat)
                How much total value did this player's shooting generate
                beyond what league-average shooting from the same zones
                would have produced?

    SQS       = 0.5 * z(PPSA_adj) + 0.5 * z(PAE)

Methodology:
    Difficulty is derived from pull-up rate (leaguedashplayerptshot):
        pull-up rate = pu_fga / (cs_fga + pu_fga)
        difficulty_scalar = 1 + (pull_up_rate - league_avg_pu_rate) * K

    Expected points per zone come from league-average PPA per zone
    (leaguedashplayershotlocations):
        PAE = Σ zones [ actual_pts_zone - (player_fga_zone * league_ppa_zone) ]

Outputs:
    outputs/sqs_rankings.csv
    outputs/sqs_analysis.png

Run after data_pull.py:
    python3 sqs.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW  = ROOT / "data" / "raw"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Difficulty scalar constant.
# Controls how much pull-up rate moves the PPSA_adj denominator.
# 0.20 means a player 1 std above average pull-up rate gets ~10% difficulty credit.
K_DIFFICULTY = 0.20

# Zones and their point values
ZONES = {
    "ra":      2, # restricted area
    "paint":   2, # paint non-RA
    "mid":     2, # mid-range
    "lc3":     3, # left corner 3
    "rc3":     3, # right corner 3
    "ab3":     3, # above the break 3
    "corner3": 3, # corner 3
}
    # backcourt excluded — near-zero volume, noise


# ── load ──────────────────────────────────────────────────────────────────────

def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "player_stats.csv":  "Run data_pull.py first.",
        "shot_locations.csv":"Run data_pull.py first.",
        "catch_shoot.csv":   "Run data_pull.py first.",
        "pull_up.csv":       "Run data_pull.py first.",
    }
    for fname, msg in required.items():
        if not (RAW / fname).exists():
            raise FileNotFoundError(f"data/raw/{fname} not found. {msg}")

    base      = pd.read_csv(RAW / "player_stats.csv")
    shot_loc  = pd.read_csv(RAW / "shot_locations.csv")
    catch_sh  = pd.read_csv(RAW / "catch_shoot.csv")
    pull_up   = pd.read_csv(RAW / "pull_up.csv")

    print(f"Loaded:")
    print(f"  player_stats.csv   {len(base)} players")
    print(f"  shot_locations.csv {len(shot_loc)} players")
    print(f"  catch_shoot.csv    {len(catch_sh)} players")
    print(f"  pull_up.csv        {len(pull_up)} players")

    return base, shot_loc, catch_sh, pull_up


# ── PAE ───────────────────────────────────────────────────────────────────────

def compute_pae(shot_loc: pd.DataFrame) -> pd.DataFrame:
    """
    League-average points per attempt (PPA) per zone:
        league_ppa_zone = (Σ FGM_zone * point_value) / Σ FGA_zone

    Per-player PAE:
        PAE = Σ zones [ (player_fgm_zone * pv) - (player_fga_zone * league_ppa_zone) ]

    A positive PAE means the player produced more points from their actual
    shot distribution than a league-average shooter would have.
    """
    shot_loc = shot_loc.copy()

    # Compute league-average PPA per zone
    league_ppa = {}
    for z, pv in ZONES.items():
        total_fgm = shot_loc[f"{z}_fgm"].sum()
        total_fga = shot_loc[f"{z}_fga"].sum()
        league_ppa[z] = (total_fgm * pv) / total_fga if total_fga > 0 else 0

    print(f"\n{'─'*50}")
    print("  League-average PPA by zone:")
    for z, ppa in league_ppa.items():
        print(f"    {z:8s}  {ppa:.3f} pts/attempt")

    # Per-player PAE + zone-level breakdown
    pae_vals = []
    zone_pae = {z: [] for z in ZONES}

    for _, row in shot_loc.iterrows():
        pae = 0.0
        for z, pv in ZONES.items():
            actual   = row[f"{z}_fgm"] * pv
            expected = row[f"{z}_fga"] * league_ppa[z]
            delta    = actual - expected
            pae += delta
            zone_pae[z].append(delta)
        pae_vals.append(pae)

    shot_loc["pae"] = pae_vals
    for z in ZONES:
        shot_loc[f"{z}_pae"] = zone_pae[z]

    return shot_loc, league_ppa


# ── PPSA_adj ──────────────────────────────────────────────────────────────────

def compute_ppsa_adj(base: pd.DataFrame,
                     catch_sh: pd.DataFrame,
                     pull_up: pd.DataFrame) -> pd.DataFrame:
    """
    Merge catch-and-shoot and pull-up FGA counts onto base stats.

    pull_up_rate = pu_fga / (cs_fga + pu_fga)

    difficulty_scalar = 1 + (pull_up_rate - league_avg_pu_rate) * K_DIFFICULTY
        > 1 for players above-average pull-up rate (harder shots)
        < 1 for players below-average pull-up rate (easier shots)

    PPSA_adj = PTS / ((FGA + FTA/2) * difficulty_scalar)

    Higher pull-up rate inflates the denominator less — the player gets
    credit for taking harder shots. Lower pull-up rate deflates it — the
    player's efficiency is discounted for taking easier shots.
    """
    creation = catch_sh.merge(
        pull_up[["PLAYER_ID", "pu_fga", "pu_fgm",
                 "pu_fg3a", "pu_fg3m"]],
        on="PLAYER_ID", how="outer"
    ).fillna(0)

    df = base.merge(
        creation[["PLAYER_ID", "cs_fga", "cs_fgm",
                  "pu_fga", "pu_fgm"]],
        left_on="player_id", right_on="PLAYER_ID", how="left"
    ).drop(columns=["PLAYER_ID"])

    df["cs_fga"] = df["cs_fga"].fillna(0)
    df["pu_fga"] = df["pu_fga"].fillna(0)

    total_creation = df["cs_fga"] + df["pu_fga"]
    df["pull_up_rate"] = np.where(
        total_creation > 0,
        df["pu_fga"] / total_creation,
        np.nan
    )

    league_avg_pu_rate = (
        df["pu_fga"].sum() /
        (df["cs_fga"].sum() + df["pu_fga"].sum())
    )

    print(f"\n  League-average pull-up rate: {league_avg_pu_rate:.3f}")
    print(f"  K_DIFFICULTY: {K_DIFFICULTY}")

    df["difficulty_scalar"] = (
        1 + (df["pull_up_rate"] - league_avg_pu_rate) * K_DIFFICULTY
    ).fillna(1.0)

    df["ppsa"]     = df["pts"] / (df["fga"] + df["fta"] / 2)
    denom          = (df["fga"] + df["fta"] / 2) * df["difficulty_scalar"]
    df["ppsa_adj"] = df["pts"] / denom

    return df


# ── SQS ───────────────────────────────────────────────────────────────────────

def compute_sqs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise PPSA_adj and PAE to z-scores within the qualified sample,
    then combine 50/50.

    SQS > 0: above league average on combined shooting quality
    SQS < 0: below league average
    """
    valid = df["ppsa_adj"].notna() & df["pae"].notna()
    df.loc[valid, "z_ppsa_adj"] = stats.zscore(df.loc[valid, "ppsa_adj"])
    df.loc[valid, "z_pae"]      = stats.zscore(df.loc[valid, "pae"])
    df.loc[valid, "sqs"]        = (
        0.5 * df.loc[valid, "z_ppsa_adj"] +
        0.5 * df.loc[valid, "z_pae"]
    )
    return df


# ── validate ──────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame):
    sub = df[["player_name", "sqs", "ppsa_adj", "pae",
              "e_net_rating", "pts", "usg_pct"]].dropna()

    print(f"\n{'─'*55}")
    print("  VALIDATION vs e_net_rating")
    print(f"{'─'*55}")
    print(f"  Sample: {len(sub)} players\n")

    for col, label in [("sqs","SQS     "), ("ppsa_adj","PPSA_adj"),
                       ("pae","PAE     ")]:
        r, p = stats.pearsonr(sub[col], sub["e_net_rating"])
        print(f"  {label}  r = {r:.3f}   p = {p:.4f}")

    Z = np.column_stack([sub[["pts", "usg_pct"]].values, np.ones(len(sub))])

    def partial_r(x, y, Z):
        xr = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        yr = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        return stats.pearsonr(xr, yr)

    print(f"\n  Partial correlations (controlling for PTS, USG%):")
    for col, label in [("sqs","SQS     "), ("ppsa_adj","PPSA_adj"),
                       ("pae","PAE     ")]:
        r, p = partial_r(sub[col].values, sub["e_net_rating"].values, Z)
        print(f"  {label}  r = {r:.3f}   p = {p:.4f}")


# ── surface findings ──────────────────────────────────────────────────────────

def surface_findings(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["sqs", "ppsa_adj", "pae"]).copy()
    sub["rank_sqs"]      = sub["sqs"].rank(ascending=False).astype(int)
    sub["rank_ppsa_adj"] = sub["ppsa_adj"].rank(ascending=False).astype(int)
    sub["rank_pae"]      = sub["pae"].rank(ascending=False).astype(int)

    print(f"\n{'─'*55}")
    print("  TOP 20 by SQS")
    print(f"{'─'*55}")
    cols = ["player_name", "sqs", "ppsa_adj", "pae", "pull_up_rate",
            "rank_sqs", "rank_ppsa_adj", "rank_pae", "e_net_rating"]
    print(sub.sort_values("rank_sqs")[cols].head(20).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'─'*55}")
    print("  HIGH SKILL / LOW VOLUME")
    print("  Efficient on hard shots — underutilised")
    print(f"{'─'*55}")
    mask = (sub["z_ppsa_adj"] > 0.8) & (sub["z_pae"] < 0)
    print(sub[mask].sort_values("ppsa_adj", ascending=False)[
        ["player_name", "ppsa_adj", "pae", "pull_up_rate",
         "rank_ppsa_adj", "rank_pae"]
    ].head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'─'*55}")
    print("  HIGH VOLUME / LOWER SKILL")
    print("  Generating surplus through volume, not per-shot efficiency")
    print(f"{'─'*55}")
    mask = (sub["z_pae"] > 0.8) & (sub["z_ppsa_adj"] < 0.5)
    print(sub[mask].sort_values("pae", ascending=False)[
        ["player_name", "ppsa_adj", "pae", "pull_up_rate",
         "rank_ppsa_adj", "rank_pae"]
    ].head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return sub


# ── plots ─────────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame):
    sub = df.dropna(subset=["sqs", "ppsa_adj", "pae",
                             "e_net_rating", "pull_up_rate"])

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

    AXIS_BG = "#1a1d27"
    C1, C2, C3, C4 = "#4fc3f7", "#f06292", "#81c784", "#ffb74d"
    TEXT, GRID = "#e0e0e0", "#2e3244"

    def style(ax, title):
        ax.set_facecolor(AXIS_BG)
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=9)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

    def annotate_top(ax, xcol, ycol, n=5):
        for _, row in sub.nlargest(n, "e_net_rating").iterrows():
            ax.annotate(
                row["player_name"].split()[-1],
                (row[xcol], row[ycol]),
                fontsize=6.5, color=TEXT, alpha=0.85,
                xytext=(4, 2), textcoords="offset points"
            )

    # P1: SQS vs e_net_rating
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(sub["sqs"], sub["e_net_rating"],
               alpha=0.5, s=18, color=C1, linewidths=0)
    m, b, r, *_ = stats.linregress(sub["sqs"], sub["e_net_rating"])
    xs = np.linspace(sub["sqs"].min(), sub["sqs"].max(), 100)
    ax.plot(xs, m * xs + b, color=C1, lw=1.5)
    annotate_top(ax, "sqs", "e_net_rating")
    ax.set_xlabel("SQS"); ax.set_ylabel("e_net_rating")
    style(ax, f"SQS vs e_net_rating  (r={r:.3f})")

    # P2: PPSA_adj vs e_net_rating
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(sub["ppsa_adj"], sub["e_net_rating"],
               alpha=0.5, s=18, color=C2, linewidths=0)
    m, b, r, *_ = stats.linregress(sub["ppsa_adj"], sub["e_net_rating"])
    xs = np.linspace(sub["ppsa_adj"].min(), sub["ppsa_adj"].max(), 100)
    ax.plot(xs, m * xs + b, color=C2, lw=1.5)
    annotate_top(ax, "ppsa_adj", "e_net_rating")
    ax.set_xlabel("PPSA_adj"); ax.set_ylabel("e_net_rating")
    style(ax, f"PPSA_adj vs e_net_rating  (r={r:.3f})")

    # P3: PAE vs e_net_rating
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(sub["pae"], sub["e_net_rating"],
               alpha=0.5, s=18, color=C3, linewidths=0)
    m, b, r, *_ = stats.linregress(sub["pae"], sub["e_net_rating"])
    xs = np.linspace(sub["pae"].min(), sub["pae"].max(), 100)
    ax.plot(xs, m * xs + b, color=C3, lw=1.5)
    annotate_top(ax, "pae", "e_net_rating")
    ax.set_xlabel("PAE  (pts above expected)"); ax.set_ylabel("e_net_rating")
    style(ax, f"PAE vs e_net_rating  (r={r:.3f})")

    # P4: Quadrant map — skill vs volume contribution
    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(sub["ppsa_adj"], sub["pae"],
                    c=sub["sqs"], cmap="coolwarm",
                    alpha=0.65, s=22, linewidths=0)
    ax.axvline(sub["ppsa_adj"].median(), color=TEXT, lw=0.8,
               alpha=0.35, ls="--")
    ax.axhline(sub["pae"].median(), color=TEXT, lw=0.8,
               alpha=0.35, ls="--")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("SQS", color=TEXT)
    cb.ax.yaxis.label.set_color(TEXT)
    cb.ax.tick_params(colors=TEXT)
    for txt, xy in [
        ("High skill\nHigh volume", (0.75, 0.88)),
        ("High skill\nLow volume",  (0.75, 0.12)),
        ("Low skill\nHigh volume",  (0.12, 0.88)),
        ("Low skill\nLow volume",   (0.12, 0.12)),
    ]:
        ax.text(*xy, txt, transform=ax.transAxes, fontsize=6.5,
                color=TEXT, alpha=0.5, ha="center")
    ax.set_xlabel("PPSA_adj  (skill)")
    ax.set_ylabel("PAE  (volume contribution)")
    style(ax, "Skill vs Volume — SQS quadrant map")

    # P5: Pull-up rate vs PPSA_adj — does creation difficulty predict efficiency?
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(sub["pull_up_rate"], sub["ppsa_adj"],
               alpha=0.5, s=18, color=C4, linewidths=0)
    valid = sub["pull_up_rate"].notna()
    m, b, r, *_ = stats.linregress(
        sub.loc[valid, "pull_up_rate"],
        sub.loc[valid, "ppsa_adj"]
    )
    xs = np.linspace(sub["pull_up_rate"].min(), sub["pull_up_rate"].max(), 100)
    ax.plot(xs, m * xs + b, color=C4, lw=1.5)
    ax.set_xlabel("Pull-up rate  (creation difficulty)")
    ax.set_ylabel("PPSA_adj")
    style(ax, f"Creation difficulty vs PPSA_adj  (r={r:.3f})")

    # P6: Zone PAE breakdown — top 10 by SQS
    ax = fig.add_subplot(gs[1, 2])
    zone_pae_cols = [f"{z}_pae" for z in ZONES if f"{z}_pae" in sub.columns]
    if zone_pae_cols:
        top10 = sub.nlargest(10, "sqs")[["player_name"] + zone_pae_cols]
        top10 = top10.set_index("player_name")[zone_pae_cols]
        top10.plot(kind="barh", stacked=True, ax=ax,
                   color=[C1, C2, C3, C4, "#ce93d8", "#80cbc4"])
        ax.set_xlabel("PAE by zone")
        ax.legend(fontsize=6, loc="lower right",
                  labelcolor=TEXT, facecolor=AXIS_BG, edgecolor=GRID)
        ax.tick_params(colors=TEXT, labelsize=7)
    else:
        ax.text(0.5, 0.5, "Zone PAE breakdown",
                ha="center", va="center", color=TEXT,
                transform=ax.transAxes)
    style(ax, "Top 10 SQS — PAE by zone")

    fig.suptitle("Shooting Quality Score (SQS)  ·  2023-24 NBA",
                 color=TEXT, fontsize=14, fontweight="bold", y=0.98)

    path = OUT / "sqs_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✓ Chart saved: {path}")


# ── save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame):
    cols = ["player_name", "team_abbreviation", "gp", "min", "pts",
            "fga", "fta", "pull_up_rate", "difficulty_scalar",
            "ppsa", "ppsa_adj", "pae", "sqs",
            "z_ppsa_adj", "z_pae", "e_net_rating"]
    cols = [c for c in cols if c in df.columns]

    out = df[cols].dropna(subset=["sqs"]).sort_values(
        "sqs", ascending=False).copy()
    out["rank_sqs"]      = range(1, len(out) + 1)
    out["rank_ppsa_adj"] = out["ppsa_adj"].rank(ascending=False).astype(int)
    out["rank_pae"]      = out["pae"].rank(ascending=False).astype(int)

    path = OUT / "sqs_rankings.csv"
    out.to_csv(path, index=False)
    print(f"  ✓ Rankings saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base, shot_loc, catch_sh, pull_up = load()

    print("\n── Computing PAE...")
    shot_loc, league_ppa = compute_pae(shot_loc)

    print("\n── Computing PPSA_adj...")
    df = compute_ppsa_adj(base, catch_sh, pull_up)

    print("\n── Merging zone data...")
    zone_cols = ["pae"] + [f"{z}_pae" for z in ZONES]
    df = df.merge(
        shot_loc[["PLAYER_ID"] + zone_cols],
        left_on="player_id", right_on="PLAYER_ID", how="left"
    ).drop(columns=["PLAYER_ID"])

    print("\n── Computing SQS...")
    df = compute_sqs(df)

    validate(df)
    ranked = surface_findings(df)
    plot(ranked)
    save(ranked)

    print("\n✓ Done.")
    print("  outputs/sqs_analysis.png")
    print("  outputs/sqs_rankings.csv")
