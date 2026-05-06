"""
ppsa.py
-------
Points Per Scoring Action (PPSA) — a standalone shooting efficiency metric.

Addresses the two core limitations of TS% and eFG%:
    - eFG% ignores free throws entirely
    - TS% uses a fixed 0.44 approximation in the denominator rather than
      the geometrically correct 0.5 (one FT pair = one scoring action)

Formula:
    PPSA = PTS / (FGA + FTA / 2)

Outputs:
    outputs/ppsa_rankings.csv   — full player rankings
    outputs/ppsa_analysis.png   — validation + divergence charts

Run after data_pull.py:
    python3 ppsa.py
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


# ── load ──────────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    path = RAW / "player_stats.csv"
    if not path.exists():
        raise FileNotFoundError("data/raw/player_stats.csv not found. Run data_pull.py first.")
    df = pd.read_csv(path)
    print(f"Loaded player_stats.csv: {len(df)} players")
    return df


# ── compute ───────────────────────────────────────────────────────────────────

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    PPSA  = PTS / (FGA + FTA/2)
    TS%   = PTS / (2 * (FGA + 0.44*FTA))   [standard formula, for comparison]
    eFG%  = (FGM + 0.5*FG3M) / FGA         [ignores FTs entirely]
    """
    df = df.copy()
    df["ppsa"]        = df["pts"] / (df["fga"] + df["fta"] / 2)
    df["ts_pct_calc"] = df["pts"] / (2 * (df["fga"] + 0.44 * df["fta"]))
    df["efg_pct"]     = (df["fgm"] + 0.5 * df["fg3m"]) / df["fga"]
    df["fta_rate"]    = df["fta"] / df["fga"]
    return df


# ── validate ──────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame):
    sub = df[["player_name", "ppsa", "ts_pct_calc", "e_net_rating",
              "pts", "usg_pct"]].dropna()

    print(f"\n{'─'*55}")
    print("  VALIDATION: PPSA vs e_net_rating")
    print(f"{'─'*55}")
    print(f"  Sample: {len(sub)} players\n")

    for col, label in [("ppsa", "PPSA"), ("ts_pct_calc", "TS%  ")]:
        r, p = stats.pearsonr(sub[col], sub["e_net_rating"])
        print(f"  {label}   r = {r:.3f}   p = {p:.4f}")

    # Partial correlation — control for volume
    Z = np.column_stack([sub[["pts", "usg_pct"]].values, np.ones(len(sub))])

    def partial_r(x, y, Z):
        xr = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        yr = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        return stats.pearsonr(xr, yr)

    print(f"\n  Partial correlations (controlling for PTS, USG%):")
    for col, label in [("ppsa", "PPSA"), ("ts_pct_calc", "TS%  ")]:
        r, p = partial_r(sub[col].values, sub["e_net_rating"].values, Z)
        print(f"  {label}   r = {r:.3f}   p = {p:.4f}")


# ── rank + divergence ─────────────────────────────────────────────────────────

def rank(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[["player_name", "ppsa", "ts_pct_calc", "efg_pct",
              "fta_rate", "fta", "fga", "pts", "e_net_rating"]].dropna()

    sub = sub.copy()
    sub["rank_ppsa"] = sub["ppsa"].rank(ascending=False).astype(int)
    sub["rank_ts"]   = sub["ts_pct_calc"].rank(ascending=False).astype(int)
    # Positive = ranks higher under PPSA than TS%
    # Negative = ranks lower under PPSA than TS%
    sub["rank_delta"] = sub["rank_ts"] - sub["rank_ppsa"]

    print(f"\n{'─'*55}")
    print("  RANKING DIVERGENCE: PPSA vs TS%")
    print(f"{'─'*55}")
    print("  (+) ranks higher under PPSA  |  (-) ranks lower\n")

    disp = sub.reindex(sub["rank_delta"].abs().sort_values(ascending=False).index)
    print(disp[["player_name", "ppsa", "ts_pct_calc", "fta_rate",
                "rank_ppsa", "rank_ts", "rank_delta"]].head(20).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'─'*55}")
    print("  TOP 20 by PPSA")
    print(f"{'─'*55}")
    print(sub.sort_values("rank_ppsa")[["player_name", "ppsa", "ts_pct_calc",
        "fta_rate", "rank_ppsa", "rank_ts", "rank_delta"]].head(20).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    return sub


# ── plots ─────────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame, ranked: pd.DataFrame):
    sub = df[["ppsa", "ts_pct_calc", "efg_pct",
              "fta_rate", "e_net_rating"]].dropna()

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    AXIS_BG = "#1a1d27"
    C1, C2, C3 = "#4fc3f7", "#f06292", "#81c784"
    TEXT, GRID  = "#e0e0e0", "#2e3244"

    def style(ax, title):
        ax.set_facecolor(AXIS_BG)
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

    # P1: PPSA vs e_net_rating
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(sub["ppsa"], sub["e_net_rating"],
               alpha=0.5, s=18, color=C1, linewidths=0)
    m, b, r, *_ = stats.linregress(sub["ppsa"], sub["e_net_rating"])
    xs = np.linspace(sub["ppsa"].min(), sub["ppsa"].max(), 100)
    ax.plot(xs, m * xs + b, color=C1, linewidth=1.5)
    ax.set_xlabel("PPSA"); ax.set_ylabel("e_net_rating")
    style(ax, f"PPSA vs e_net_rating  (r={r:.3f})")

    # P2: TS% vs e_net_rating
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(sub["ts_pct_calc"], sub["e_net_rating"],
               alpha=0.5, s=18, color=C2, linewidths=0)
    m2, b2, r2, *_ = stats.linregress(sub["ts_pct_calc"], sub["e_net_rating"])
    xs2 = np.linspace(sub["ts_pct_calc"].min(), sub["ts_pct_calc"].max(), 100)
    ax.plot(xs2, m2 * xs2 + b2, color=C2, linewidth=1.5)
    ax.set_xlabel("TS%"); ax.set_ylabel("e_net_rating")
    style(ax, f"TS% vs e_net_rating  (r={r2:.3f})")

    # P3: Rank delta vs FTA rate — where and why they diverge
    ax = fig.add_subplot(gs[1, 0])
    colors = [C1 if v >= 0 else C2 for v in ranked["rank_delta"]]
    ax.scatter(ranked["fta_rate"], ranked["rank_delta"],
               c=colors, alpha=0.5, s=18, linewidths=0)
    ax.axhline(0, color=TEXT, linewidth=0.8, alpha=0.4)
    for _, row in pd.concat([
        ranked.nlargest(4, "rank_delta"),
        ranked.nsmallest(4, "rank_delta")
    ]).iterrows():
        ax.annotate(row["player_name"].split()[-1],
                    (row["fta_rate"], row["rank_delta"]),
                    fontsize=6.5, color=TEXT, alpha=0.85,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("FTA / FGA  (free throw rate)")
    ax.set_ylabel("Rank delta  (TS% rank − PPSA rank)")
    style(ax, "Where PPSA & TS% disagree  →  driven by FT rate")

    # P4: PPSA vs TS% — how correlated are they?
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(sub["ts_pct_calc"], sub["ppsa"],
               alpha=0.5, s=18, color=C3, linewidths=0)
    m4, b4, r4, *_ = stats.linregress(sub["ts_pct_calc"], sub["ppsa"])
    xs4 = np.linspace(sub["ts_pct_calc"].min(), sub["ts_pct_calc"].max(), 100)
    ax.plot(xs4, m4 * xs4 + b4, color=C3, linewidth=1.5)
    ax.set_xlabel("TS%"); ax.set_ylabel("PPSA")
    style(ax, f"PPSA vs TS%  (r={r4:.3f})  — not identical")

    fig.suptitle("Points Per Scoring Action (PPSA)  ·  2023-24 NBA",
                 color=TEXT, fontsize=14, fontweight="bold", y=0.98)

    path = OUT / "ppsa_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✓ Chart saved: {path}")


# ── save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, ranked: pd.DataFrame):
    out_cols = ["player_name", "team_abbreviation", "gp", "min",
                "pts", "fga", "fta", "ppsa", "ts_pct_calc",
                "efg_pct", "fta_rate", "e_net_rating"]
    out_cols = [c for c in out_cols if c in df.columns]

    out = df[out_cols].dropna(subset=["ppsa"]).copy()
    out["rank_ppsa"]  = out["ppsa"].rank(ascending=False).astype(int)
    out["rank_ts"]    = out["ts_pct_calc"].rank(ascending=False).astype(int)
    out["rank_delta"] = out["rank_ts"] - out["rank_ppsa"]
    out = out.sort_values("rank_ppsa")

    path = OUT / "ppsa_rankings.csv"
    out.to_csv(path, index=False)
    print(f"  ✓ Rankings saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df     = load()
    df     = compute(df)
    validate(df)
    ranked = rank(df)
    plot(df, ranked)
    save(df, ranked)

    print("\n✓ Done.")
    print("  outputs/ppsa_analysis.png")
    print("  outputs/ppsa_rankings.csv")
    print("\nNext: python3 sqs.py")
