# NBA Shooting Quality Model · 2023-24

A shooting efficiency model that decomposes player performance into two components — per-shot skill and total volume contribution — using shot location and creation difficulty data from public NBA tracking endpoints.

---

## TL;DR

Most shooting efficiency metrics describe outcomes but ignore how shots were created or where they came from. This model addresses both gaps. PPSA gives a cleaner per-shot efficiency baseline than TS%. SQS combines a difficulty-adjusted rate stat with a zone-level counting stat to answer two distinct questions simultaneously: how good a shooter is this player, and how much did their shooting actually contribute?

---

## The question

How do you evaluate shooting efficiency in a way that accounts for shot quality — both the difficulty of creation and the value of the locations being targeted?

---

## Metrics

### PPSA — Points Per Scoring Action

A cleaner baseline than TS% and eFG%.

```
PPSA = PTS / (FGA + FTA / 2)
```

**What it fixes:**
- eFG% ignores free throws entirely
- TS% uses a fixed 0.44 approximation in the denominator — a league-average blend constant, not a geometrically correct value
- FTA / 2 reflects that a two-shot foul trip is one scoring action, which is geometrically true rather than statistically approximated
- Units are interpretable: actual points per actual scoring action

PPSA is a standalone module. Run it independently to compare player efficiency without the shot quality layer.

### SQS — Shooting Quality Score

A two-component metric that accounts for shot quality.

**Component 1 — PPSA_adj (rate)**

Difficulty-weighted version of PPSA. Pull-up rate from tracking data is used as a proxy for creation difficulty — a player whose shot diet is heavily pull-up based is working harder per attempt than one who primarily catches and shoots.

```
pull_up_rate      = pu_fga / (cs_fga + pu_fga)
difficulty_scalar = 1 + (pull_up_rate - league_avg_pu_rate) × K
PPSA_adj          = PTS / ((FGA + FTA/2) × difficulty_scalar)
```

A player above league-average pull-up rate gets a scalar > 1, meaning their denominator inflates less — they receive credit for the harder shot diet. A player below average gets a scalar < 1, discounting their efficiency for taking easier shots.

**Component 2 — PAE — Points Above Expected (counting)**

For each of six shot zones, the league-average points per attempt (PPA) is computed from aggregate shot location data. Each player's actual point production from each zone is compared against what league-average shooting would have produced from the same number of attempts.

```
league_ppa_zone = (Σ FGM_zone × point_value) / Σ FGA_zone   [league totals]
PAE             = Σ zones [ (player FGM_zone × pv) - (player FGA_zone × league_ppa_zone) ]
```

Zones: Restricted Area (2pt), Paint Non-RA (2pt), Mid-Range (2pt), Left Corner 3, Right Corner 3, Above the Break 3.

PAE answers: given exactly the shots this player took and from exactly those zones, how many more points did they produce than a league-average shooter would have?

**Combined — SQS**

Both components normalised to z-scores within the qualified sample, then combined equally:

```
SQS = 0.5 × z(PPSA_adj) + 0.5 × z(PAE)
```

SQS > 0: above league average on combined shooting quality.
SQS < 0: below.

The equal weighting is the honest default — there is no prior reason to favour skill over volume contribution without empirical evidence from the data.

---

## What the quadrant map tells you

The PPSA_adj vs PAE scatter surfaces four player archetypes:

| | High PAE | Low PAE |
|---|---|---|
| **High PPSA_adj** | High skill, high volume — SQS leaders | High skill, underutilised — efficient but not enough attempts |
| **Low PPSA_adj** | Volume-driven surplus — shot hunters | Low skill, low contribution |

Players in the high-skill/low-volume quadrant are analytically interesting: they are converting difficult shots at an elite rate but not generating large absolute surplus because their volume is limited. Whether that reflects role constraints or genuine usage ceiling is a question the model flags but cannot answer alone.

---

## Data sources

All pulled from public NBA.com endpoints via `nba_api` with `curl_cffi` for bot detection bypass.

| Source | Endpoint | Used for |
|---|---|---|
| Base + Advanced stats | `leaguedashplayerstats` | PTS, FGA, FTA, e_net_rating |
| Shot locations | `leaguedashplayershotlocations` | Zone FGA/FGM for PAE |
| Catch-and-shoot | `leaguedashplayerptshot` (0 dribbles) | Creation difficulty |
| Pull-up | `leaguedashplayerptshot` (2+ dribbles) | Creation difficulty |

Filter: GP ≥ 20, MIN ≥ 15 per game. Removes garbage-time noise consistent with the original model.

---

## Run order

```bash
pip install -r requirements.txt

python3 data_pull.py   # pulls all sources → data/raw/
python3 ppsa.py        # standalone PPSA module → outputs/
python3 sqs.py         # PPSA_adj + PAE + SQS → outputs/
```

---

## Outputs

```
outputs/
├── ppsa_rankings.csv     # per-player PPSA vs TS% with rank divergences
├── ppsa_analysis.png     # validation + divergence charts
├── sqs_rankings.csv      # per-player SQS, PPSA_adj, PAE, component ranks
└── sqs_analysis.png      # six-panel chart: validation, quadrant map, zone breakdown
```

---

## Validation approach

Both PPSA_adj and PAE are validated against `e_net_rating` — the same target variable used in the companion model ([nba-analytics](https://github.com/babyteej/nba-analytics)). Pearson correlation and partial correlation (controlling for PTS and USG%) are reported. Partial correlation isolates whether efficiency is the driver rather than raw scoring volume.

A metric that correlates with player value and captures something TS%/eFG% were not measuring is the bar this model is held to.

---

## Honest limitations

**PPSA vs TS%:** The correlation between PPSA and TS% is high (~0.97). The denominators differ and PPSA is more interpretable, but the ranking differences are small and concentrated at extreme FTA-rate players. PPSA is a better-constructed metric, not a dramatically different one.

**K_DIFFICULTY is a tuning parameter:** The constant controlling how much pull-up rate moves the PPSA_adj denominator is set at 0.20. This is a reasonable starting point but is not empirically derived. Future work could regress this constant against `e_net_rating` to learn the optimal value from the data.

**Pull-up rate as difficulty proxy:** Using dribbles before shot as creation difficulty ignores shot distance, defender proximity, and touch time — all of which are available from tracking endpoints but not yet incorporated. The current implementation uses pull-up rate as an accessible, defensible first approximation.

**PAE does not capture shot selection quality:** A player can post a positive PAE by taking a high volume of above-the-break 3s and shooting at exactly league average. PAE measures surplus over average from the zones actually used — it does not penalise players for selecting low-value zones in the first place.

---

## File structure

```
nba-sqs/
├── data/
│   └── raw/              # CSVs from data_pull.py
├── outputs/              # charts and rankings
├── data_pull.py          # pulls all four data sources
├── ppsa.py               # standalone PPSA module
├── sqs.py                # PAE + PPSA_adj + SQS
├── requirements.txt
└── README.md
```

---

## Relationship to nba-analytics

This repo extends the work in [nba-analytics](https://github.com/babyteej/nba-analytics). That project found that `e_net_rating` carries approximately 22% team context — meaning individual shooting metrics need to be interpreted with that bias in mind. SQS is a purely individual-level metric and does not correct for team context. The two models are complementary: use the Ridge model's predicted vs actual gap to identify context-inflated or context-depressed players, and use SQS to evaluate the quality of their shooting contribution independently of team outcome.

---

*Built as a portfolio project in quantitative basketball analysis.*
