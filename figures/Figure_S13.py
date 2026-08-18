"""
Plot a 2x1 comparison figure: top panel = a flash drought event, bottom
panel = a slow-onset drought event, both showing only the soil moisture
(SM) percentile trajectory with S0/S1/S2 marked. No GPP in this figure --
purely an illustration of flash vs. slow drought onset behavior.

Illustration figure for PNAS methods section.

Paths assume this is run on NERSC Perlmutter:
  /global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/
    sm_gpa_percentiles_cluster_1.nc
    drought_events_fixed_season_cluster_1_with_GPP_Zhang2025.csv
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -----------------------------------------------------------------------
# 0. USER SETTINGS
# -----------------------------------------------------------------------
BASE_DIR = "/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist"

CSV_PATH = f"{BASE_DIR}/drought_events_fixed_season_cluster_1_with_GPP_Zhang2025.csv"
SM_NC_PATH = f"{BASE_DIR}/sm_gpa_percentiles_cluster_1.nc"

OUT_FIG = "flash_vs_slow_drought_illustration.png"

YEAR0 = 1981  # first calendar year present in the SM nc file

# Row index override per panel (set to a specific CSV row index to force
# a particular event; leave None to auto-select)
FLASH_ROW_INDEX = None
SLOW_ROW_INDEX = None

# Which auto-selected candidate to use (0 = "best" match, ranked by
# duration). Bump this if a given auto-pick doesn't look clean.
FLASH_CANDIDATE_RANK = 0
SLOW_CANDIDATE_RANK = 0

# Flash droughts: onset (S0->S1) within <=6 pentads (<=30 days) per
# project convention. Slow droughts: onset longer than that.
MAX_FLASH_INTENSIFICATION_PENTADS = 6
MIN_SLOW_INTENSIFICATION_PENTADS = 7

# Moderate (non-extreme) SM percentile bounds for a clean illustration
S0_PCTL_RANGE = (40, 70)   # SM percentile at onset (S0)
S2_PCTL_RANGE = (5, 30)    # event-window minimum SM percentile (used only
                            # as a selection filter, not as S2's plotted value)

# Minimum total event duration (in pentads) for the slow-drought panel,
# so its onset visibly looks gradual compared to the flash panel
MIN_SLOW_TOTAL_PENTADS = 15

# Growing-season pentad index bounds (0-based) corresponding to Apr 1 -
# Oct 31: pentad idx*5+1 = day-of-year, so Apr1 (doy 91) -> idx 18,
# Oct31 (doy 304) -> idx 60. Require S2_index to fall inside this range,
# otherwise the drought-termination point lands outside the plotted
# window and never appears on the figure.
SEASON_PENTAD_MIN = 18
SEASON_PENTAD_MAX = 60


# -----------------------------------------------------------------------
# 1. LOAD CSV AND SELECT ONE FLASH + ONE SLOW EVENT
# -----------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)


def select_event(subset, row_index, candidate_rank, label):
    if row_index is not None:
        return subset.loc[row_index]
    ranked = subset.sort_values("total_event_pentads", ascending=False)
    rank = min(candidate_rank, len(ranked) - 1)
    ev = ranked.iloc[rank]
    print(f"({label}: using candidate rank {rank} of {len(ranked)} matching events)")
    return ev


flash_pool = df[
    (df["drought_type"] == "Flash") &
    (df["intensification_pentads"] <= MAX_FLASH_INTENSIFICATION_PENTADS) &
    (df["SM_at_S0"].between(*S0_PCTL_RANGE)) &
    (df["min_percentile"].between(*S2_PCTL_RANGE)) &
    (df["S2_index"].between(SEASON_PENTAD_MIN, SEASON_PENTAD_MAX))
]
if flash_pool.empty:
    print("No flash event matched moderate SM bounds; relaxing bounds.")
    flash_pool = df[
        (df["drought_type"] == "Flash") &
        (df["intensification_pentads"] <= MAX_FLASH_INTENSIFICATION_PENTADS) &
        (df["S2_index"].between(SEASON_PENTAD_MIN, SEASON_PENTAD_MAX))
    ]

slow_pool = df[
    (df["drought_type"] == "Slow") &
    (df["intensification_pentads"] >= MIN_SLOW_INTENSIFICATION_PENTADS) &
    (df["total_event_pentads"] >= MIN_SLOW_TOTAL_PENTADS) &
    (df["SM_at_S0"].between(*S0_PCTL_RANGE)) &
    (df["min_percentile"].between(*S2_PCTL_RANGE)) &
    (df["S2_index"].between(SEASON_PENTAD_MIN, SEASON_PENTAD_MAX))
]
if slow_pool.empty:
    print("No slow event matched moderate SM bounds; relaxing bounds.")
    slow_pool = df[
        (df["drought_type"] == "Slow") &
        (df["intensification_pentads"] >= MIN_SLOW_INTENSIFICATION_PENTADS) &
        (df["total_event_pentads"] >= MIN_SLOW_TOTAL_PENTADS) &
        (df["S2_index"].between(SEASON_PENTAD_MIN, SEASON_PENTAD_MAX))
    ]

flash_event = select_event(flash_pool, FLASH_ROW_INDEX, FLASH_CANDIDATE_RANK, "FLASH")
slow_event = select_event(slow_pool, SLOW_ROW_INDEX, SLOW_CANDIDATE_RANK, "SLOW")

print("\nFLASH event:")
print(flash_event)
print("\nSLOW event:")
print(slow_event)


# -----------------------------------------------------------------------
# 2. HELPER: EXTRACT SM SERIES + DATES + S0/S1/S2 FOR ONE EVENT
# -----------------------------------------------------------------------
def pentad_idx_to_date(idx, year):
    doy = idx * 5 + 1  # start day-of-year of this pentad, 1-based
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(doy) - 1)


def extract_event_data(event, sm_ds):
    year = int(event["year"])
    year_pos = year - YEAR0
    cell_id_sm = int(event["cell_id_sm"])

    S0_idx = int(event["S0_index"])
    S1_idx = int(event["S1_index"])
    S2_idx = int(event["S2_index"])

    sm_series = sm_ds["sm_percentiles"].isel(cell=cell_id_sm, year=year_pos).values

    S0_pctl = float(event["SM_at_S0"])
    S1_pctl = float(event["SM_at_S1"])
    # S2's plotted value = actual SM series value at S2_index (drought
    # termination pentad), NOT "min_percentile" (which is the event-window
    # minimum and can occur at a different pentad than S2 -- confirmed via
    # diagnostic in the single-event script).
    S2_pctl = float(sm_series[S2_idx])

    # sanity check S0/S1 reproduce the CSV values
    for lbl, idx, expected in [("S0", S0_idx, S0_pctl), ("S1", S1_idx, S1_pctl)]:
        actual = float(sm_series[idx])
        if abs(actual - expected) > 1.0:
            print(f"  WARNING: {lbl} mismatch -- sm_series[{idx}]={actual:.2f} "
                  f"vs CSV={expected:.2f}")

    n_pentad = sm_series.shape[0]
    pentad_axis = np.arange(n_pentad)
    dates = np.array([pentad_idx_to_date(i, year) for i in pentad_axis])

    season_start = pd.Timestamp(year=year, month=4, day=1)
    season_end = pd.Timestamp(year=year, month=10, day=31)
    season_mask = (dates >= season_start) & (dates <= season_end)

    return dict(
        year=year,
        dates_plot=dates[season_mask],
        sm_plot=sm_series[season_mask],
        S0_date=pentad_idx_to_date(S0_idx, year), S0_pctl=S0_pctl,
        S1_date=pentad_idx_to_date(S1_idx, year), S1_pctl=S1_pctl,
        S2_date=pentad_idx_to_date(S2_idx, year), S2_pctl=S2_pctl,
        season_start=season_start, season_end=season_end,
    )


sm_ds = xr.open_dataset(SM_NC_PATH)
flash_data = extract_event_data(flash_event, sm_ds)
slow_data = extract_event_data(slow_event, sm_ds)
sm_ds.close()


# -----------------------------------------------------------------------
# 3. PLOT
# -----------------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"

sm_color = "#A0522D"  # soil-brown
point_marker = "o"

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

panel_info = [
    (axes[0], flash_data, "(a) Flash drought"),
    (axes[1], slow_data, "(b) Slow-onset drought"),
]

for ax, d, panel_label in panel_info:
    ax.plot(d["dates_plot"], d["sm_plot"], color=sm_color, lw=1.8, zorder=3)

    ax.axhline(20, color="gray", ls="--", lw=0.9, zorder=1)
    ax.axhline(40, color="gray", ls="--", lw=0.9, zorder=1)

    s_points = [("S0", d["S0_date"], d["S0_pctl"]),
                ("S1", d["S1_date"], d["S1_pctl"]),
                ("S2", d["S2_date"], d["S2_pctl"])]
    for label, dt, pctl in s_points:
        ax.scatter(dt, pctl, marker=point_marker, color=sm_color,
                   edgecolor="black", zorder=5, s=70)
        ax.annotate(label, (dt, pctl), textcoords="offset points", xytext=(8, 8),
                    ha="left", fontsize=10, color=sm_color, fontweight="bold")

    # step-shaped dashed line S0->S1 (horizontal=time, vertical=percentile change)
    ax.plot([d["S0_date"], d["S1_date"]], [d["S0_pctl"], d["S0_pctl"]],
             color=sm_color, ls="--", lw=1.1, alpha=0.7, zorder=2)
    ax.plot([d["S1_date"], d["S1_date"]], [d["S0_pctl"], d["S1_pctl"]],
             color=sm_color, ls="--", lw=1.1, alpha=0.7, zorder=2)

    # shaded drought onset duration window
    ax.axvspan(d["S0_date"], d["S1_date"], color=sm_color, alpha=0.12, zorder=0)

    ax.set_ylabel("SM percentile")
    ax.set_ylim(0, 100)
    ax.set_xlim(d["season_start"], d["season_end"])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.01, 0.98, panel_label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="left", va="top")

axes[1].set_xlabel("Date (mm/dd)")

# --- shared legend above both panels ---
legend_handles = [
    Line2D([0], [0], color=sm_color, lw=1.8, label="Soil moisture percentile"),
    Patch(facecolor=sm_color, alpha=0.12, label="Drought onset duration"),
    Line2D([0], [0], marker=point_marker, color="none", markerfacecolor=sm_color,
           markeredgecolor="black", markersize=8, label="S0: Drought onset"),
    Line2D([0], [0], marker=point_marker, color="none", markerfacecolor=sm_color,
           markeredgecolor="black", markersize=8, label="S1: Drought onset end"),
    Line2D([0], [0], marker=point_marker, color="none", markerfacecolor=sm_color,
           markeredgecolor="black", markersize=8, label="S2: Drought termination"),
]
fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
           ncol=3, frameon=False, fontsize=9, columnspacing=1.2, handletextpad=0.6)

fig.autofmt_xdate(rotation=0, ha="center")
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
print(f"\nSaved figure to {OUT_FIG}")