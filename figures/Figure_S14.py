"""
Plot soil moisture (SM) and GPP percentile time series for a single flash
drought event, marking S0/S1/S2 (SM-based drought phase points) and
G0/G1 (GPP response points; G1 here is the same as "G2" in earlier scripts,
i.e. the GPP minimum point).

Illustration figure for PNAS methods section.

Paths assume this is run on NERSC Perlmutter:
  /global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/
    gpp_percentiles_cluster_1_gridcell_mean.nc
    sm_gpa_percentiles_cluster_1.nc
    drought_events_fixed_season_cluster_1_with_GPP_Zhang2025.csv
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# 0. USER SETTINGS
# -----------------------------------------------------------------------
BASE_DIR = "/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist"

CSV_PATH = f"{BASE_DIR}/drought_events_fixed_season_cluster_1_with_GPP_Zhang2025.csv"
GPP_NC_PATH = f"{BASE_DIR}/gpp_percentiles_cluster_1_gridcell_mean.nc"
SM_NC_PATH = f"{BASE_DIR}/sm_gpa_percentiles_cluster_1.nc"

OUT_FIG = "flash_drought_event_illustration.png"

# First calendar year present in both nc files (35 years: 1981-2015)
YEAR0 = 1981

# How to pick the event to plot. Two options:
#   (A) give a specific row index in the (Flash-filtered) dataframe
#       (use flash.loc[ROW_INDEX], i.e. the original df index shown when
#       you print `event` -- e.g. 32117 from an earlier run)
#   (B) leave ROW_INDEX = None and it will auto-pick a "moderate" flash
#       event: G0_pctl in a mid-high range and G-min (G1/G2) pctl in a
#       mid-low range, so the plot shows a clear but not extreme (99% -> 0%)
#       decline, similar to a typical event rather than a worst-case one
ROW_INDEX = None

# Only consider "true" flash droughts (matches the project convention:
# intensification_pentads <= 6, i.e. <= 30 days onset)
MAX_INTENSIFICATION_PENTADS = 6

# Minimum GPP response duration (G-min index - G0 index, in pentads) for
# an event to be eligible for auto-selection -- avoids picking a
# degenerate near-instant single-step drop as the "illustration" example
MIN_GPP_RESPONSE_DURATION = 4

# Which candidate to use, ranked by longest duration first (0 = longest).
# Increase this to cycle to a different event if a given one doesn't look
# clean once plotted (e.g. data-quality issue at one of the S/G points).
CANDIDATE_RANK = 2

# Bounds for a "moderate" (non-extreme) illustrative event -- avoids the
# 99% -> 0% collapse on either the SM or the GPP side
S0_PCTL_RANGE = (40, 70)   # SM percentile at onset (S0)
S2_PCTL_RANGE = (5, 30)    # SM percentile at the minimum (S2) -- not exactly 0
G0_PCTL_RANGE = (45, 70)   # GPP percentile at onset (G0)
G1_PCTL_RANGE = (10, 30)   # GPP percentile at the minimum (G1/G2)


# -----------------------------------------------------------------------
# 1. LOAD CSV AND SELECT ONE EVENT
# -----------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

flash = df[(df["drought_type"] == "Flash") &
           (df["intensification_pentads"] <= MAX_INTENSIFICATION_PENTADS)].copy()

if ROW_INDEX is not None:
    event = flash.loc[ROW_INDEX]
else:
    # prefer a "moderate" event: enough duration to show curve shape, and
    # both SM and GPP starting/ending in mid-range percentiles (i.e. not
    # an extreme 99% -> 0% collapse on either variable)
    candidates = flash[
        (flash["GPP_response_duration"] >= MIN_GPP_RESPONSE_DURATION) &
        (flash["SM_at_S0"].between(*S0_PCTL_RANGE)) &
        (flash["min_percentile"].between(*S2_PCTL_RANGE)) &
        (flash["G0_pctl"].between(*G0_PCTL_RANGE)) &
        (flash["G2_pctl"].between(*G1_PCTL_RANGE))
    ]
    if candidates.empty:
        print("No event matched the moderate SM/GPP percentile bounds; "
              "relaxing to GPP bounds + duration only.")
        candidates = flash[
            (flash["GPP_response_duration"] >= MIN_GPP_RESPONSE_DURATION) &
            (flash["G0_pctl"].between(*G0_PCTL_RANGE)) &
            (flash["G2_pctl"].between(*G1_PCTL_RANGE))
        ]
    if candidates.empty:
        print("Still no match; relaxing to duration filter only.")
        candidates = flash[flash["GPP_response_duration"] >= MIN_GPP_RESPONSE_DURATION]
    if candidates.empty:
        candidates = flash  # final fallback

    # among candidates, rank by longest duration for the clearest visual
    # curve, then pick the CANDIDATE_RANK'th one (0 = longest)
    ranked = candidates.sort_values("GPP_response_duration", ascending=False)
    rank = min(CANDIDATE_RANK, len(ranked) - 1)
    event = ranked.iloc[rank]
    print(f"(Using candidate rank {rank} of {len(ranked)} matching events)")

print("Selected event:")
print(event)

year = int(event["year"])
year_pos = year - YEAR0  # 0-based position, valid for both nc files

cell_id_sm = int(event["cell_id_sm"])       # 0-based position into SM nc 'cell' dim
gpp_cell_idx = int(event["GPP_cell_idx"])   # 0-based position into GPP nc 'cell' dim

S0_idx = int(event["S0_index"])
S1_idx = int(event["S1_index"])
S2_idx = int(event["S2_index"])
G0_idx = int(event["G0_index"])
G1_idx = int(event["G2_index"])  # NOTE: "G2" in the CSV/older code == "G1" (GPP min) here

S0_pctl = float(event["SM_at_S0"])
S1_pctl = float(event["SM_at_S1"])
# NOTE: "min_percentile" is the minimum SM percentile anywhere during the
# whole drought event (which can occur at a different pentad than S2 --
# confirmed via diagnostic below: S2_index is the drought TERMINATION
# pentad, not necessarily the pentad of the minimum value). So S2's plotted
# value must come directly from the SM series at S2_index, not from
# "min_percentile". This is filled in after sm_series is extracted below.
S2_pctl = None  # placeholder, set after sm_series is loaded
G0_pctl = float(event["G0_pctl"])
G1_pctl = float(event["G2_pctl"])


# -----------------------------------------------------------------------
# 2. OPEN NC FILES AND EXTRACT FULL-YEAR PENTAD SERIES FOR THIS CELL
# -----------------------------------------------------------------------
sm_ds = xr.open_dataset(SM_NC_PATH)
gpp_ds = xr.open_dataset(GPP_NC_PATH)

# --- sanity check: confirm positional indexing assumption is correct ---
# Note: the SM file (sm_gpa_percentiles_cluster_1.nc) has no lat/lon
# variables, so we can only cross-check the GPP side directly. The GPP
# file's lon is in -180/180 convention while the CSV's lat/lon (SM grid)
# is in 0-360, so we convert before comparing.
gpp_lat_check = float(gpp_ds["lat"].isel(cell=gpp_cell_idx).values)
gpp_lon_check = float(gpp_ds["lon"].isel(cell=gpp_cell_idx).values)

tol = 0.01
gpp_ok = (abs(gpp_lat_check - event["GPP_lat"]) < tol and
          abs(gpp_lon_check - event["GPP_lon"]) < tol)

if not gpp_ok:
    print(f"WARNING: GPP cell lat/lon mismatch. "
          f"nc=({gpp_lat_check},{gpp_lon_check}) csv=({event['GPP_lat']},{event['GPP_lon']}). "
          f"Try GPP_cell_idx +/- 1 (0-based vs 1-based indexing).")
else:
    print("GPP cell indexing check passed: nc lat/lon match CSV GPP_lat/GPP_lon.")

# cross-check that cell_id_sm and gpp_cell_idx refer to the same physical
# grid cell, using the CSV's own lat/lon (0-360) vs GPP_lat/GPP_lon (-180/180)
sm_lon_180 = event["lon"] - 360 if event["lon"] > 180 else event["lon"]
same_cell_ok = (abs(event["lat"] - event["GPP_lat"]) < tol and
                abs(sm_lon_180 - event["GPP_lon"]) < tol)
if not same_cell_ok:
    print("WARNING: CSV lat/lon and GPP_lat/GPP_lon do not match for this row "
          "-- cell_id_sm and GPP_cell_idx may not refer to the same location.")
else:
    print("Row-level check passed: cell_id_sm and GPP_cell_idx refer to the same physical grid cell.")

# --- DIAGNOSTIC: verify GPP_cell_idx actually reproduces G0_pctl/G2_pctl ---
# NOTE: Gridcell_Avg_GPP_percentile is stored as a fraction (0-1) in the
# nc file, while G0_pctl in the CSV is on a 0-100 scale -- so we compare
# after multiplying by 100.
gpp_val_at_stored_idx = float(
    gpp_ds["Gridcell_Avg_GPP_percentile"].isel(year=year_pos, pentad=G0_idx, cell=gpp_cell_idx).values
) * 100.0
print(f"\nDIAGNOSTIC: GPP value (x100) at stored GPP_cell_idx={gpp_cell_idx}, "
      f"year_pos={year_pos}, pentad_idx={G0_idx} = {gpp_val_at_stored_idx:.4f} "
      f"(expected G0_pctl = {G0_pctl:.4f})")

if abs(gpp_val_at_stored_idx - G0_pctl) > 1.0:
    print("MISMATCH: stored GPP_cell_idx does not reproduce G0_pctl. "
          "Searching all cells in the GPP file for the one that matches...")
    slice_at_G0 = gpp_ds["Gridcell_Avg_GPP_percentile"].isel(year=year_pos, pentad=G0_idx).values * 100.0
    diffs = np.abs(slice_at_G0 - G0_pctl)
    best_matches = np.argsort(diffs)[:5]
    print("Top 5 candidate cell positions (cell_idx, GPP value, |diff|, lat, lon):")
    for c in best_matches:
        c = int(c)
        clat = float(gpp_ds["lat"].isel(cell=c).values)
        clon = float(gpp_ds["lon"].isel(cell=c).values)
        print(f"  cell={c}, val={slice_at_G0[c]:.4f}, diff={diffs[c]:.4f}, lat={clat}, lon={clon}")
    print(f"CSV says GPP_lat={event['GPP_lat']}, GPP_lon={event['GPP_lon']}")
else:
    print("OK: stored GPP_cell_idx reproduces G0_pctl correctly.")
sm_series = sm_ds["sm_percentiles"].isel(cell=cell_id_sm, year=year_pos).values
gpp_series = gpp_ds["Gridcell_Avg_GPP_percentile"].isel(year=year_pos, cell=gpp_cell_idx).values

# IMPORTANT: Gridcell_Avg_GPP_percentile is stored as a fraction (0-1) in
# the nc file, while G0_pctl/G2_pctl in the CSV are on a 0-100 scale.
# Confirmed via diagnostic: value at G0 pentad = 0.6368, CSV G0_pctl = 63.6752
# (0.6368 * 100 = 63.68). Rescale to 0-100 so it matches the CSV values
# and the SM percentile series (which is already 0-100).
gpp_series = gpp_series * 100.0

n_pentad = sm_series.shape[0]
pentad_axis = np.arange(n_pentad)  # 0-based pentad index, matches S/G indices

# S2's plotted value = the actual SM series value at S2_index (drought
# termination pentad), so the marker sits exactly on the curve.
S2_pctl = float(sm_series[S2_idx])

# For reference/QA: report where the event-wide minimum actually occurred
# (this is what "min_percentile" in the CSV represents -- not necessarily
# at S2_index)
min_idx_actual = int(np.argmin(sm_series[S0_idx:S2_idx + 1])) + S0_idx
print(f"\nNote: CSV min_percentile={event['min_percentile']:.2f} occurs at pentad "
      f"index {min_idx_actual} (within S0-S2 window), while S2_index={S2_idx} is the "
      f"drought termination pentad with SM value={S2_pctl:.2f}.")

sm_ds.close()
gpp_ds.close()

# --- convert pentad index -> calendar date ---
# Pentad p (1-based) covers days [(p-1)*5+1, (p-1)*5+5] of the year.
# For a 0-based index idx, the pentad number is idx+1, so the start day
# of that pentad (day-of-year) is idx*5 + 1. We use the event's actual
# year to place the date correctly relative to Apr 1 / Oct 31 (leap years
# shift day-of-year slightly, so use the real year here rather than a
# fixed reference year).
def pentad_idx_to_date(idx, year):
    doy = idx * 5 + 1  # start day-of-year of this pentad, 1-based
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(doy) - 1)

dates = np.array([pentad_idx_to_date(i, year) for i in pentad_axis])

# --- restrict to the growing season window (Apr 1 - Oct 31) ---
season_start = pd.Timestamp(year=year, month=4, day=1)
season_end = pd.Timestamp(year=year, month=10, day=31)
season_mask = (dates >= season_start) & (dates <= season_end)

dates_plot = dates[season_mask]
sm_plot = sm_series[season_mask]
gpp_plot = gpp_series[season_mask]

S0_date = pentad_idx_to_date(S0_idx, year)
S1_date = pentad_idx_to_date(S1_idx, year)
S2_date = pentad_idx_to_date(S2_idx, year)
G0_date = pentad_idx_to_date(G0_idx, year)
G1_date = pentad_idx_to_date(G1_idx, year)

# --- DIAGNOSTIC: verify S0/S1/S2 indices actually reproduce the CSV's
#     stored SM percentile values in sm_series (same check we did for GPP,
#     which caught the 0-1 vs 0-100 scale bug) ---
print("\nDIAGNOSTIC: checking SM series values at S0/S1/S2 indices against CSV:")
for label, idx, expected in [("S0", S0_idx, S0_pctl), ("S1", S1_idx, S1_pctl), ("S2", S2_idx, S2_pctl)]:
    actual = float(sm_series[idx])
    match = "OK" if abs(actual - expected) < 1.0 else "MISMATCH"
    print(f"  {label}: sm_series[{idx}] = {actual:.4f}, CSV value = {expected:.4f}  -> {match}")
    if match == "MISMATCH":
        diffs = np.abs(sm_series - expected)
        best = np.argsort(diffs)[:5]
        print(f"    nearest matching indices in sm_series: "
              f"{[(int(b), round(float(sm_series[b]), 2)) for b in best]}")


# -----------------------------------------------------------------------
# 3. SLOPE CALCULATION
# -----------------------------------------------------------------------
# GPP_slope = rate of GPP percentile decline from GPP onset (G0) to the
# GPP minimum (G1, i.e. "G2" in the CSV), in percentile points per pentad.
# Confirmed against CSV: GPP_slope = (G2_pctl - G0_pctl) / (G2_index - G0_index)
gpp_slope = (G1_pctl - G0_pctl) / (G1_idx - G0_idx)
print(f"\nComputed GPP_slope = {gpp_slope:.4f} (CSV value = {event['GPP_slope']:.4f})")

# For reference, an analogous SM intensification rate (not in CSV but
# useful for the figure caption / methods text):
sm_intensification_rate = (S2_pctl - S0_pctl) / (S2_idx - S0_idx)
print(f"SM intensification rate S0->S2 = {sm_intensification_rate:.4f} pctl/pentad")


# -----------------------------------------------------------------------
# 4. PLOT
# -----------------------------------------------------------------------
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(8, 5.5))

# soil-brown for SM, vegetation-green for GPP
sm_color = "#A0522D"
gpp_color = "#2D6A4F"

ax.plot(dates_plot, sm_plot, color=sm_color, lw=1.8, zorder=3)
ax.plot(dates_plot, gpp_plot, color=gpp_color, lw=1.8, zorder=3)

# --- reference percentile threshold lines (no text labels) ---
ax.axhline(20, color="gray", ls="--", lw=0.9, zorder=1)
ax.axhline(40, color="gray", ls="--", lw=0.9, zorder=1)

# --- marker style: same marker type (circle) for every point, color
#     distinguishes SM (brown) vs GPP (green) points; text labels placed
#     near each marker identify S0/S1/S2/G0/G1. S-labels go up-right,
#     G-labels go down-left, so they don't collide when points are close
#     together in date/value (as G0/S0 often are near drought onset). ---
point_marker = "o"
s_style = dict(marker=point_marker, color=sm_color, edgecolor="black", zorder=5, s=70)
g_style = dict(marker=point_marker, color=gpp_color, edgecolor="black", zorder=5, s=70)

s_points = [("S0", S0_date, S0_pctl), ("S1", S1_date, S1_pctl), ("S2", S2_date, S2_pctl)]
g_points = [("G0", G0_date, G0_pctl), ("G1", G1_date, G1_pctl)]

for label, d, pctl in s_points:
    ax.scatter(d, pctl, **s_style)
    ax.annotate(label, (d, pctl), textcoords="offset points", xytext=(8, 8),
                ha="left", fontsize=10, color=sm_color, fontweight="bold")

for label, d, pctl in g_points:
    ax.scatter(d, pctl, **g_style)
    ax.annotate(label, (d, pctl), textcoords="offset points", xytext=(-8, -16),
                ha="right", fontsize=10, color=gpp_color, fontweight="bold")

# --- step-shaped (horizontal + vertical) dashed lines illustrating the
#     rate of change: horizontal segment = time change (run), vertical
#     segment = percentile change (rise), for S0->S1 and G0->G1 ---
ax.plot([S0_date, S1_date], [S0_pctl, S0_pctl], color=sm_color, ls="--", lw=1.1, alpha=0.7, zorder=2)
ax.plot([S1_date, S1_date], [S0_pctl, S1_pctl], color=sm_color, ls="--", lw=1.1, alpha=0.7, zorder=2)

ax.plot([G0_date, G1_date], [G0_pctl, G0_pctl], color=gpp_color, ls="--", lw=1.1, alpha=0.7, zorder=2)
ax.plot([G1_date, G1_date], [G0_pctl, G1_pctl], color=gpp_color, ls="--", lw=1.1, alpha=0.7, zorder=2)

# --- shade the flash-drought intensification window (S0 to S1) ---
ax.axvspan(S0_date, S1_date, color=sm_color, alpha=0.12, zorder=0)

ax.set_xlabel("Date (mm/dd)")
ax.set_ylabel("Percentile")
ax.set_ylim(0, 100)
ax.set_xlim(season_start, season_end)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- legend: lines + shaded-window patch + point-meaning entries ---
legend_handles = [
    Line2D([0], [0], color=sm_color, lw=1.8, label="Soil moisture percentile"),
    Line2D([0], [0], color=gpp_color, lw=1.8, label="GPP percentile"),
    Patch(facecolor=sm_color, alpha=0.12, label="Drought onset duration"),
    Line2D([0], [0], marker=point_marker, color="none", markerfacecolor=sm_color,
           markeredgecolor="black", markersize=8, label="S0: Drought onset"),
    Line2D([0], [0], marker=point_marker, color="none", markerfacecolor=sm_color,
           markeredgecolor="black", markersize=8, label="S1: Drought onset end"),
    Line2D([0], [0], marker=point_marker, color="none", markerfacecolor=sm_color,
           markeredgecolor="black", markersize=8, label="S2: Drought termination"),
    Line2D([0], [0], marker=point_marker, color="none", markerfacecolor=gpp_color,
           markeredgecolor="black", markersize=8, label="G0: GPP response onset"),
    Line2D([0], [0], marker=point_marker, color="none", markerfacecolor=gpp_color,
           markeredgecolor="black", markersize=8, label="G1: GPP minimum"),
]
ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
          ncol=4, frameon=False, fontsize=8.5, columnspacing=1.2, handletextpad=0.6)

fig.autofmt_xdate(rotation=0, ha="center")
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
print(f"\nSaved figure to {OUT_FIG}")