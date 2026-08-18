import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- Data: term, relative_pct ----
data = [
    ("GS Precip", 8.797610),
    ("GS Solar", 5.993224),
    ("Aridity", 5.743810),
    ("Soil Sand %", 5.113836),
    ("Forest Cover", 4.489470),
    ("T/ET Ratio", 4.285789),
    ("GPP 95th", 4.051611),
    ("Crop Cover", 4.049665),
    ("VPD 95th", 3.938898),
    ("Soil Clay %", 3.832922),
    ("Max Dry Days", 3.549119),
    ("iWUE", 2.880263),
    ("Heatwave Days", 2.742151),
    ("Shrub Cover", 2.688153),
    ("Grass Cover", 2.272192),
    ("Aridity x GPP 95th", 2.206350),
    ("iWUE x T/ET Ratio", 2.124152),
    ("Forest Cover x T/ET Ratio", 2.053607),
    ("GPP 95th x GS Solar", 1.985178),
    ("Aridity x T/ET Ratio", 1.911412),
    ("Aridity x GS Precip", 1.845748),
    ("T/ET Ratio x GS Precip", 1.762727),
    ("Grass Cover x iWUE", 1.717084),
    ("GPP 95th x VPD 95th", 1.701511),
    ("GPP 95th x Grass Cover", 1.549919),
    ("Crop Cover x T/ET Ratio", 1.543666),
    ("GPP 95th x Forest Cover", 1.474240),
    ("Forest Cover x iWUE", 1.446693),
    ("T/ET Ratio x Soil Sand %", 1.369148),
    ("T/ET Ratio x Soil Clay %", 1.312348),
    ("Crop Cover x Max Dry Days", 1.115244),
    ("GS Precip x Soil Clay %", 1.112152),
    ("GPP 95th x Max Dry Days", 1.056989),
    ("Crop Cover x iWUE", 1.042962),
    ("Max Dry Days x Soil Clay %", 1.006150),
    ("Max Dry Days x Soil Sand %", 1.005725),
    ("GPP 95th x Shrub Cover", 0.979119),
    ("Shrub Cover x T/ET Ratio", 0.962225),
    ("GPP 95th x Soil Sand %", 0.753893),
    ("GPP 95th x Soil Clay %", 0.533045),
]

# ---- Category mapping ----
category_map = {
    "Aridity": "Climatology",
    "GS Solar": "Climatology",
    "GS Precip": "Climatology",
    "VPD 95th": "Atmospheric Extremes",
    "Heatwave Days": "Atmospheric Extremes",
    "Max Dry Days": "Atmospheric Extremes",
    "Forest Cover": "Land Cover",
    "Crop Cover": "Land Cover",
    "Shrub Cover": "Land Cover",
    "Grass Cover": "Land Cover",
    "GPP 95th": "Vegetation Biophysics",
    "iWUE": "Vegetation Biophysics",
    "T/ET Ratio": "Vegetation Biophysics",
    "Soil Sand %": "Soil Physics",
    "Soil Clay %": "Soil Physics",
}

color_map = {
    "Climatology": "#8f7fc7",             # purple
    "Atmospheric Extremes": "#4daf4a",    # green
    "Land Cover": "#e34a4a",              # red
    "Vegetation Biophysics": "#f5a623",   # orange
    "Soil Physics": "#3f7fbf",            # blue
    "Pairwise Interaction": "#b3b3b3",    # gray
}

def get_category(term):
    if " x " in term:
        return "Pairwise Interaction"
    return category_map[term]

# Sort descending by relative_pct
data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
terms = [d[0].replace(" x ", " \u00d7 ") for d in data_sorted]
values = [d[1] for d in data_sorted]
cats = [get_category(d[0]) for d in data_sorted]
colors = [color_map[c] for c in cats]

fig, ax = plt.subplots(figsize=(8, 12))

y_pos = range(len(terms))
ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.6, height=0.7)

# Largest at top
ax.set_yticks(y_pos)
ax.set_yticklabels(terms, fontsize=9)
ax.invert_yaxis()

# Value labels
for y, v in zip(y_pos, values):
    ax.text(v + 0.15, y, f"{v:.1f}", va="center", fontsize=8)

ax.set_xlabel("Relative Importance (%)", fontsize=11)
ax.set_xlim(0, max(values) * 1.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

# Legend
legend_order = ["Climatology", "Atmospheric Extremes", "Land Cover",
                 "Vegetation Biophysics", "Soil Physics", "Pairwise Interaction"]
handles = [Patch(facecolor=color_map[c], edgecolor="black", label=c) for c in legend_order]
ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=9, bbox_to_anchor=(0.98, 0.02))

plt.tight_layout()
out_path = "ebm_feature_importance.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)