"""
Explainable Boosting Machine (EBM) for ratio_flash
Target  : ratio_flash
Features: aridity_index -> PCT_CLAY_WTMEAN  (drop T_mean & SM4_10 -- high correlation)
Quality filter: remove grid cells with < 5 total drought events
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from interpret.glassbox import ExplainableBoostingRegressor

# -- 0. Load data -------------------------------------------------------------
df = pd.read_csv("spatial_flash_drought_properties.csv")

# -- 1. Quality filter: remove cells with too few total droughts --------------
df["total_drought"] = df["num_slow"] + df["num_flash"]
before = len(df)
df = df[df["total_drought"] >= 5].copy()
after  = len(df)
print(f"Quality filter: removed {before - after} cells "
      f"({(before - after) / before * 100:.1f}%) with < 5 total droughts")
print(f"Remaining grid cells: {after}\n")

# -- 2. Feature selection -----------------------------------------------------
all_cols = df.columns.tolist()
start = all_cols.index("aridity_index")
end   = all_cols.index("SM4_10")
feature_cols = all_cols[start : end + 1]

DROP = ["T_mean", "SM4_10", "SM4_6", "SM3"]
feature_cols = [c for c in feature_cols if c not in DROP]
print(f"Features used ({len(feature_cols)}):\n  {feature_cols}\n")

TARGET = "ratio_flash"
data = df[feature_cols + [TARGET]].dropna()
X = data[feature_cols]
y = data[TARGET]
print(f"Dataset shape: {X.shape}   |   target range: [{y.min():.3f}, {y.max():.3f}]\n")

# -- 3. Train / test split ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -- 4. Build EBM -------------------------------------------------------------
ebm = ExplainableBoostingRegressor(
    feature_names=feature_cols,
    interactions=25,
    max_bins=1024,
    learning_rate=0.005,
    min_samples_leaf=2,
    max_leaves=7,
    outer_bags=16,
    inner_bags=4,
    random_state=42,
    n_jobs=-1,
)
ebm.fit(X_train, y_train)

# -- 5. Performance -----------------------------------------------------------
y_pred_train = ebm.predict(X_train)
y_pred_test  = ebm.predict(X_test)

r2_train  = r2_score(y_train, y_pred_train)
r2_test   = r2_score(y_test,  y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("=" * 55)
print(f"  Train R2  : {r2_train:.4f}")
print(f"  Test  R2  : {r2_test:.4f}")
print(f"  Test RMSE : {rmse_test:.4f}")
print("=" * 55)

# -- 6. Cross-validated R2 (same hyperparams as final model) ------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_train = []
cv_r2_val   = []
for train_idx, val_idx in kf.split(X):
    m = ExplainableBoostingRegressor(
        feature_names=feature_cols,
        interactions=25,
        max_bins=1024,
        learning_rate=0.005,
        min_samples_leaf=2,
        max_leaves=7,
        outer_bags=16,
        inner_bags=4,
        random_state=42,
        n_jobs=-1,
    )
    m.fit(X.iloc[train_idx], y.iloc[train_idx])
    cv_r2_train.append(r2_score(y.iloc[train_idx], m.predict(X.iloc[train_idx])))
    cv_r2_val.append(r2_score(y.iloc[val_idx],     m.predict(X.iloc[val_idx])))

print(f"\n  5-Fold CV Train R2 : {np.mean(cv_r2_train):.4f} +/- {np.std(cv_r2_train):.4f}")
print(f"  5-Fold CV Val   R2 : {np.mean(cv_r2_val):.4f} +/- {np.std(cv_r2_val):.4f}\n")

# -- 7. Global feature importance ---------------------------------------------
ebm_global = ebm.explain_global(name="EBM - ratio_flash")

term_names  = ebm_global.data()["names"]
term_scores = ebm_global.data()["scores"]

importance_df = (
    pd.DataFrame({"term": term_names, "abs_importance": term_scores})
    .sort_values("abs_importance", ascending=False)
    .reset_index(drop=True)
)
importance_df["relative_pct"] = (
    importance_df["abs_importance"] / importance_df["abs_importance"].sum() * 100
)

print("-- Global term importances ------------------------------")
print(importance_df.to_string(index=False, float_format="%.4f"))
print()

main_df = importance_df[~importance_df["term"].str.contains(" & ", regex=False)].copy()
iact_df = importance_df[ importance_df["term"].str.contains(" & ", regex=False)].copy()

print("-- Main effects (ranked) --------------------------------")
print(main_df.to_string(index=False, float_format="%.4f"))
print()
print("-- Pairwise interactions (ranked) -----------------------")
print(iact_df.to_string(index=False, float_format="%.4f"))
print()
print(f"  Total main effect importance   : {main_df['relative_pct'].sum():.1f}%")
print(f"  Total interaction importance   : {iact_df['relative_pct'].sum():.1f}%")

# -- 8. Figure 1 - bar charts: abs importance + relative % -------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
ax.barh(main_df["term"][::-1], main_df["abs_importance"][::-1], color="#2E86AB")
ax.set_title("Main Effects -- Mean Absolute Score", fontsize=12, fontweight="bold")
ax.set_xlabel("Mean Absolute Score (ratio_flash units)")
ax.tick_params(axis="y", labelsize=9)

ax = axes[0, 1]
ax.barh(main_df["term"][::-1], main_df["relative_pct"][::-1], color="#2E86AB")
ax.set_title("Main Effects -- Relative Importance (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("% Contribution to Total Model Importance")
ax.tick_params(axis="y", labelsize=9)
for i, val in enumerate(main_df["relative_pct"][::-1]):
    ax.text(val + 0.1, i, f"{val:.1f}%", va="center", fontsize=8)

ax = axes[1, 0]
iact_top = iact_df.head(15)
ax.barh(iact_top["term"][::-1], iact_top["abs_importance"][::-1], color="#E84855")
ax.set_title("Top 15 Interactions -- Mean Absolute Score", fontsize=12, fontweight="bold")
ax.set_xlabel("Mean Absolute Score (ratio_flash units)")
ax.tick_params(axis="y", labelsize=8)

ax = axes[1, 1]
ax.barh(iact_top["term"][::-1], iact_top["relative_pct"][::-1], color="#E84855")
ax.set_title("Top 15 Interactions -- Relative Importance (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("% Contribution to Total Model Importance")
ax.tick_params(axis="y", labelsize=8)
for i, val in enumerate(iact_top["relative_pct"][::-1]):
    ax.text(val + 0.02, i, f"{val:.1f}%", va="center", fontsize=8)

plt.suptitle(f"EBM Feature Importance -- ratio_flash  (Test R2={r2_test:.3f})",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("ebm_importances.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved -> ebm_importances.png")

# -- 9. Helper: extract shape function safely ---------------------------------
def get_shape(ebm_global, ebm, feat_name):
    """
    Return (xplot, scores, lower, upper) for a main-effect term.
    Trims names and scores to same length to avoid broadcast errors.
    """
    term_name_list = list(ebm.term_names_)
    idx       = term_name_list.index(feat_name)
    term_data = ebm_global.data(idx)

    scores = np.array(term_data["scores"], dtype=float)
    names  = np.array(term_data["names"])

    n      = min(len(names), len(scores))
    names  = names[:n]
    scores = scores[:n]

    if "upper_bounds" in term_data and term_data["upper_bounds"] is not None:
        upper = np.array(term_data["upper_bounds"], dtype=float)[:n]
        lower = np.array(term_data["lower_bounds"], dtype=float)[:n]
    else:
        upper = scores.copy()
        lower = scores.copy()

    try:
        xplot = names.astype(float)
    except (ValueError, TypeError):
        def parse_right_edge(s):
            s = str(s).strip().rstrip("]").rstrip(")")
            right = s.split(",")[-1].strip()
            try:
                return float(right)
            except ValueError:
                return np.nan
        xplot = np.array([parse_right_edge(nm) for nm in names], dtype=float)

    mask = np.isfinite(xplot) & np.isfinite(scores)
    return xplot[mask], scores[mask], lower[mask], upper[mask]


# -- 10. Shape function grid plotting helper ----------------------------------
def plot_shape_grid(feat_list, main_df, title, filename,
                    ncols=3, color="#2E86AB"):
    """
    Plot shape functions for a list of features in an automatic grid layout.
    """
    n_feats = len(feat_list)
    nrows   = int(np.ceil(n_feats / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(feat_list):
        ax = axes[i]
        try:
            xplot, scores, lower, upper = get_shape(ebm_global, ebm, feat)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(feat, fontsize=9)
            continue

        effect_range  = scores.max() - scores.min()
        rel_pct_vals  = main_df.loc[main_df["term"] == feat, "relative_pct"].values
        rel_pct       = rel_pct_vals[0] if len(rel_pct_vals) > 0 else 0.0
        rank_vals     = main_df[main_df["term"] == feat].index
        rank          = rank_vals[0] + 1 if len(rank_vals) > 0 else "?"

        ax.plot(xplot, scores, color=color, linewidth=2)
        ax.fill_between(xplot, lower, upper, alpha=0.2, color=color)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(f"#{rank}  {feat}\n"
                     f"rel: {rel_pct:.1f}%  |  range: {effect_range:.3f}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel("EBM score", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)

    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {filename}")


# -- 11. Shape function figures -----------------------------------------------
all_main_feats = main_df["term"].tolist()

# All 16 features in one figure (4 columns)
plot_shape_grid(
    feat_list = all_main_feats,
    main_df   = main_df,
    title     = ("EBM Shape Functions -- All Main Effects\n"
                 "(Y-axis: additive contribution to predicted ratio_flash)\n"
                 f"Quality filter: >= 5 total droughts  |  n={after} grid cells"),
    filename  = "ebm_shape_all_main.png",
    ncols     = 4,
    color     = "#2E86AB",
)

# Top 6 only -- clean publication figure (3 columns)
plot_shape_grid(
    feat_list = all_main_feats[:6],
    main_df   = main_df,
    title     = ("EBM Shape Functions -- Top 6 Main Drivers of ratio_flash\n"
                 "(Y-axis: additive contribution to predicted ratio_flash)"),
    filename  = "ebm_shape_top6.png",
    ncols     = 3,
    color     = "#2E86AB",
)

# Remaining features (7 onward) -- supplementary figure
plot_shape_grid(
    feat_list = all_main_feats[6:],
    main_df   = main_df,
    title     = ("EBM Shape Functions -- Remaining Main Effects (rank 7+)\n"
                 "(Y-axis: additive contribution to predicted ratio_flash)"),
    filename  = "ebm_shape_remaining.png",
    ncols     = 4,
    color     = "#5B8C5A",
)

# -- 12. Figure: observed vs. predicted ---------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred_test, alpha=0.4, color="#2E86AB", edgecolors="none", s=20)
lims = [min(y_test.min(), y_pred_test.min()) - 0.02,
        max(y_test.max(), y_pred_test.max()) + 0.02]
ax.plot(lims, lims, "r--", linewidth=1.2, label="1:1 line")
ax.set_xlabel("Observed ratio_flash", fontsize=12)
ax.set_ylabel("Predicted ratio_flash", fontsize=12)
ax.set_title(f"EBM -- Observed vs. Predicted\n"
             f"Test R2={r2_test:.3f}  |  RMSE={rmse_test:.3f}  |  n={after} cells",
             fontsize=12, fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ebm_obs_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> ebm_obs_vs_pred.png")

# -- 13. Save importance table to CSV -----------------------------------------
importance_df.to_csv("ebm_importance_table.csv", index=False, float_format="%.6f")
print("Saved -> ebm_importance_table.csv")

print("\nDone. Files written:")
print("  ebm_importances.png")
print("  ebm_shape_all_main.png      <- all 16 main effects")
print("  ebm_shape_top6.png          <- top 6 publication figure")
print("  ebm_shape_remaining.png     <- rank 7+ supplementary figure")
print("  ebm_obs_vs_pred.png")
print("  ebm_importance_table.csv")
print(f"\nDataset summary:")
print(f"  Original grid cells  : {before}")
print(f"  Removed (< 5 events) : {before - after} ({(before-after)/before*100:.1f}%)")
print(f"  Final grid cells     : {after}")