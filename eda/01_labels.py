# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "numpy", "matplotlib", "pyarrow", "scikit-learn"]
# ///
"""Label analytics: distributions, per-category structure, the large mode, and sMAPE baselines.

Run:  uv run eda/01_labels.py            (add --refresh to rebuild the index)
Or step through the ``# %%`` cells in VS Code / Jupyter.
"""

# %% Setup
from __future__ import annotations

import sys

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import OUT_DIR, load_index, savefig, smape, smape_per_sample

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 60)

df_all: pd.DataFrame = load_index(refresh="--refresh" in sys.argv)
train: pd.DataFrame = df_all[df_all["split"] == "train"].reset_index(drop=True)

# %% 1. What is in each split
print("== samples per split / source")
print(pd.crosstab(df_all["split"], df_all["source"], margins=True))
print("\n== folder groups per split")
print(pd.crosstab(df_all["group"], df_all["split"], margins=True))
eval_cats = set(df_all.loc[df_all["split"] != "train", "category"])
print(f"\ncategories: train={train['category'].nunique()}  non-train={len(eval_cats)}"
      f"  non-train categories missing from train={sorted(eval_cats - set(train['category']))}")

# %% 2. Target distribution (log scale)
print("\n== volume_cm3 (train)")
print(train["volume_cm3"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(2))
fig, ax = plt.subplots(figsize=(9, 4))
bins = np.arange(-0.5, 4.75, 0.125)
for src, g in train.groupby("source"):
    ax.hist(np.log10(g["volume_cm3"]), bins=bins, alpha=0.6, label=f"{src} (n={len(g)})")
ax.set(xlabel="log10 volume (cm³)", ylabel="count", title="Train volume distribution by source")
ax.legend()
savefig(fig, "01_volume_hist_by_source.png")

fig, ax = plt.subplots(figsize=(9, 4))
order = train.groupby("category_l1")["ln_vol"].median().sort_values().index
ax.boxplot([np.log10(train.loc[train["category_l1"] == c, "volume_cm3"]) for c in order], labels=order, vert=True)
ax.set(ylabel="log10 volume (cm³)", title="Volume by coarse category (category_l1)")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
savefig(fig, "01_volume_by_l1.png")

# %% 3. Per-category table
cat_stats = (
    train.groupby("category")
    .agg(
        n=("volume_cm3", "size"),
        source=("source", lambda s: "/".join(sorted(s.unique()))),
        l1=("category_l1", "first"),
        median=("volume_cm3", "median"),
        min=("volume_cm3", "min"),
        max=("volume_cm3", "max"),
        ln_std=("ln_vol", "std"),
        n_large=("large_mode", "sum"),
    )
    .assign(
        large_rate=lambda d: d["n_large"] / d["n"],
        ln_std_base=train[~train["large_mode"]].groupby("category")["ln_vol"].std(),
    )
    .sort_values("median")
)
cat_stats.to_csv(OUT_DIR / "01_category_stats.csv")
print("\n== objects per category:", cat_stats["n"].describe().round(1).to_dict())
print("categories with <=3 objects:", int((cat_stats["n"] <= 3).sum()))
print("within-category ln-std (base mode only): median",
      round(cat_stats["ln_std_base"].median(), 3), " mean", round(cat_stats["ln_std_base"].mean(), 3))
print("\n== widest categories (base mode)")
print(cat_stats.sort_values("ln_std_base", ascending=False).head(15).round(2))

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(cat_stats["n"], cat_stats["ln_std_base"], s=12)
ax.set(xlabel="objects in category", ylabel="std of ln(volume), base mode", title="Within-category spread vs size")
savefig(fig, "01_category_spread_vs_n.png")

# %% 4. The large (oversized) mode
large = train[train["large_mode"]]
print(f"\n== large mode: {len(large)} objects ({len(large) / len(train):.1%} of train)")
print(pd.crosstab(train["source"], train["large_mode"]))
mixed = cat_stats[(cat_stats["n_large"] > 0)].copy()
base_med = train[~train["large_mode"]].groupby("category")["volume_cm3"].median()
large_med = large.groupby("category")["volume_cm3"].median()
mixed["base_median"] = base_med
mixed["large_median"] = large_med
mixed["ratio"] = (mixed["large_median"] / mixed["base_median"]).round(0)
print(mixed[["n", "n_large", "large_rate", "base_median", "large_median", "ratio"]]
      .sort_values("large_rate", ascending=False).round(2))
print("\nlarge-mode dim_max_mm:", large["dim_max_mm"].describe().round(0).to_dict())
print("base-mode  dim_max_mm:", train.loc[~train["large_mode"], "dim_max_mm"].describe().round(0).to_dict())

# %% 5. Theoretical floor: sMAPE as a function of log-error spread
# For ratio r = pred/true, per-sample sMAPE = 2*tanh(|ln r| / 2).
rng = np.random.default_rng(0)
z = rng.standard_normal(200_000)
sigmas = np.linspace(0.05, 2.0, 40)
floor = [100 * np.mean(2 * np.tanh(np.abs(s * z) / 2)) for s in sigmas]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sigmas, floor)
for s in (0.40, 0.52, 1.89):
    val = 100 * np.mean(2 * np.tanh(np.abs(s * z) / 2))
    ax.axvline(s, ls=":", c="grey")
    ax.annotate(f"σ={s}: {val:.1f}%", (s, val), textcoords="offset points", xytext=(5, -12))
ax.set(xlabel="std of ln(pred/true)", ylabel="expected sMAPE (%)", title="sMAPE floor for a log-normal error")
savefig(fig, "01_smape_floor.png")

# %% 6. Baselines (in-sample and out-of-fold)
y = train["volume_cm3"].to_numpy()


def oof_category_baseline(data: pd.DataFrame, mode_aware: bool, shrink: float, seed: int) -> np.ndarray:
    """Out-of-fold prediction from category statistics only.

    Args:
        data: Train index.
        mode_aware: Predict the large-mode median when the category's large rate > 0.5.
        shrink: Pseudo-count pulling the category median toward the category_l1 median (0 = none).
        seed: Fold seed.

    Returns:
        Predicted volumes aligned with ``data``.
    """
    pred = np.empty(len(data))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(data, data["category"]):
        tr = data.iloc[tr_idx]
        base = tr[~tr["large_mode"]]
        l1_med = base.groupby("category_l1")["ln_vol"].median()
        g_med = base["ln_vol"].median()
        cat_n = base.groupby("category")["ln_vol"].size()
        cat_med = base.groupby("category")["ln_vol"].median()
        large_rate = tr.groupby("category")["large_mode"].mean()
        large_shift = (tr[tr["large_mode"]]["ln_vol"].median() - g_med) if tr["large_mode"].any() else 0.0
        large_cat_med = tr[tr["large_mode"]].groupby("category")["ln_vol"].median()
        for i in va_idx:
            row = data.iloc[i]
            c, l1 = row["category"], row["category_l1"]
            prior = l1_med.get(l1, g_med)
            if c in cat_med:
                n = cat_n[c]
                m = (n * cat_med[c] + shrink * prior) / (n + shrink)
            else:
                m = prior
            if mode_aware and large_rate.get(c, 0.0) > 0.5:
                m = large_cat_med.get(c, m + large_shift)
            elif not mode_aware and c in large_rate.index:
                # plain category median over all objects (both modes mixed)
                m = tr.loc[tr["category"] == c, "ln_vol"].median()
            pred[i] = np.exp(m)
    return pred


rows: list[dict[str, object]] = [
    {"baseline": "global median (in-sample)", "smape": smape(y, np.full_like(y, np.median(y)))},
    {"baseline": "category median (in-sample)",
     "smape": smape(y, train.groupby("category")["volume_cm3"].transform("median"))},
]
for name, kw in {
    "OOF category median (mixed modes)": dict(mode_aware=False, shrink=0.0),
    "OOF base-mode median, mode-aware": dict(mode_aware=True, shrink=0.0),
    "OOF mode-aware + shrink to l1 (k=2)": dict(mode_aware=True, shrink=2.0),
}.items():
    scores = []
    per_sample = np.zeros(len(train))
    for seed in range(3):
        p = oof_category_baseline(train, seed=seed, **kw)
        scores.append(smape(y, p))
        per_sample += smape_per_sample(y, p) / 3
    rows.append({"baseline": name, "smape": np.mean(scores), "smape_std_over_seeds": np.std(scores),
                 "smape_base_mode": per_sample[~train["large_mode"]].mean(),
                 "smape_large_mode": per_sample[train["large_mode"]].mean(),
                 "smape_mf3d": per_sample[train["source"] == "metafood3d"].mean(),
                 "smape_omni": per_sample[train["source"] == "omniobject3d"].mean()})
baselines = pd.DataFrame(rows)
baselines.to_csv(OUT_DIR / "01_baselines.csv", index=False)
print("\n== sMAPE baselines (%)")
print(baselines.round(2).to_string(index=False))

# %% 7. Shape: bbox, fill ratio, density
print("\n== fill ratio (volume / bbox volume):", train["fill_ratio"].describe().round(3).to_dict())
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(train["fill_ratio"], bins=40)
axes[0].set(xlabel="fill ratio", title="Fill ratio")
axes[1].scatter(3 * np.log(train["dim_max_mm"] / 10), train["ln_vol"], s=4,
                c=train["large_mode"].map({False: "tab:blue", True: "tab:red"}))
axes[1].set(xlabel="3·ln(max bbox dim, cm)", ylabel="ln volume", title="Volume vs size³ (red = large mode)")
mf = train[train["density_g_cm3"].notna()]
axes[2].hist(mf["density_g_cm3"].clip(upper=3), bins=40)
axes[2].set(xlabel="weight / volume (g/cm³, clipped at 3)", title=f"Density (MetaFood3D, n={len(mf)})")
savefig(fig, "01_shape_and_density.png")

within_fill = train[~train["large_mode"]].groupby("category")["fill_ratio"].apply(lambda s: np.log(s).std())
print("within-category std of ln(fill): median", round(within_fill.median(), 3),
      "-> upper bound on what a perfect shape model could remove")
print("density by l1 (MetaFood3D):")
print(mf.groupby("category_l1")["density_g_cm3"].describe()[["count", "50%", "min", "max"]].round(2))

# %% 8. Metadata sanity checks
print("\n== gt_confidence x source")
print(pd.crosstab(train["gt_confidence"], train["source"]))
print("mesh decimated:", train["mesh_decimated"].value_counts().to_dict())
print("duplicate sample_ids across splits:", int(df_all["sample_id"].duplicated().sum()))
plt.close("all")
