# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""B3: B2-style cluster values + image corrections from frozen DINOv2-L features.

Stages (all fitted inside each training fold):
  [2] reference table: per category, 1-2 volume clusters (normal / oversized), each with its sMAPE-best value
  [3] ridge on PCA(256) features -> r = ln(true) - ln(cluster value)       (size correction)
  [4] logistic on the same features -> P(oversized), used only in 2-cluster categories (mode choice)
  [5] volume = chosen cluster value * exp(r_hat)

Also runs within-category permutation tests for [3] and [4].
Needs features/dinov2l_reg_518.npz (run features/extract_dinov2.py first).

Run:  uv run baselines/b3.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from b0_b1 import PROJECT, load_split, smape, write_submission
from b2 import GAP, GRID, add_l1

FEATURES = PROJECT / "features" / "dinov2l_reg_518.npz"
SEEDS: tuple[int, ...] = (0, 1, 2)
N_PCA: int = 256
R_CLIP: float = 1.5
N_PERM: int = 30


# ---------------------------------------------------------------- [2] reference table
def cluster_labels(df: pd.DataFrame) -> np.ndarray:
    """u=1 for oversized objects: above a >=30x gap in their category, or whole categories >=30x above l1."""
    u = np.zeros(len(df), dtype=int)
    ln = df["ln"].to_numpy()
    for _, idx in df.groupby("category").indices.items():
        z = np.sort(ln[idx])
        if len(z) > 1 and np.diff(z).max() >= GAP:
            cut = z[np.argmax(np.diff(z)) + 1]
            u[idx[ln[idx] >= cut]] = 1
    base = pd.Series(ln[u == 0]).groupby(df["l1"].to_numpy()[u == 0]).median()
    cat_med = df.groupby("category")["ln"].median()
    cat_l1 = df.groupby("category")["l1"].first()
    all_large = set(cat_med.index[(cat_med - cat_l1.map(base)) >= GAP])
    u[df["category"].isin(all_large).to_numpy()] = 1
    return u


def argmin_t(z: np.ndarray) -> float:
    """sMAPE-best ln value for a set of ln volumes."""
    return float(GRID[np.argmin(np.tanh(np.abs(GRID[:, None] - z[None, :]) / 2).sum(axis=1))])


def loo_argmin(z: np.ndarray) -> np.ndarray:
    """sMAPE-best ln value for each element, computed without that element (needs len >= 2)."""
    loss = np.tanh(np.abs(GRID[:, None] - z[None, :]) / 2)
    total = loss.sum(axis=1, keepdims=True)
    return GRID[np.argmin(total - loss, axis=0)]


@dataclass
class Table:
    """Per-category cluster values (ln cm3) and the fallback for unseen categories."""

    values: dict[str, dict[int, float]]
    fallback: float


def build_table(df: pd.DataFrame, u: np.ndarray) -> tuple[Table, np.ndarray]:
    """Cluster values + leave-one-out reference for every training row (NaN for singleton clusters)."""
    ln = df["ln"].to_numpy()
    values: dict[str, dict[int, float]] = {}
    ref_loo = np.full(len(df), np.nan)
    for (cat, mode), idx in df.assign(u=u).groupby(["category", "u"]).indices.items():
        values.setdefault(cat, {})[mode] = argmin_t(ln[idx])
        if len(idx) >= 2:
            ref_loo[idx] = loo_argmin(ln[idx])
    return Table(values, argmin_t(ln)), ref_loo


# ---------------------------------------------------------------- fold machinery
@dataclass
class Fold:
    tr: np.ndarray
    va: np.ndarray
    Z_tr: np.ndarray
    Z_va: np.ndarray


def make_folds(train: pd.DataFrame, X: np.ndarray, seed: int) -> list[Fold]:
    """Stratified 5-fold split with standardise + PCA fitted on the training part only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(train, train["category"]))
    folds = []
    for tr, va in splits:
        scaler = StandardScaler().fit(X[tr])
        pca = PCA(N_PCA, random_state=0).fit(scaler.transform(X[tr]))
        folds.append(Fold(tr, va, pca.transform(scaler.transform(X[tr])), pca.transform(scaler.transform(X[va]))))
    return folds


@dataclass
class FoldOutput:
    """Everything needed to score one fold's held-out rows."""

    ref_major: np.ndarray   # value of the category's majority cluster (no image) = B2-like
    ref_normal: np.ndarray  # normal-cluster value (NaN if absent)
    ref_large: np.ndarray   # oversized-cluster value (NaN if absent)
    mixed: np.ndarray       # category has both clusters in this training fold
    r_hat: np.ndarray       # ridge size correction
    p_large: np.ndarray     # classifier P(oversized), NaN where not mixed


def run_fold(train: pd.DataFrame, f: Fold, rng: np.random.Generator | None = None) -> FoldOutput:
    """Fit stages [2]-[4] on f.tr, predict f.va. With rng, targets are shuffled within category (null)."""
    T, Q = train.iloc[f.tr], train.iloc[f.va]
    u_tr = cluster_labels(T)
    table, ref_loo = build_table(T, u_tr)

    # [3] ridge on residuals vs the leave-one-out cluster value
    r = T["ln"].to_numpy() - ref_loo
    ok = np.isfinite(r) & (np.abs(r) < GAP)
    cats_tr = T["category"].to_numpy()
    if rng is not None:
        r = shuffle_within(r, cats_tr, ok, rng)
    ridge = RidgeCV(alphas=np.logspace(0, 5, 21)).fit(f.Z_tr[ok], r[ok])
    r_hat = np.clip(ridge.predict(f.Z_va), -R_CLIP, R_CLIP)

    # [4] mode classifier, trained on rows of categories that have both clusters
    mixed_cats = {c for c, v in table.values.items() if len(v) == 2}
    m_tr = np.isin(cats_tr, list(mixed_cats))
    u_fit = shuffle_within(u_tr.astype(float), cats_tr, m_tr, rng).astype(int) if rng is not None else u_tr
    q_cats = Q["category"].to_numpy()
    mixed = np.isin(q_cats, list(mixed_cats))
    p_large = np.full(len(Q), np.nan)
    if m_tr.sum() >= 20 and len(set(u_fit[m_tr])) == 2 and mixed.any():
        clf = LogisticRegression(C=0.05, max_iter=5000).fit(f.Z_tr[m_tr], u_fit[m_tr])
        p_large[mixed] = clf.predict_proba(f.Z_va[mixed])[:, 1]

    counts = pd.Series(u_tr).groupby(cats_tr).agg(["mean"])["mean"]
    ref_major, ref_normal, ref_large = (np.full(len(Q), np.nan) for _ in range(3))
    for i, c in enumerate(q_cats):
        v = table.values.get(c)
        if v is None:
            ref_major[i] = table.fallback
            continue
        ref_normal[i], ref_large[i] = v.get(0, np.nan), v.get(1, np.nan)
        ref_major[i] = v[1] if (len(v) == 1 and 1 in v) or (len(v) == 2 and counts[c] > 0.5) else v.get(0, v.get(1))
    return FoldOutput(ref_major, ref_normal, ref_large, mixed, r_hat, p_large)


def shuffle_within(values: np.ndarray, groups: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute values among masked rows of the same group."""
    out = values.copy()
    for g in np.unique(groups[mask]):
        idx = np.flatnonzero(mask & (groups == g))
        out[idx] = values[rng.permutation(idx)]
    return out


def combine(o: FoldOutput, use_ridge: bool, use_mode: bool) -> np.ndarray:
    """[5] ln prediction from fold outputs."""
    ln = o.ref_major.copy()
    if use_mode:
        pick = o.mixed & np.isfinite(o.p_large)
        ln[pick] = np.where(o.p_large[pick] > 0.5, o.ref_large[pick], o.ref_normal[pick])
    return ln + (o.r_hat if use_ridge else 0.0)


# ---------------------------------------------------------------- diagnostics
def r2_vs_zero(r_true: np.ndarray, r_hat: np.ndarray) -> float:
    """1 - SSE/SS(r): how much of the within-cluster residual the image explains (0 = nothing)."""
    m = np.isfinite(r_true) & (np.abs(r_true) < GAP)
    return float(1 - np.sum((r_true[m] - r_hat[m]) ** 2) / np.sum(r_true[m] ** 2))


def within_category_auc(u: np.ndarray, p: np.ndarray, cats: np.ndarray) -> float:
    """Mean AUC inside each category that has both modes (category rate alone scores 0.5)."""
    aucs, weights = [], []
    for c in np.unique(cats):
        m = (cats == c) & np.isfinite(p)
        if m.sum() >= 4 and 0 < u[m].sum() < m.sum():
            aucs.append(roc_auc_score(u[m], p[m]))
            weights.append(m.sum())
    return float(np.average(aucs, weights=weights)) if aucs else np.nan


def oof_outputs(train: pd.DataFrame, folds: list[Fold], rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Run all folds and gather OOF arrays in train order."""
    n = len(train)
    out = {k: np.full(n, np.nan) for k in ("ref_major", "ref_normal", "ref_large", "r_hat", "p_large")}
    out["mixed"] = np.zeros(n, dtype=bool)
    preds = {k: np.full(n, np.nan) for k in ("no image (B2-like)", "+ size correction", "+ mode choice", "B3 = both")}
    for f in folds:
        o = run_fold(train, f, rng)
        for k in out:
            out[k][f.va] = getattr(o, k)
        for k, (a, b) in {"no image (B2-like)": (False, False), "+ size correction": (True, False),
                          "+ mode choice": (False, True), "B3 = both": (True, True)}.items():
            preds[k][f.va] = combine(o, a, b)
    out |= {f"pred:{k}": v for k, v in preds.items()}
    return out


def main() -> None:
    train = add_l1(load_split("train"), "train")
    val = add_l1(load_split("val"), "val")
    npz = np.load(FEATURES)
    row = {k: i for i, k in enumerate(npz["keys"])}
    X_all = npz["X"]
    X = X_all[[row[f"train/{s}"] for s in train["sample_id"]]]
    X_val = X_all[[row[f"val/{s}"] for s in val["sample_id"]]]
    y, ln = train["volume"].to_numpy(), train["ln"].to_numpy()
    cats = train["category"].to_numpy()
    u_true = cluster_labels(train)
    print(f"features {X.shape}; oversized objects (full-train flag) = {u_true.sum()}")

    # ---------------- OOF scores
    rows, first = [], None
    for seed in SEEDS:
        folds = make_folds(train, X, seed)
        o = oof_outputs(train, folds)
        first = first or (folds, o)
        ref_true = np.where(u_true == 1, o["ref_large"], o["ref_normal"])
        rows.append({
            **{k[5:]: smape(y, np.exp(v)) for k, v in o.items() if k.startswith("pred:")},
            "R2_size": r2_vs_zero(ln - ref_true, o["r_hat"]),
            "spearman_size": spearmanr((ln - ref_true)[np.isfinite(ref_true)], o["r_hat"][np.isfinite(ref_true)])[0],
            "AUC_mode_within_cat": within_category_auc(u_true, o["p_large"], cats),
        })
    res = pd.DataFrame(rows)
    print("\nsMAPE (%) — 5-fold stratified OOF, mean (std) over 3 seeds")
    for k in [c for c in res.columns if not c.startswith(("R2", "spear", "AUC"))]:
        print(f"  {k:22s} {res[k].mean():6.2f}  ({res[k].std():.2f})")
    print(f"\nsize correction : R² = {res.R2_size.mean():.4f}   Spearman = {res.spearman_size.mean():.3f}")
    print(f"mode classifier : within-category AUC = {res.AUC_mode_within_cat.mean():.3f}  (0.5 = guessing)")

    # sMAPE split by object type for B3 vs no-image (seed 0)
    folds0, o0 = first
    per = {k: 200 * np.abs(np.exp(o0[f"pred:{k}"]) - y) / (np.exp(o0[f"pred:{k}"]) + y)
           for k in ("no image (B2-like)", "B3 = both")}
    for name, m in {"normal objects": u_true == 0, "oversized objects": u_true == 1,
                    "mixed categories": o0["mixed"]}.items():
        print(f"  {name:18s} n={m.sum():4d}  no-image {per['no image (B2-like)'][m].mean():6.2f}"
              f"   B3 {per['B3 = both'][m].mean():6.2f}")

    # ---------------- permutation null (seed-0 folds; targets shuffled within category)
    print(f"\npermutation test: {N_PERM} within-category shuffles ...")
    rng = np.random.default_rng(0)
    null_r2, null_auc = [], []
    ref_true0 = np.where(u_true == 1, o0["ref_large"], o0["ref_normal"])
    for _ in range(N_PERM):
        on = oof_outputs(train, folds0, rng)
        null_r2.append(r2_vs_zero(ln - ref_true0, on["r_hat"]))
        null_auc.append(within_category_auc(u_true, on["p_large"], cats))
    real_r2 = r2_vs_zero(ln - ref_true0, o0["r_hat"])
    real_auc = within_category_auc(u_true, o0["p_large"], cats)
    null_r2, null_auc = np.array(null_r2), np.array(null_auc)
    print(f"  size R² : real {real_r2:.4f} | null mean {null_r2.mean():.4f}, max {null_r2.max():.4f}"
          f" | p = {(np.sum(null_r2 >= real_r2) + 1) / (N_PERM + 1):.3f}")
    print(f"  mode AUC: real {real_auc:.3f} | null mean {np.nanmean(null_auc):.3f}, max {np.nanmax(null_auc):.3f}"
          f" | p = {(np.sum(null_auc >= real_auc) + 1) / (N_PERM + 1):.3f}")

    # ---------------- val submission (fit on all train)
    scaler = StandardScaler().fit(X)
    pca = PCA(N_PCA, random_state=0).fit(scaler.transform(X))
    q = pd.concat([train, val.assign(ln=np.nan, volume=np.nan)], ignore_index=True)
    o = run_fold(q, Fold(np.arange(len(train)), np.arange(len(train), len(q)),
                         pca.transform(scaler.transform(X)), pca.transform(scaler.transform(X_val))))
    best = res[[c for c in res.columns if c in ("no image (B2-like)", "+ size correction", "+ mode choice", "B3 = both")]
               ].mean().idxmin()
    flags = {"no image (B2-like)": (False, False), "+ size correction": (True, False),
             "+ mode choice": (False, True), "B3 = both": (True, True)}[best]
    pred = np.exp(combine(o, *flags))
    path = write_submission("b3_dinov2_ridge", val, pred)
    print(f"\nsubmission from best OOF variant '{best}': min={pred.min():.2f} median={np.median(pred):.2f} "
          f"max={pred.max():.2f} -> {path}")


if __name__ == "__main__":
    main()
