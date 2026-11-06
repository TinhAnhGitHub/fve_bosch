# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn"]
# ///
"""B2: category-statistics baselines that fix B1's two weaknesses (small categories, oversized mode).

Variants, all computed strictly inside each training fold:
  B2a  per-category sMAPE-optimal constant (argmin of expected sMAPE over the category's train volumes)
  B2b  B2a + shrinkage: the coarse group (category_l1) volumes join with total weight k (pseudo-count)
  B2c  explicit two-mode: shrunk base-mode median, switched to the oversized-mode value when the
       Beta-Binomial-smoothed oversized rate of the category exceeds tau

Hyperparameters are picked on OOF seeds 0-2 and re-scored on unseen seeds 3-5 (honest estimate).

Run:  uv run baselines/b2.py
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from b0_b1 import DATA, load_split, predict_b1, smape, write_submission

SELECT_SEEDS: tuple[int, ...] = (0, 1, 2)
CHECK_SEEDS: tuple[int, ...] = (3, 4, 5)
GRID: np.ndarray = np.arange(np.log(0.2), np.log(60000), 0.01)  # candidate ln-volume predictions
GAP: float = np.log(30.0)  # a jump of >=30x inside a category separates the oversized mode

Predictor = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def add_l1(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Attach category_l1 (from train gt.json; val inherits it through the category)."""
    l1 = {}
    for p in (DATA / "train").rglob("gt.json"):
        g = json.loads(p.read_text())
        l1[g["category"]] = g["category_l1"]
    df = df.assign(l1=df["category"].map(l1))
    if split == "train":
        df["ln"] = np.log(df["volume"])
    return df


def smape_argmin(z: np.ndarray, w: np.ndarray) -> float:
    """ln-volume t minimising the weighted mean of 2*tanh(|t - z_i| / 2), i.e. expected sMAPE."""
    loss = (w[None, :] * np.tanh(np.abs(GRID[:, None] - z[None, :]) / 2)).sum(axis=1)
    return float(GRID[np.argmin(loss)])


def make_b2b(k: float) -> Predictor:
    """B2a (k=0) / B2b: per-category sMAPE argmin, coarse group mixed in with total weight k."""
    def predict(train: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
        out: dict[str, float] = {}
        global_t = smape_argmin(train["ln"].to_numpy(), np.ones(len(train)))
        groups = {g: d["ln"].to_numpy() for g, d in train.groupby("l1")}
        for cat in query["category"].unique():
            c = train[train["category"] == cat]
            if c.empty:
                out[cat] = global_t
                continue
            zc = c["ln"].to_numpy()
            zg = groups[c["l1"].iloc[0]]
            z = np.concatenate([zc, zg]) if k > 0 else zc
            w = np.concatenate([np.ones(len(zc)), np.full(len(zg), k / len(zg))]) if k > 0 else np.ones(len(zc))
            out[cat] = smape_argmin(z, w)
        return np.exp(query["category"].map(out).to_numpy())
    return predict


def flag_large(train: pd.DataFrame) -> pd.Series:
    """In-fold oversized flag: above the largest >=30x gap inside the category."""
    flag = pd.Series(False, index=train.index)
    for _, g in train.groupby("category"):
        z = np.sort(g["ln"].to_numpy())
        if len(z) > 1 and np.diff(z).max() >= GAP:
            cut = z[np.argmax(np.diff(z)) + 1]
            flag[g.index[g["ln"] >= cut]] = True
    return flag


def make_b2c(k: float, tau: float, a: float = 2.0) -> Predictor:
    """B2c: explicit modes. Base median shrunk to the l1 base median; oversized if smoothed rate > tau.

    Categories with no internal gap but whose median sits >=30x above the l1 base median
    (all-oversized categories such as hazelnut/longan/loquat) are treated as fully oversized.
    """
    def predict(train: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
        train = train.assign(large=flag_large(train))
        base = train[~train["large"]]
        l1_base = base.groupby("l1")["ln"].median()
        cat_med = train.groupby("category")["ln"].median()
        all_large = cat_med.index[(cat_med - train.groupby("category")["l1"].first().map(l1_base)) >= GAP]
        train.loc[train["category"].isin(all_large), "large"] = True
        base = train[~train["large"]]
        p0 = train["large"].mean()
        shift = (train[train["large"]].groupby("category")["ln"].median()
                 - base.groupby("category")["ln"].median()).dropna()
        delta = float(shift.median()) if len(shift) else np.log(1000.0)
        out: dict[str, float] = {}
        for cat in query["category"].unique():
            c = train[train["category"] == cat]
            if c.empty:
                out[cat] = float(train["ln"].median())
                continue
            cb, cl = c[~c["large"]], c[c["large"]]
            prior = l1_base.get(c["l1"].iloc[0], base["ln"].median())
            m = (len(cb) * cb["ln"].median() + k * prior) / (len(cb) + k) if len(cb) else prior
            rate = (len(cl) + a * p0) / (len(c) + a)
            out[cat] = (cl["ln"].median() if len(cl) else m + delta) if rate > tau else m
        return np.exp(query["category"].map(out).to_numpy())
    return predict


def oof_score(train: pd.DataFrame, predict: Predictor, seeds: tuple[int, ...]) -> tuple[float, np.ndarray]:
    """Mean OOF sMAPE over seeds, plus per-sample error averaged over seeds."""
    y = train["volume"].to_numpy()
    scores, per = [], np.zeros(len(train))
    for seed in seeds:
        pred = np.empty(len(train))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(train, train["category"]))
        for tr_idx, va_idx in folds:
            pred[va_idx] = predict(train.iloc[tr_idx], train.iloc[va_idx])
        scores.append(smape(y, pred))
        per += 200 * np.abs(pred - y) / (pred + y) / len(seeds)
    return float(np.mean(scores)), per


def main() -> None:
    train = add_l1(load_split("train"), "train")
    val = add_l1(load_split("val"), "val")
    large_all = flag_large(train)

    candidates: dict[str, Predictor] = {"B1 category median": predict_b1, "B2a argmin (k=0)": make_b2b(0)}
    candidates |= {f"B2b argmin k={k}": make_b2b(k) for k in (0.5, 1, 2, 4, 8)}
    candidates |= {f"B2c modes k={k} tau={t}": make_b2c(k, t) for k in (0, 1, 2, 4) for t in (0.3, 0.5, 0.7)}

    rows = []
    for name, fn in candidates.items():
        s, per = oof_score(train, fn, SELECT_SEEDS)
        rows.append({"model": name, "oof_seeds_0-2": s,
                     "normal_objects": per[~large_all].mean(), "oversized_objects": per[large_all].mean()})
    res = pd.DataFrame(rows).sort_values("oof_seeds_0-2")
    print("sMAPE (%) — 5-fold stratified OOF; per-group columns split by the oversized flag")
    print(res.round(2).to_string(index=False))

    best_b2b = res[res.model.str.startswith("B2b") | res.model.str.startswith("B2a")].iloc[0].model
    best_b2c = res[res.model.str.startswith("B2c")].iloc[0].model
    print("\nhonest re-check on unseen seeds 3-5:")
    for name in ("B1 category median", best_b2b, best_b2c):
        print(f"  {name:28s} {oof_score(train, candidates[name], CHECK_SEEDS)[0]:.2f}")

    best = res[res.model != "B1 category median"].iloc[0].model
    pred = candidates[best](train, val)
    path = write_submission("b2_category_stats", val, pred)
    print(f"\nsubmission from '{best}': {len(pred)} preds, min={pred.min():.2f} "
          f"median={np.median(pred):.2f} max={pred.max():.2f} -> {path}")
    changed = val.assign(b1=predict_b1(train, val), b2=pred)
    changed = changed[np.abs(np.log(changed.b2 / changed.b1)) > np.log(3)]
    print(f"categories where B2 differs from B1 by >3x: "
          f"{sorted(changed.category.unique())}")


if __name__ == "__main__":
    main()
