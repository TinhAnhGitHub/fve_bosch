# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn"]
# ///
"""B0 (global median) and B1 (category median) baselines.

Scores both on train (in-sample and out-of-fold), then writes val predictions.

Run:  uv run baselines/b0_b1.py
"""

from __future__ import annotations

import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT: Path = Path(__file__).resolve().parent.parent
DATA: Path = PROJECT / "data" / "fve"
OUT: Path = PROJECT / "submissions"
SEEDS: tuple[int, ...] = (0, 1, 2)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean sMAPE in percent (competition definition).

    Examples:
        >>> round(smape(np.array([100.0]), np.array([50.0])), 1)
        66.7
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))


def load_split(split: str) -> pd.DataFrame:
    """One row per sample: group, category, sample_id and (train only) volume."""
    rows: list[dict[str, object]] = []
    for img in sorted((DATA / split).rglob("rgb_industrial.png")):
        group, category, sample_id = img.parent.relative_to(DATA / split).parts
        row: dict[str, object] = {"group": group, "category": category, "sample_id": sample_id}
        gt = img.parent / "gt.json"
        if gt.exists():
            row["volume"] = json.loads(gt.read_text())["volume"]["primary_cm3"]
        rows.append(row)
    return pd.DataFrame(rows)


def predict_b0(train: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    """B0: the global train median for every query."""
    return np.full(len(query), train["volume"].median())


def predict_b1(train: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    """B1: the train median of the query's category, falling back to the global median."""
    cat_median = train.groupby("category")["volume"].median()
    return query["category"].map(cat_median).fillna(train["volume"].median()).to_numpy()


def oof(train: pd.DataFrame, predict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions plus a mask of rows whose category had in-fold training data."""
    pred = np.empty(len(train))
    seen = np.zeros(len(train), dtype=bool)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    with warnings.catch_warnings():  # categories with <5 objects trigger a harmless warning
        warnings.simplefilter("ignore", UserWarning)
        splits = list(skf.split(train, train["category"]))
    for tr_idx, va_idx in splits:
        fold_tr, fold_va = train.iloc[tr_idx], train.iloc[va_idx]
        pred[va_idx] = predict(fold_tr, fold_va)
        seen[va_idx] = fold_va["category"].isin(fold_tr["category"]).to_numpy()
    return pred, seen


def write_submission(name: str, val: pd.DataFrame, pred: np.ndarray) -> Path:
    """Write preds_volume.zip in the official layout: volume/{sample_id}.json -> {"volume_cm3": <number>}.

    A CSV copy is kept next to the zip for inspection only (it is not part of the submission).
    """
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    val.assign(volume_cm3=pred)[["group", "category", "sample_id", "volume_cm3"]].to_csv(
        folder / "preds_volume.csv", index=False)
    zpath = folder / "preds_volume.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for sample_id, v in zip(val["sample_id"], pred):
            zf.writestr(f"volume/{sample_id}.json", json.dumps({"volume_cm3": round(float(v), 4)}))
    validate_submission(zpath, set(val["sample_id"]))
    return zpath


def validate_submission(zpath: Path, expected_ids: set[str]) -> None:
    """Check the zip against the official rules; raise on any violation."""
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        assert all(n.startswith("volume/") and n.endswith(".json") and n.count("/") == 1 for n in names), \
            "every file must be volume/{sample_id}.json"
        ids = {n[len("volume/"):-len(".json")] for n in names}
        assert len(ids) == len(names), "duplicate sample files"
        assert ids == expected_ids, f"missing={len(expected_ids - ids)} extra={len(ids - expected_ids)}"
        for n in names:
            obj = json.loads(zf.read(n))
            assert isinstance(obj, dict) and set(obj) == {"volume_cm3"}, f"{n}: must be one object with volume_cm3"
            v = obj["volume_cm3"]
            assert isinstance(v, (int, float)) and not isinstance(v, bool), f"{n}: volume_cm3 must be a number"
            assert np.isfinite(v) and v > 0, f"{n}: volume_cm3 must be finite and > 0"


def main() -> None:
    train, val = load_split("train"), load_split("val")
    y = train["volume"].to_numpy()
    print(f"train={len(train)}  val={len(val)}  categories train={train.category.nunique()} "
          f"val={val.category.nunique()}  val categories unseen in train="
          f"{sorted(set(val.category) - set(train.category))}")

    rows = []
    for name, predict in {"B0 global median": predict_b0, "B1 category median": predict_b1}.items():
        in_sample = smape(y, predict(train, train))
        overall, seen_only, unseen_only = [], [], []
        for seed in SEEDS:
            p, seen = oof(train, predict, seed)
            overall.append(smape(y, p))
            seen_only.append(smape(y[seen], p[seen]))
            unseen_only.append(smape(y[~seen], p[~seen]) if (~seen).any() else np.nan)
        rows.append({
            "baseline": name,
            "in_sample": in_sample,
            "oof_overall": np.mean(overall),
            "oof_std_seeds": np.std(overall),
            "oof_cat_seen (LB proxy)": np.mean(seen_only),
            "oof_cat_unseen": np.nanmean(unseen_only),
            "n_unseen_rows": int((~seen).sum()),
        })
    print("\nsMAPE (%) — OOF = 5-fold stratified by category, mean of 3 seeds")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    for name, predict in {"b0_global_median": predict_b0, "b1_category_median": predict_b1}.items():
        pred = predict(train, val)
        path = write_submission(name, val, pred)
        print(f"\n{name}: {len(pred)} val preds, min={pred.min():.2f} median={np.median(pred):.2f} "
              f"max={pred.max():.2f}, finite={np.isfinite(pred).all()} -> {path}")


if __name__ == "__main__":
    main()
