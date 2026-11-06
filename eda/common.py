"""Shared helpers for the Food Volume Estimation EDA scripts.

Paths default to ``<project>/data/fve`` and ``<project>/eda/out``. Override them
with the ``FVE_DATA`` and ``FVE_OUT`` environment variables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = Path(os.environ.get("FVE_DATA", PROJECT_DIR / "data" / "fve"))
OUT_DIR: Path = Path(os.environ.get("FVE_OUT", PROJECT_DIR / "eda" / "out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS: tuple[str, ...] = ("train", "val", "eval", "test")
IMAGE_NAME: str = "rgb_industrial.png"

# A gap of at least this ratio between consecutive sorted volumes inside one
# category separates the "large" (likely mesh-unit-error) mode from the base mode.
LARGE_MODE_GAP_RATIO: float = 30.0
# Categories whose every training object sits in the large mode, so no
# within-category gap exists. Found by inspection on 2026-11-04 (e.g. 20-30 cm hazelnuts).
ALL_LARGE_CATEGORIES: frozenset[str] = frozenset({"hazelnut", "longan", "loquat"})


def smape_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sample sMAPE in percent, using the competition definition.

    Args:
        y_true: Ground-truth volumes.
        y_pred: Predicted volumes.

    Returns:
        Array of per-sample errors in [0, 200].

    Examples:
        >>> smape_per_sample(np.array([100.0]), np.array([50.0])).round(1)
        array([66.7])
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(denom > 0, np.abs(y_pred - y_true) / denom, 0.0)
    return 100.0 * err


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean sMAPE in percent.

    Examples:
        >>> round(smape(np.array([100.0, 100.0]), np.array([100.0, 200.0])), 2)
        33.33
    """
    return float(np.mean(smape_per_sample(y_true, y_pred)))


def _read_gt(path: Path) -> dict[str, object]:
    """Flatten one ``gt.json`` into a single row."""
    g = json.loads(path.read_text())
    v = g["volume"]
    bx, by, bz = g["bbox_mm"]
    return {
        "volume_cm3": v["primary_cm3"],
        "gt_confidence": v["gt_confidence"],
        "gt_protocol": v["gt_protocol"],
        "category_l1": g["category_l1"],
        "weight_g": g.get("weight_g"),
        "bbox_x_mm": bx,
        "bbox_y_mm": by,
        "bbox_z_mm": bz,
        "mesh_decimated": g["mesh"]["decimated"],
        "mesh_n_faces": g["mesh"]["n_faces"],
    }


def build_index(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Walk every split and return one row per sample, with labels where available.

    Args:
        data_dir: Root holding ``train/``, ``val/`` (or ``eval/``) and optionally ``test/``.

    Returns:
        DataFrame with paths, folder metadata, labels and derived columns.
    """
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        for img in sorted(split_dir.rglob(IMAGE_NAME)):
            sample_dir = img.parent
            rel = sample_dir.relative_to(split_dir).parts  # (group, category, sample_id)
            row: dict[str, object] = {
                "split": split,
                "group": rel[0] if len(rel) >= 3 else None,
                "category": rel[-2] if len(rel) >= 2 else None,
                "sample_id": sample_dir.name,
                "source": "metafood3d" if sample_dir.name.startswith("mf3d_") else "omniobject3d",
                "img_path": str(img),
                "mesh_path": str(sample_dir / "mesh.ply") if (sample_dir / "mesh.ply").exists() else None,
            }
            gt = sample_dir / "gt.json"
            if gt.exists():
                row.update(_read_gt(gt))
            rows.append(row)
    df = pd.DataFrame(rows)
    if "volume_cm3" in df:
        df["ln_vol"] = np.log(df["volume_cm3"])
        dims = df[["bbox_x_mm", "bbox_y_mm", "bbox_z_mm"]].to_numpy()
        dims_sorted = np.sort(dims, axis=1)[:, ::-1]
        df["dim_max_mm"], df["dim_mid_mm"], df["dim_min_mm"] = dims_sorted.T
        df["bbox_vol_cm3"] = dims.prod(axis=1) / 1000
        df["fill_ratio"] = df["volume_cm3"] / df["bbox_vol_cm3"]
        df["density_g_cm3"] = df["weight_g"] / df["volume_cm3"]
        df["large_mode"] = flag_large_mode(df)
    return df


def flag_large_mode(df: pd.DataFrame, gap_ratio: float = LARGE_MODE_GAP_RATIO) -> pd.Series:
    """Flag labelled samples in the oversized (~200-1000x) mode of their category.

    Inside each category, the volumes are sorted and split at the largest
    consecutive gap if that gap is at least ``gap_ratio``. Samples above it are
    flagged. Categories in ``ALL_LARGE_CATEGORIES`` are flagged entirely.

    Args:
        df: Index with ``category`` and ``volume_cm3``.
        gap_ratio: Minimum ratio between neighbouring sorted volumes to call a split.

    Returns:
        Boolean Series aligned with ``df`` (False for unlabelled rows).
    """
    flag = pd.Series(False, index=df.index)
    labelled = df[df["volume_cm3"].notna()]
    for cat, grp in labelled.groupby("category"):
        if cat in ALL_LARGE_CATEGORIES:
            flag[grp.index] = True
            continue
        ln = np.sort(np.log(grp["volume_cm3"].to_numpy()))
        if len(ln) < 2:
            continue
        gaps = np.diff(ln)
        i = int(np.argmax(gaps))
        if gaps[i] >= np.log(gap_ratio):
            threshold = np.exp(ln[i + 1])
            flag[grp.index[grp["volume_cm3"] >= threshold]] = True
    return flag


def load_index(refresh: bool = False) -> pd.DataFrame:
    """Load the cached sample index, building it on first use.

    Args:
        refresh: Rebuild even if the cache exists.

    Returns:
        The sample index DataFrame.
    """
    cache = OUT_DIR / "index.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    df = build_index()
    df.to_parquet(cache, index=False)
    return df


def savefig(fig: "object", name: str) -> Path:
    """Save a matplotlib figure under ``OUT_DIR`` and return its path."""
    path = OUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")  # type: ignore[attr-defined]
    print(f"saved {path}")
    return path
