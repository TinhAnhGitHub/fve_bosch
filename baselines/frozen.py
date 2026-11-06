"""Shared harness for the B3.x frozen-feature experiments.

Pipeline (identical to B3, but every knob is in ``Config``):
  [2] per-category clusters (normal / oversized) with sMAPE-best values
  [3] ridge on PCA features -> size correction r = ln(true) - ln(cluster value)
  [4] logistic on PCA features -> P(oversized) in two-cluster categories
  [5] ln volume = chosen cluster value + r_hat

Usage:
    from frozen import Config, Experiment
    exp = Experiment()
    res = exp.evaluate(Config(name="b3"))     # OOF over 3 seeds, logged to experiments/log.csv
    exp.submit(Config(name="b3"), "b3_x")     # fit on all train, write validated preds_volume.zip
"""

from __future__ import annotations

import csv
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from b0_b1 import PROJECT, load_split, smape, write_submission
from b2 import GAP, add_l1
from b3 import argmin_t, build_table, cluster_labels, shuffle_within, within_category_auc

FEATURE_DIR = PROJECT / "features"
LOG = PROJECT / "experiments" / "log.csv"
BLOCKS_PER_FILE = 6  # [L_a cls, L_a mean, L_b cls, L_b mean, L_c cls, L_c mean]


@dataclass(frozen=True)
class Config:
    """One experiment. ``features`` = ((npz file stem, block indices), ...)."""

    name: str
    features: tuple[tuple[str, tuple[int, ...]], ...] = (("dinov2l_reg_518", (0, 1, 2, 3, 4, 5)),)
    n_pca: int | None = 256
    C: float = 0.05
    r_clip: float = 1.5
    tau: float = 0.5
    use_ridge: bool = True
    use_mode: bool = True


@dataclass
class Fold:
    tr: np.ndarray
    va: np.ndarray
    Z_tr: np.ndarray
    Z_va: np.ndarray


class Experiment:
    """Loads data once, caches features and per-fold PCA, evaluates configs."""

    def __init__(self) -> None:
        self.train = add_l1(load_split("train"), "train")
        self.val = add_l1(load_split("val"), "val")
        self.y = self.train["volume"].to_numpy()
        self.ln = self.train["ln"].to_numpy()
        self.cats = self.train["category"].to_numpy()
        self.u_true = cluster_labels(self.train)
        self._npz: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._folds: dict[tuple, list[Fold]] = {}
        self._splits: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}

    # ------------------------------------------------------------ features
    def features(self, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
        """(X_train, X_val) for the config's feature files and blocks, concatenated."""
        parts_tr, parts_va = [], []
        for stem, blocks in cfg.features:
            if stem not in self._npz:
                npz = np.load(FEATURE_DIR / f"{stem}.npz")
                row = {k: i for i, k in enumerate(npz["keys"])}
                X = npz["X"]
                self._npz[stem] = (X[[row[f"train/{s}"] for s in self.train["sample_id"]]],
                                   X[[row[f"val/{s}"] for s in self.val["sample_id"]]])
            Xtr, Xva = self._npz[stem]
            d = Xtr.shape[1] // BLOCKS_PER_FILE
            cols = np.concatenate([np.arange(b * d, (b + 1) * d) for b in blocks])
            parts_tr.append(Xtr[:, cols])
            parts_va.append(Xva[:, cols])
        return np.hstack(parts_tr), np.hstack(parts_va)

    def splits(self, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
        if seed not in self._splits:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._splits[seed] = list(StratifiedKFold(5, shuffle=True, random_state=seed)
                                          .split(self.train, self.cats))
        return self._splits[seed]

    @staticmethod
    def _project(X_fit: np.ndarray, X_apply: np.ndarray, n_pca: int | None) -> tuple[np.ndarray, np.ndarray]:
        scaler = StandardScaler().fit(X_fit)
        A, B = scaler.transform(X_fit), scaler.transform(X_apply)
        if n_pca is None:
            return A, B
        pca = PCA(n_pca, random_state=0).fit(A)
        return pca.transform(A), pca.transform(B)

    def folds(self, cfg: Config, seed: int) -> list[Fold]:
        key = (cfg.features, cfg.n_pca, seed)
        if key not in self._folds:
            X, _ = self.features(cfg)
            self._folds[key] = [Fold(tr, va, *self._project(X[tr], X[va], cfg.n_pca)) for tr, va in self.splits(seed)]
        return self._folds[key]

    # ------------------------------------------------------------ model
    @staticmethod
    def fit_predict(T: pd.DataFrame, Q: pd.DataFrame, Z_tr: np.ndarray, Z_va: np.ndarray, cfg: Config,
                    rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        """Stages [2]-[5] fitted on T, applied to Q. Returns ln predictions and diagnostics."""
        u_tr = cluster_labels(T)
        table, ref_loo = build_table(T, u_tr)
        cats_tr, q_cats = T["category"].to_numpy(), Q["category"].to_numpy()

        r = T["ln"].to_numpy() - ref_loo
        ok = np.isfinite(r) & (np.abs(r) < GAP)
        if rng is not None:
            r = shuffle_within(r, cats_tr, ok, rng)
        r_hat = np.zeros(len(Q))
        if cfg.use_ridge:
            ridge = RidgeCV(alphas=np.logspace(0, 6, 25)).fit(Z_tr[ok], r[ok])
            r_hat = np.clip(ridge.predict(Z_va), -cfg.r_clip, cfg.r_clip)

        mixed_cats = [c for c, v in table.values.items() if len(v) == 2]
        m_tr = np.isin(cats_tr, mixed_cats)
        mixed = np.isin(q_cats, mixed_cats)
        u_fit = shuffle_within(u_tr.astype(float), cats_tr, m_tr, rng).astype(int) if rng is not None else u_tr
        p_large = np.full(len(Q), np.nan)
        if cfg.use_mode and m_tr.sum() >= 20 and len(set(u_fit[m_tr])) == 2 and mixed.any():
            clf = LogisticRegression(C=cfg.C, max_iter=5000).fit(Z_tr[m_tr], u_fit[m_tr])
            p_large[mixed] = clf.predict_proba(Z_va[mixed])[:, 1]

        rate = pd.Series(u_tr).groupby(cats_tr).mean()
        ln = np.empty(len(Q))
        ref_n, ref_l = np.full(len(Q), np.nan), np.full(len(Q), np.nan)
        for i, c in enumerate(q_cats):
            v = table.values.get(c)
            if v is None:
                ln[i] = table.fallback
                continue
            ref_n[i], ref_l[i] = v.get(0, np.nan), v.get(1, np.nan)
            if len(v) == 1:
                ln[i] = next(iter(v.values()))
            elif np.isfinite(p_large[i]):
                ln[i] = v[1] if p_large[i] > cfg.tau else v[0]
            else:
                ln[i] = v[1] if rate[c] > 0.5 else v[0]
        return {"ln": ln + r_hat, "r_hat": r_hat, "p_large": p_large, "ref_n": ref_n, "ref_l": ref_l}

    def oof(self, cfg: Config, seed: int, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        n = len(self.train)
        out = {k: np.full(n, np.nan) for k in ("ln", "r_hat", "p_large", "ref_n", "ref_l")}
        for f in self.folds(cfg, seed):
            res = self.fit_predict(self.train.iloc[f.tr], self.train.iloc[f.va], f.Z_tr, f.Z_va, cfg, rng)
            for k, v in res.items():
                out[k][f.va] = v
        return out

    # ------------------------------------------------------------ scoring
    def diagnostics(self, o: dict[str, np.ndarray]) -> dict[str, float]:
        pred = np.exp(o["ln"])
        per = 200 * np.abs(pred - self.y) / (pred + self.y)
        ref = np.where(self.u_true == 1, o["ref_l"], o["ref_n"])
        r_true = self.ln - ref
        m = np.isfinite(r_true) & (np.abs(r_true) < GAP)
        return {"smape": smape(self.y, pred),
                "smape_normal": per[self.u_true == 0].mean(), "smape_oversized": per[self.u_true == 1].mean(),
                "R2_size": 1 - np.sum((r_true[m] - o["r_hat"][m]) ** 2) / np.sum(r_true[m] ** 2),
                "AUC_mode": within_category_auc(self.u_true, o["p_large"], self.cats)}

    def evaluate(self, cfg: Config, seeds: tuple[int, ...] = (0, 1, 2), log: bool = True,
                 quiet: bool = False) -> dict[str, float]:
        """Mean OOF metrics over seeds; appends a row to experiments/log.csv."""
        t0 = time.time()
        rows = [self.diagnostics(self.oof(cfg, s)) for s in seeds]
        df = pd.DataFrame(rows)
        res = {**df.mean().to_dict(), "smape_std": float(df["smape"].std()), "seeds": str(seeds),
               "seconds": round(time.time() - t0, 1)}
        if log:
            self.log(cfg, res)
        if not quiet:
            print(f"{cfg.name:45s} sMAPE {res['smape']:6.2f} ±{res['smape_std']:.2f} | normal "
                  f"{res['smape_normal']:5.2f} oversized {res['smape_oversized']:5.2f} | R² {res['R2_size']:.3f} "
                  f"AUC {res['AUC_mode']:.3f} | {res['seconds']:.0f}s")
        return res

    def permutation_test(self, cfg: Config, n_perm: int = 30, seed: int = 0) -> dict[str, float]:
        """Within-category shuffles of the size and mode targets; p-values for R² and AUC."""
        real = self.diagnostics(self.oof(cfg, seed))
        rng = np.random.default_rng(0)
        null = pd.DataFrame([self.diagnostics(self.oof(cfg, seed, rng)) for _ in range(n_perm)])
        out = {"R2_real": real["R2_size"], "R2_null_max": null["R2_size"].max(),
               "p_R2": (np.sum(null["R2_size"] >= real["R2_size"]) + 1) / (n_perm + 1),
               "AUC_real": real["AUC_mode"], "AUC_null_max": null["AUC_mode"].max(),
               "p_AUC": (np.sum(null["AUC_mode"] >= real["AUC_mode"]) + 1) / (n_perm + 1)}
        print("permutation test:", {k: round(v, 4) for k, v in out.items()})
        return out

    @staticmethod
    def log(cfg: Config, res: dict[str, float]) -> None:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        row = {"time": datetime.now().isoformat(timespec="seconds"), **asdict(cfg), **res}
        row["features"] = str(cfg.features)
        new = not LOG.exists()
        with LOG.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    # ------------------------------------------------------------ submission
    def submit(self, cfg: Config, folder: str) -> np.ndarray:
        """Fit on all train, predict val, write a validated preds_volume.zip."""
        X, X_val = self.features(cfg)
        Z_tr, Z_va = self._project(X, X_val, cfg.n_pca)
        res = self.fit_predict(self.train, self.val.assign(ln=np.nan), Z_tr, Z_va, cfg)
        pred = np.exp(res["ln"])
        path = write_submission(folder, self.val, pred)
        print(f"submission {cfg.name}: min={pred.min():.2f} median={np.median(pred):.2f} "
              f"max={pred.max():.2f} -> {path}")
        return pred
