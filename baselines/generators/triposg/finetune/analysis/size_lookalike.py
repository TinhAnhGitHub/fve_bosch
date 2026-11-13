"""Look-alike size guess for the LoRA shapes, tested on the holdout without leakage.

The LoRA draws a good shape at an unknown size, and its own size head is no better than the food type's median size.
This script tries another size guess, built from training objects only:

  look-alikes  the K training photos of the same food type whose image features are closest (cosine); their real radii,
               averaged in log space, are the guess (K = 2)
  formula      ridge regression on the same features, as a correction around the food type's median radius
  type median  the median radius of the food type
  blend        the three mixed in log space; the weights are chosen out-of-fold on the pool, never on the objects being judged

Pool and queries never overlap: the holdout test uses the 1,838 train objects as the pool and judges the 204 holdout objects;
the public photos use all 2,042 training objects as the pool. The shape is scaled to the guessed radius with
finetune/common.py::metric_points, so only the size number changes.

Run with:  uv run --no-project --with numpy --with scipy --with trimesh python -I size_lookalike.py <command> ...
  selftest                                            checks the pieces on synthetic data
  labels  DATA_DIR METADATA OUT_DIR                   radius of every training mesh -> OUT_DIR/radii.npz
  select  FEATURE_DIR METADATA OUT_DIR                out-of-fold choice on the train pool -> OUT_DIR/config.json
  holdout FEATURE_DIR METADATA OUT_DIR LORA_VAL_NPZ   guesses for the 204 holdout objects (pool = train) -> OUT_DIR/holdout_sizes.npz
  twins   DATA_DIR FEATURE_DIR METADATA OUT_DIR       is the holdout as close to the pool as the public photos are to all 2,042?
  score   DATA_DIR METADATA OUT_DIR LORA_VAL_NPZ [ARM ...]   CD of the LoRA shape at each guess (rotation + translation only)
  table    OUT_DIR HOLDOUT_DIR                        add the look-alike score to the holdout page (lora_vs_truth/index_lookalike.csv)
  public  DATA_DIR FEATURE_DIR METADATA OUT_DIR LORA_EVAL_NPZ  guesses for the 511 public photos (pool = all 2,042) + npz + zip
  neighbours DATA_DIR FEATURE_DIR METADATA OUT_DIR LORA_EVAL_NPZ  the look-alikes behind each public guess -> OUT_DIR/public_neighbours.csv
"""
import ast
import csv
import json
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FINETUNE = Path(__file__).resolve().parents[1]
BASELINES = FINETUNE.parents[2]
K = 2
# The cached features come from the volume task, which calls its test split "val": the 511 public photos are stored as val/<id>.
# They are byte-identical to data/f3d/eval/*/*/<id>/rgb_industrial.png (checked for all 2,553 photos, train and public).
PUBLIC_FEATURE_KEY = "val"
ALPHAS =(1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5)
FEATURE_OPTIONS = {
    "dinov2_cls": "dinov2l_reg_518:0,2,4",
    "dinov3h_cls": "dinov3_vith16plus_512:0,2,4",
    "dinov2+dinov3h_cls": "dinov2l_reg_518:0,2,4+dinov3_vith16plus_512:0,2,4",
    "dinov2+dinov3h_all": "dinov2l_reg_518:0,1,2,3,4,5+dinov3_vith16plus_512:0,1,2,3,4,5",
    "dinov3l_cls": "dinov3_vitl16_512:0,2,4",
    "siglip2_cls": "siglip2_so400m_512:0,2,4",
    "cradio_cls": "cradio_v4h_512:0,2,4",
}


# ---------------------------------------------------------------- reuse of the repo's own functions (common.py needs torch)
def repo_function(path, name):
    """Take one top-level function from a repo file without importing the file (common.py imports torch at module level)."""
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module([node], []), str(path), "exec"), namespace)
    return namespace[name]


def metric_points():
    return repo_function(FINETUNE / "common.py", "metric_points")


def aligned_cd():
    """finetune/evaluate.py::aligned_cd, the scorer behind the 79.27 and 24.17 of the report (needs a stand-in for common)."""
    common = types.ModuleType("common")
    common.metric_points = metric_points()
    common.read_json = lambda path: json.loads(Path(path).read_text())
    sys.modules["common"] = common
    sys.path.insert(0, str(FINETUNE))
    from evaluate import aligned_cd as function
    return function


# ---------------------------------------------------------------- data
def read_metadata(path):
    meta = json.loads(Path(path).read_text())
    return meta["rows"], float(meta["mesh_unit_mm"])


def sample_truth(job):
    """Centred surface samples of a mesh in mm, exactly as score_local.py::truth_of (8,192 points, seed 42)."""
    import trimesh
    data, row, unit_mm = job
    mesh = trimesh.load(Path(data) / row["mesh"], force="mesh", process=False)
    points, _ = trimesh.sample.sample_surface(mesh, 8192, seed=42)
    points = points * unit_mm
    return points - points.mean(0)


def radius_of(job):
    truth = sample_truth(job)
    return float(np.sqrt(np.square(truth).sum(1).mean()))


def load_features(feature_dir, spec):
    """spec 'dinov2l_reg_518:0,2,4+dinov3_vith16plus_512:0,2,4': each file holds six equal chunks (CLS and mean patch of three layers);
    take the listed chunks of every file. Returns the row keys ('train/<id>', 'eval/<id>') and the matrix."""
    keys, parts = None, []
    for item in spec.split("+"):
        name, chunks = item.split(":")
        with np.load(Path(feature_dir) / f"{name}.npz", allow_pickle=False) as z:
            matrix, these = z["X"], z["keys"].tolist()
        if keys is None:
            keys = these
        elif keys != these:
            raise ValueError(f"{name} has different rows than the first feature file")
        width = matrix.shape[1] // 6
        parts.append(np.concatenate([matrix[:, int(c) * width:(int(c) + 1) * width] for c in chunks.split(",")], 1))
    return keys, np.concatenate(parts, 1).astype(np.float64)


def feature_rows(keys, matrix, wanted):
    index = {k: i for i, k in enumerate(keys)}
    return matrix[[index[w] for w in wanted]]


def standardize(pool, query):
    """z-score with the pool's own statistics, then unit length (so a dot product is the cosine)."""
    mean, std = pool.mean(0), pool.std(0) + 1e-6
    a, b = (pool - mean) / std, (query - mean) / std
    return a / np.linalg.norm(a, axis=1, keepdims=True), b / np.linalg.norm(b, axis=1, keepdims=True)


# ---------------------------------------------------------------- the three estimators, all in log radius
def category_median(q_cat, pool_cat, pool_lnr, forbid=None):
    """Median log radius of the query's food type in the pool (forbid[i, j]: pool row j may not be used for query i)."""
    glob = np.median(pool_lnr)
    by_cat = {c: np.flatnonzero(pool_cat == c) for c in np.unique(pool_cat)}
    out = np.empty(len(q_cat))
    for i, c in enumerate(q_cat):
        cand = by_cat.get(c, np.empty(0, int))
        if forbid is not None:
            cand = cand[~forbid[i, cand]]
        out[i] = np.median(pool_lnr[cand]) if len(cand) else glob
    return out


def look_alikes(sim, q_cat, pool_cat, pool_lnr, k=K, forbid=None, record=None):
    """The k most similar pool objects of the same food type (any food type when it has none). Returns the similarity-weighted mean
    log radius, the best similarity, and the log-radius gap between the two closest (NaN with one neighbour). A list passed as
    `record` receives each query's neighbour indices into the pool, closest first."""
    by_cat = {c: np.flatnonzero(pool_cat == c) for c in np.unique(pool_cat)}
    guess, best, gap, fallback = np.empty(len(q_cat)), np.empty(len(q_cat)), np.full(len(q_cat), np.nan), 0
    for i, c in enumerate(q_cat):
        cand = by_cat.get(c, np.empty(0, int))
        if forbid is not None:
            cand = cand[~forbid[i, cand]]
        if len(cand) == 0:
            fallback += 1
            cand = np.flatnonzero(~forbid[i]) if forbid is not None else np.arange(len(pool_cat))
        near = cand[np.argsort(-sim[i, cand])][:k]
        if record is not None:
            record.append(near)
        weight = np.clip(sim[i, near], 1e-3, None)
        guess[i] = (weight * pool_lnr[near]).sum() / weight.sum()
        best[i] = sim[i, near[0]]
        if len(near) > 1:
            gap[i] = abs(pool_lnr[near[0]] - pool_lnr[near[1]])
    return guess, best, gap, fallback


def ridge_correction(a_pool, target, a_query, alphas):
    """Dual ridge on standardized features: one prediction per alpha for every query row, from one eigendecomposition."""
    mean = target.mean()
    values, vectors = np.linalg.eigh(a_pool @ a_pool.T)
    projected = vectors.T @ (target - mean)
    cross = a_query @ a_pool.T @ vectors
    return np.stack([mean + cross @ (projected / (np.maximum(values, 0) + alpha)) for alpha in alphas])


def ridge_formula(a_pool, pool_cat, pool_lnr, a_query, q_cat, alphas):
    """Per alpha: the food type's median log radius plus a ridge correction learned on the pool's residuals from its type medians."""
    medians = {c: np.median(pool_lnr[pool_cat == c]) for c in np.unique(pool_cat)}
    glob = np.median(pool_lnr)
    residual = pool_lnr - np.array([medians[c] for c in pool_cat])
    base = np.array([medians.get(c, glob) for c in q_cat])
    return base + ridge_correction(a_pool, residual, a_query, alphas)


def folds_by_group(groups, count=10, seed=0):
    unique = np.unique(groups)
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of = {g: i % count for g, i in zip(unique[order], range(len(unique)))}
    return np.array([fold_of[g] for g in groups])


def simplex(step=0.05, parts=3):
    """All weight triples on a grid that sum to one."""
    n = round(1 / step)
    return [np.array([i, j, n - i - j]) / n for i in range(n + 1) for j in range(n + 1 - i)]


def mix(weights, columns):
    return np.exp(sum(w * c for w, c in zip(weights, columns)))


def error_stats(guess, true):
    rel = np.abs(guess / true - 1)
    return dict(median_pct=100 * float(np.median(rel)), within10_pct=100 * float((rel < 0.1).mean()),
                over30_pct=100 * float((rel > 0.3).mean()), mse_mm2=float(np.mean((guess - true) ** 2)))


def line(name, stats):
    return (f"{name:34s} median error {stats['median_pct']:5.1f}%   within 10% {stats['within10_pct']:4.0f}%   "
            f"off by >30% {stats['over30_pct']:4.0f}%   radius MSE {stats['mse_mm2']:7.1f} mm2")


# ---------------------------------------------------------------- commands
def cmd_selftest():
    rng = np.random.default_rng(1)
    # ridge, dual form against the primal normal equations
    a, b, t = rng.normal(size=(40, 12)), rng.normal(size=(7, 12)), rng.normal(size=40)
    for alpha in (0.5, 30.0):
        primal = b @ np.linalg.solve(a.T @ a + alpha * np.eye(12), a.T @ (t - t.mean())) + t.mean()
        assert np.allclose(primal, ridge_correction(a, t, b, (alpha,))[0]), "dual ridge differs from the primal one"
    # look-alikes: same food type only, never itself or a copy of itself, closest first
    sim = np.array([[1.0, 0.2, 0.9, 0.8, 0.95], [0.2, 1.0, 0.1, 0.3, 0.2]])
    cat = np.array(["a", "b", "a", "a", "b"])
    lnr = np.log(np.array([10.0, 20.0, 30.0, 50.0, 40.0]))
    forbid = np.zeros((2, 5), bool)
    forbid[0, 0] = True          # query 0 is pool row 0
    forbid[1, 1] = forbid[1, 4] = True  # query 1 is pool row 1 and row 4 is a copy of it: no neighbour is left in its food type
    guess, best, gap, fallback = look_alikes(sim, np.array(["a", "b"]), cat, lnr, k=2, forbid=forbid)
    want = (0.9 * np.log(30.0) + 0.8 * np.log(50.0)) / 1.7
    assert abs(guess[0] - want) < 1e-12 and best[0] == 0.9 and abs(gap[0] - np.log(50 / 30)) < 1e-12
    assert fallback == 1 and np.isfinite(guess[1])  # food type "b" has nobody left: falls back to the other food types
    # type median and folds
    assert np.allclose(category_median(np.array(["a"]), cat, lnr), np.log(30.0))
    folds = folds_by_group(np.array(list("aabbccddee")), count=3)
    assert all(len(set(folds[[i for i, g in enumerate("aabbccddee") if g == x]])) == 1 for x in "abcde"), "a group was split"
    # blend grid
    grid = simplex()
    assert len(grid) == 231 and all(abs(w.sum() - 1) < 1e-12 for w in grid)
    # the repo's own metric_points scales to the radius asked for
    cloud = metric_points()(rng.normal(size=(500, 3)) * [3, 2, 1] + 7, 25.0)
    assert abs(np.sqrt(np.square(cloud - cloud.mean(0)).sum(1).mean()) - 25.0) < 1e-3
    aligned_cd()  # imports the repo's scorer
    print("selftest passed")


def cmd_labels(data, metadata, out):
    rows, unit_mm = read_metadata(metadata)
    with ProcessPoolExecutor(14) as pool:
        radii = list(pool.map(radius_of, [(data, r, unit_mm) for r in rows], chunksize=8))
    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez(Path(out) / "radii.npz", ids=np.array([r["id"] for r in rows]), radius_mm=np.array(radii))
    print(f"{len(rows)} radii: median {np.median(radii):.1f} mm, min {min(radii):.1f}, max {max(radii):.1f}")


def load_pool(out, rows, split):
    """Rows of the given split(s), their log radii (from labels), and the arrays the estimators need."""
    with np.load(Path(out) / "radii.npz") as z:
        radius = dict(zip(z["ids"].tolist(), z["radius_mm"].tolist()))
    chosen = [r for r in rows if r["split"] in split]
    return chosen, np.array([radius[r["id"]] for r in chosen])


def out_of_fold(feature_dir, pool_rows, pool_r, spec):
    """Out-of-fold guesses on the pool for one feature choice: the look-alikes (leave one out, the same mesh also left out),
    and the ridge formula per alpha (10 folds, whole groups)."""
    keys, matrix = load_features(feature_dir, spec)
    raw = feature_rows(keys, matrix, [f"train/{r['id']}" for r in pool_rows])
    unit_pool, _ = standardize(raw, raw[:1])
    cat = np.array([r["category"] for r in pool_rows])
    group = np.array([r["group"] for r in pool_rows])
    lnr = np.log(pool_r)
    forbid = group[:, None] == group[None, :]
    sim = unit_pool @ unit_pool.T
    knn, best, gap, fallback = look_alikes(sim, cat, cat, lnr, forbid=forbid)
    cat_median = category_median(cat, cat, lnr, forbid=forbid)
    formula = np.zeros((len(ALPHAS), len(pool_rows)))
    fold = folds_by_group(group)
    for f in range(10):
        va, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        mean, std = raw[tr].mean(0), raw[tr].std(0) + 1e-6
        a_tr, a_va = (raw[tr] - mean) / std, (raw[va] - mean) / std
        a_tr /= np.linalg.norm(a_tr, axis=1, keepdims=True)
        a_va /= np.linalg.norm(a_va, axis=1, keepdims=True)
        formula[:, va] = ridge_formula(a_tr, cat[tr], lnr[tr], a_va, cat[va], ALPHAS)
    return dict(knn=knn, cat=cat_median, formula=formula, best=best, gap=gap, fallback=fallback)


def best_blend(oof, true, alpha_index):
    best = None
    for weights in simplex():
        mse = float(np.mean((mix(weights, (oof["cat"], oof["knn"], oof["formula"][alpha_index])) - true) ** 2))
        if best is None or mse < best[0]:
            best = (mse, weights)
    return best


def cmd_select(feature_dir, metadata, out):
    rows, _ = read_metadata(metadata)
    pool_rows, pool_r = load_pool(out, rows, ("train",))
    print(f"pool: {len(pool_rows)} train objects, {len({r['group'] for r in pool_rows})} distinct meshes; out-of-fold, no holdout object is used")
    results = {}
    for name, spec in FEATURE_OPTIONS.items():
        oof = out_of_fold(feature_dir, pool_rows, pool_r, spec)
        mses = [float(np.mean((np.exp(oof["formula"][a]) - pool_r) ** 2)) for a in range(len(ALPHAS))]
        a = int(np.argmin(mses))
        mse, weights = best_blend(oof, pool_r, a)
        results[name] = dict(spec=spec, alpha=ALPHAS[a], blend_mse=mse, weights=[float(w) for w in weights],
                             knn=error_stats(np.exp(oof["knn"]), pool_r), formula=error_stats(np.exp(oof["formula"][a]), pool_r),
                             blend=error_stats(mix(weights, (oof["cat"], oof["knn"], oof["formula"][a])), pool_r),
                             fallback=oof["fallback"])
        print(f"\n[{name}]  ridge alpha {ALPHAS[a]:g}; blend weights type-median/look-alikes/formula = {np.round(weights, 2).tolist()}")
        print("  " + line("look-alikes (K=2)", results[name]["knn"]))
        print("  " + line("ridge formula", results[name]["formula"]))
        print("  " + line("blend", results[name]["blend"]))
    ref = error_stats(np.exp(oof["cat"]), pool_r)
    print("\nreference on the same pool, out of fold:")
    print("  " + line("type median", ref))
    best = min(results, key=lambda n: results[n]["blend_mse"])
    config = dict(chosen=best, k=K, **results[best], reference_type_median=ref, all_options=results)
    (Path(out) / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"\nchosen features: {best} ({results[best]['spec']}); written to {Path(out) / 'config.json'}")


def guesses(config, feature_dir, pool_rows, pool_r, query_rows, query_prefix):
    """The type median, the look-alike guess, the ridge guess and the blend (radii in mm) for the query rows, from the pool only."""
    keys, matrix = load_features(feature_dir, config["spec"])
    raw_pool = feature_rows(keys, matrix, [f"train/{r['id']}" for r in pool_rows])
    raw_query = feature_rows(keys, matrix, [f"{query_prefix}/{r['id']}" for r in query_rows])
    a_pool, a_query = standardize(raw_pool, raw_query)
    cat_pool = np.array([r["category"] for r in pool_rows])
    cat_query = np.array([r["category"] for r in query_rows])
    lnr = np.log(pool_r)
    record = []
    sim = a_query @ a_pool.T
    knn, best, gap, fallback = look_alikes(sim, cat_query, cat_pool, lnr, record=record)
    cat = category_median(cat_query, cat_pool, lnr)
    formula = ridge_formula(a_pool, cat_pool, lnr, a_query, cat_query, (config["alpha"],))[0]
    weights = config["weights"]
    neighbours = np.array([list(near) + [-1] * (K - len(near)) for near in record])
    return dict(cat_median=np.exp(cat), knn=np.exp(knn), ridge=np.exp(formula), blend=mix(weights, (cat, knn, formula)),
                best_similarity=best, neighbour_gap=gap, neighbours=neighbours,
                neighbour_similarity=np.where(neighbours >= 0, np.take_along_axis(sim, np.maximum(neighbours, 0), 1), np.nan)), fallback


def cmd_holdout(feature_dir, metadata, out, lora_val):
    rows, _ = read_metadata(metadata)
    config = json.loads((Path(out) / "config.json").read_text())
    pool_rows, pool_r = load_pool(out, rows, ("train",))
    query_rows, true = load_pool(out, rows, ("val",))
    with np.load(lora_val, allow_pickle=False) as z:
        head = dict(zip(z["ids"].tolist(), z["radius_mm"].tolist()))
    sizes, fallback = guesses(config, feature_dir, pool_rows, pool_r, query_rows, "train")
    sizes["head"] = np.array([head[r["id"]] for r in query_rows])
    sizes["true"] = true
    np.savez(Path(out) / "holdout_sizes.npz", ids=np.array([r["id"] for r in query_rows]), **sizes)
    print(f"{len(query_rows)} holdout objects judged with a pool of {len(pool_rows)} train objects (no holdout object in the pool); "
          f"{fallback} had no same-type neighbour")
    for name, label in (("cat_median", "type median"), ("head", "our size head"), ("knn", "look-alikes (K=2)"),
                        ("ridge", "ridge formula"), ("blend", "BLEND (the plan)")):
        print(line(label, error_stats(sizes[name], true)))
    top = sizes["best_similarity"]
    print(f"holdout -> pool best cosine similarity: median {np.median(top):.3f}, 10th/90th percentile "
          f"{np.percentile(top, 10):.3f}/{np.percentile(top, 90):.3f}")


def eval_rows(data):
    """The public photos as pseudo rows (id, food type from the folder name)."""
    found = sorted((p.parent.name, p.parent.parent.name) for p in (Path(data) / "eval").glob("*/*/*/rgb_industrial.png"))
    return [dict(id=i, category=c) for i, c in found]


def cmd_twins(data, feature_dir, metadata, out):
    """Does the holdout look like the public test to the look-alike search? Compare how close the best same-type look-alike is
    (leave-one-out on the train pool, holdout -> train pool, public photos -> all training objects), then re-weight the holdout
    errors to the public photos' closeness profile."""
    rows, _ = read_metadata(metadata)
    config = json.loads((Path(out) / "config.json").read_text())
    train_rows, train_r = load_pool(out, rows, ("train",))
    all_rows, all_r = load_pool(out, rows, ("train", "val"))
    keys, matrix = load_features(feature_dir, config["spec"])
    raw = feature_rows(keys, matrix, [f"train/{r['id']}" for r in train_rows])
    a, _ = standardize(raw, raw[:1])
    cat = np.array([r["category"] for r in train_rows])
    group = np.array([r["group"] for r in train_rows])
    forbid = group[:, None] == group[None, :]
    best_loo = look_alikes(a @ a.T, cat, cat, np.log(train_r), forbid=forbid)[1]
    held = np.load(Path(out) / "holdout_sizes.npz")
    public, _ = guesses(config, feature_dir, all_rows, all_r, eval_rows(data), PUBLIC_FEATURE_KEY)
    groups = (("train pool, leave one out", best_loo), ("holdout -> train pool", held["best_similarity"]),
              ("public photos -> all 2,042", public["best_similarity"]))
    print("best same-type look-alike, cosine similarity percentiles 10/25/50/75/90:")
    for name, values in groups:
        print(f"  {name:28s} " + "  ".join(f"{p:.3f}" for p in np.percentile(values, [10, 25, 50, 75, 90])))
    edges = np.quantile(public["best_similarity"], [1 / 3, 2 / 3])
    bin_held, bin_public = np.digitize(held["best_similarity"], edges), np.digitize(public["best_similarity"], edges)
    print(f"\nbins are the thirds of the public photos' closeness (cut at {edges[0]:.3f} and {edges[1]:.3f}):")
    print("  bin     public share  holdout share   holdout size error: head            blend           (median %, squared mm2)")
    blend_mse = np.empty(3)
    head_mse = np.empty(3)
    for b, label in enumerate(("far", "middle", "close")):
        m = bin_held == b
        rel_head = np.abs(held["head"][m] / held["true"][m] - 1)
        rel_blend = np.abs(held["blend"][m] / held["true"][m] - 1)
        head_mse[b] = np.mean((held["head"][m] - held["true"][m]) ** 2)
        blend_mse[b] = np.mean((held["blend"][m] - held["true"][m]) ** 2)
        print(f"  {label:7s} {np.mean(bin_public == b) * 100:8.0f}%    {m.mean() * 100:8.0f}% ({m.sum():3d})"
              f"          {np.median(rel_head) * 100:5.1f}%  {head_mse[b]:6.1f}        {np.median(rel_blend) * 100:5.1f}%  {blend_mse[b]:6.1f}")
    share = np.array([np.mean(bin_public == b) for b in range(3)])
    plain = (np.mean((held["head"] - held["true"]) ** 2), np.mean((held["blend"] - held["true"]) ** 2))
    print(f"\nsquared radius error, holdout as measured: head {plain[0]:.1f}, blend {plain[1]:.1f}")
    print(f"squared radius error, holdout re-weighted to the public photos' closeness: head {share @ head_mse:.1f}, blend {share @ blend_mse:.1f}")


def score_job(job):
    data, row, unit_mm, unit_cloud, radii = job
    truth = sample_truth((data, row, unit_mm))
    place, scorer = metric_points(), aligned_cd()
    return [scorer(place(unit_cloud, radius), truth) for radius in radii]


def cmd_score(data, metadata, out, lora_val, arms):
    rows, unit_mm = read_metadata(metadata)
    by_id = {r["id"]: r for r in rows}
    sizes = np.load(Path(out) / "holdout_sizes.npz")
    ids = sizes["ids"].tolist()
    with np.load(lora_val, allow_pickle=False) as z:
        # the holdout file of the first run has no "unit" array; metric_points recentres and rescales any cloud, so its own points do
        clouds = z["unit"] if "unit" in z.files else z["pts"]
        unit = dict(zip(z["ids"].tolist(), clouds))
    arms = arms or ["blend", "knn", "cat_median"]
    jobs = [(data, by_id[i], unit_mm, unit[i].astype(np.float64), [float(sizes[a][n]) for a in arms]) for n, i in enumerate(ids)]
    with ProcessPoolExecutor(14) as pool:
        scores = np.array(list(pool.map(score_job, jobs, chunksize=1)))
    np.savez(Path(out) / "holdout_cd.npz", ids=np.array(ids), arms=np.array(arms), cd=scores)
    for n, arm in enumerate(arms):
        print(f"{arm:12s} CD-L2 mean {scores[:, n].mean():7.2f}   median {np.median(scores[:, n]):7.2f}", flush=True)


def cmd_table(out, holdout):
    """Put the look-alike size and its CD next to the size head's on the holdout page: writes
    HOLDOUT/lora_vs_truth/index_lookalike.csv (index_html.py picks it up) and checks that this scorer reproduces the saved scores."""
    cd = np.load(Path(out) / "holdout_cd.npz")
    sizes = np.load(Path(out) / "holdout_sizes.npz")
    saved = np.load(Path(holdout) / "lora_scores.npz")
    ids, arms = cd["ids"].tolist(), cd["arms"].tolist()
    if ids != saved["ids"].tolist():
        raise ValueError("object order differs from lora_scores.npz")
    for arm, key in (("head", "metric"), ("true", "oracle")):
        if arm in arms:
            mine = cd["cd"][:, arms.index(arm)]
            print(f"check, arm '{arm}' against lora_scores.npz['{key}']: mean {mine.mean():.2f} vs {saved[key].mean():.2f}, "
                  f"largest difference {np.abs(mine - saved[key]).max():.3f}")
    column = cd["cd"][:, arms.index("blend")]
    path = Path(holdout) / "lora_vs_truth" / "index_lookalike.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "lookalike_radius_mm", "score_at_lookalike_size"])
        for n, sid in enumerate(ids):
            writer.writerow([sid, f"{sizes['blend'][n]:.2f}", f"{column[n]:.3f}"])
    print(f"wrote {path}: look-alike size CD mean {column.mean():.2f}, median {np.median(column):.2f}")


def cmd_neighbours(data, feature_dir, metadata, out, lora_eval):
    """The look-alikes behind each public photo's size guess -> OUT_DIR/public_neighbours.csv (read by visualize_public_lookalike.py).
    Recomputes the guesses and checks that they equal the radii already in public_lora_lookalike.npz."""
    rows, _ = read_metadata(metadata)
    config = json.loads((Path(out) / "config.json").read_text())
    pool_rows, pool_r = load_pool(out, rows, ("train", "val"))
    queries = eval_rows(data)
    sizes, _ = guesses(config, feature_dir, pool_rows, pool_r, queries, PUBLIC_FEATURE_KEY)
    with np.load(Path(out) / "public_lora_lookalike.npz", allow_pickle=False) as z:
        built = dict(zip(z["ids"].tolist(), z["radius_mm"].tolist()))
    if not np.allclose([built[q["id"]] for q in queries], sizes["blend"], rtol=1e-5):
        raise ValueError("the recomputed guesses differ from public_lora_lookalike.npz")
    with np.load(lora_eval, allow_pickle=False) as z:
        head = dict(zip(z["ids"].tolist(), z["radius_mm"].tolist()))
    columns = ["id", "category", "head_radius_mm", "lookalike_radius_mm", "formula_radius_mm", "final_radius_mm", "best_similarity",
               "neighbour_gap"]
    for n in (1, 2):
        columns += [f"nn{n}_id", f"nn{n}_similarity", f"nn{n}_radius_mm"]
    with open(Path(out) / "public_neighbours.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, q in enumerate(queries):
            line = [q["id"], q["category"], f"{head[q['id']]:.2f}", f"{sizes['knn'][i]:.2f}", f"{sizes['ridge'][i]:.2f}",
                    f"{sizes['blend'][i]:.2f}", f"{sizes['best_similarity'][i]:.4f}",
                    "" if np.isnan(sizes["neighbour_gap"][i]) else f"{sizes['neighbour_gap'][i]:.4f}"]
            for n in range(2):
                j = int(sizes["neighbours"][i][n])
                line += ["", "", ""] if j < 0 else [pool_rows[j]["id"], f"{sizes['neighbour_similarity'][i][n]:.4f}", f"{pool_r[j]:.2f}"]
            writer.writerow(line)
    print(f"wrote {Path(out) / 'public_neighbours.csv'} for {len(queries)} public photos")


def cmd_public(data, feature_dir, metadata, out, lora_eval):
    sys.path.insert(0, str(BASELINES))
    from geom import pack
    rows, _ = read_metadata(metadata)
    config = json.loads((Path(out) / "config.json").read_text())
    pool_rows, pool_r = load_pool(out, rows, ("train", "val"))
    with np.load(lora_eval, allow_pickle=False) as z:
        merged = {k: z[k] for k in z.files}
    category = {p.parent.name: p.parent.parent.name for p in (Path(data) / "eval").glob("*/*/*/rgb_industrial.png")}
    ids = [str(i) for i in merged["ids"]]
    if set(ids) != set(category) or len(ids) != 511:
        raise ValueError("the LoRA clouds and the public photos do not have the same 511 ids")
    query_rows = [dict(id=i, category=category[i]) for i in ids]
    sizes, fallback = guesses(config, feature_dir, pool_rows, pool_r, query_rows, PUBLIC_FEATURE_KEY)
    radius = sizes["blend"]
    place = metric_points()
    pts = np.stack([place(u.astype(np.float64), r) for u, r in zip(merged["unit"], radius)])
    if pts.shape != (511, 8192, 3) or not np.isfinite(pts).all():
        raise ValueError("bad scaled clouds")
    result = dict(merged, pts=pts, radius_mm=radius.astype(np.float32))
    np.savez(Path(out) / "public_lora_lookalike.npz", **result, **{f"guess_{k}": v for k, v in sizes.items()})
    pack(Path(out) / "public_lora_lookalike.zip", dict(zip(ids, pts)))
    head = merged["radius_mm"]
    print(f"{len(ids)} public photos, pool of {len(pool_rows)} training objects; {fallback} had no same-type neighbour")
    print(f"new radius: median {np.median(radius):.1f} mm (min {radius.min():.1f}, max {radius.max():.1f}); "
          f"size head: median {np.median(head):.1f} mm; median ratio new/head {np.median(radius / head):.3f}")
    print(f"best cosine similarity to the pool: median {np.median(sizes['best_similarity']):.3f}, 10th/90th percentile "
          f"{np.percentile(sizes['best_similarity'], 10):.3f}/{np.percentile(sizes['best_similarity'], 90):.3f}")


def main():
    command, args = sys.argv[1], sys.argv[2:]
    if command == "selftest":
        cmd_selftest()
    elif command == "labels":
        cmd_labels(*args)
    elif command == "select":
        cmd_select(*args)
    elif command == "holdout":
        cmd_holdout(*args)
    elif command == "table":
        cmd_table(*args)
    elif command == "neighbours":
        cmd_neighbours(*args)
    elif command == "twins":
        cmd_twins(*args)
    elif command == "score":
        cmd_score(*args[:4], args[4:])
    elif command == "public":
        cmd_public(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
