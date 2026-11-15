"""Build the Kaggle dataset folder for the SAM 3D / Hunyuan3D-Omni holdout test.

usage: python make_dataset.py DATA_DIR OUT_DIR
  DATA_DIR  the challenge data (train/<group>/<category>/<id>/rgb_industrial.png)
  OUT_DIR   new folder that becomes the dataset: hold204/<id>.png, controls.npz, sam3d_todo.txt, the run scripts

controls.npz holds RH-Mix5's holdout cloud per object (ids, pts in mm, float32), the control signal given to
Hunyuan3D-Omni (bounding box, voxels, points). sam3d_todo.txt lists the holdout ids SAM 3D has not done yet
(out/atlas/cloud/sam3d_hold.npz has 92 of 204).
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main(data: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    meta: dict = json.loads((BASE / "out/triposg/holdout/metadata.json").read_text())
    rows: list[dict] = [r for r in meta["rows"] if r["split"] == "val"]
    assert len(rows) == 204, len(rows)
    (out / "hold204").mkdir()
    for r in rows:
        photo: Path = next((data / "train").glob(f"*/*/{r['id']}/rgb_industrial.png"))
        shutil.copyfile(photo, out / "hold204" / f"{r['id']}.png")
    ids: list[str] = [r["id"] for r in rows]
    with np.load(BASE / "out/triposg/holdout/rhmix5_val.npz") as z:
        pos: dict[str, int] = {str(i): n for n, i in enumerate(z["ids"])}
        pts: np.ndarray = z["pts"][[pos[i] for i in ids]].astype(np.float32)
    np.savez(out / "controls.npz", ids=np.array(ids), pts=pts)
    with np.load(BASE / "out/atlas/cloud/sam3d_hold.npz") as z:
        done: set[str] = {str(i) for i, p in zip(z["ids"], z["pts_mesh"]) if np.isfinite(p).all()}
    todo: list[str] = [i for i in ids if i not in done]
    (out / "sam3d_todo.txt").write_text("\n".join(todo) + "\n")
    for f in ("run_omni.py", "run_sam3d.py"):
        shutil.copyfile(HERE / f, out / f)
    for f in ("gen_sam3d.py", "fix_dino.py", "fix_flow_matching_model.py"):
        shutil.copyfile(BASE / "sam3d" / f, out / f)
    print(f"{len(ids)} photos, controls {pts.shape}, SAM 3D still to do {len(todo)} -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
