"""Local: the 511 public photos at the encoder's input size (256 px, bicubic, as atlas.image_tensor), in a tar.

On the VM image_tensor's resize to 256 is then a no-op (PIL copies an image already at the target size), so the
tokens match a run on the full photos. usage: pack_photos.py DATA_ROOT OUT.tar   (DATA_ROOT holds eval/ or val/)
"""
import io
import sys
import tarfile
from pathlib import Path

from PIL import Image

root, out = Path(sys.argv[1]), Path(sys.argv[2])
paths = sorted((root / "eval").rglob("rgb_industrial.png"))
assert len(paths) == 511, len(paths)
with tarfile.open(out, "w") as tar:
    for p in paths:
        buf = io.BytesIO()
        Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC).save(buf, format="PNG")
        info = tarfile.TarInfo(str(p.relative_to(root)))
        info.size = buf.tell()
        buf.seek(0)
        tar.addfile(info, buf)
print(f"{len(paths)} photos -> {out}")
