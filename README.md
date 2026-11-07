# Food type and freshness

One fridge photo in, one COCO-results JSON out. Score = `0.6 × macro mAP + 0.4 × macro QWK`.

* `baselines/fresh.py` — YOLO11m boxes and types, CORAL freshness on Group A.
* `baselines/METHODS.md` — why this baseline.

```
python baselines/fresh.py prep
python baselines/fresh.py train
python baselines/fresh.py fresh
python baselines/fresh.py predict
```

Put the public zip contents in `data/ftaf/` (`train/annotations.json`, `train/images`, `eval/images`). Do not commit the data or the prediction JSON.
