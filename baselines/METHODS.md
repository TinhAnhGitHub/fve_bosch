# Food type and freshness

Public score to beat: **0.75** (team claude). Overall = `0.6 × macro mAP + 0.4 × max(0, macro QWK)`.

mAP is COCO-style, IoU 0.50 to 0.95, mean over all 26 foods. QWK is mean over the 10 Group A foods, and only on boxes that already match the right food at IoU ≥ 0.5. Grades are ordered: 0 fresh, 1 semi-fresh, 2 spoiled. A two-step miss counts four times a one-step miss.

1. **YOLO11m** (Jocher and Qiu, Ultralytics, 2024) for the box and the food type. COCO-pretrained, 20.1M parameters, trained at 1280. An image is repeated when its rarest food has fewer than 100 boxes, up to 8 times, so avocado, tomato, and the meats are not drowned by strawberry. No extra images.

2. **CORAL** (Cao, Mirjalili, Raschka, 2019, [arXiv:1901.07884](https://arxiv.org/abs/1901.07884)) for freshness. A MobileNetV3-Small crop encoder plus two shared-weight thresholds. Group B stays `freshness_grade: null`.

Not used: another team's weights, a retrieved training photo, or a 46-way softmax that treats spoiled and semi-fresh as unrelated labels.

Final ranking is the private board, and the model has to fit a T4: about 150 ms, 40M parameters, 6 GB. YOLO11m plus MobileNetV3-Small is about 23M. `ponytail:` 1280 is fixed; drop it if a TensorRT p95 on a T4 goes over 150 ms.
