# ACDS-DETR

Code for **ACDS-DETR**, an RT-DETR-based detector for dense cross-scale crater detection.

ACDS-DETR changes the feature hierarchy while retaining RT-DETR query selection, Transformer decoding, bipartite matching, and the detection objective. Its two main components are:

- **Spatial-SGE**, which performs location-wise selection among dilated receptive-field responses and spatially refines the selected evidence.
- **RCR**, which aligns the shallow `P2` feature to `P3` resolution and conditionally routes it into the three-level decoder-visible hierarchy without adding a fourth prediction level.

This repository contains source code and model configuration only. Datasets, checkpoints, experiment logs, and generated figures are not included.

RCR uses context-conditioned shallow-feature routing and concatenation
followed by RepC3. P2/P3/P4/P5 channels are 128/256/384/384, stage depths
are 1/1/1/3, and the decoder has 300 queries and three input feature levels.
Spatial-SGE uses dilation rates {3,5,7}, location-wise channel-shared selection,
and spatial evidence refinement. Training uses the RT-DETR detection objective.

## Installation

Use Python 3.10 or newer in an isolated environment. The numerical checks use
Python 3.10 and PyTorch 2.1.1. Install a matching PyTorch/torchvision pair for
your CUDA version, then install this local fork in editable mode. Do not
install the stock `ultralytics` package into the same environment.

```bash
git clone https://github.com/Chuan-lz/ACDS-DETR.git
cd ACDS-DETR
pip install -e .
```

## Dataset format

Annotations follow the standard Ultralytics detection format. Each image has a matching text file containing normalized labels:

```text
class_id center_x center_y width height
```

A typical directory layout is:

```text
PCDD/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Copy [`configs/dataset_example.yaml`](configs/dataset_example.yaml), then replace its dataset path with your local path.

## Training

The default command follows the paper model configuration and the 768-pixel, 150-epoch training protocol:

```bash
python train.py --data configs/pcdd.yaml --device 0
```

To initialize from a compatible checkpoint:

```bash
python train.py --data configs/pcdd.yaml --pretrained /path/to/checkpoint.pt --device 0
```

The canonical architecture is defined in [`configs/ACDS_DETR.yaml`](configs/ACDS_DETR.yaml).
Copy the dataset example to `configs/pcdd.yaml` and edit its path first.
The default PCDD recipe is batch 8, AdamW, lr=1e-4, 150 epochs, seed 0, FP32,
and horizontal flip probability 0.5; crop/HSV/scale/mosaic/mixup are disabled.
In this inherited fork, `warmup_epochs` is interpreted as **iterations**:
the PCDD setting is 2000 iterations, not 2000 epochs.

For the separate optimized lunar protocol:

```bash
python train.py --recipe lunar-stable --data configs/lunar.yaml --imgsz 416 --device 0
```

This selects 125 warmup iterations, zero initial bias LR, and cosine decay
to 0.1 of the initial LR. Use 768 to reproduce the enlarged-input protocol.
For strictly native 416 training, prepare all images at 416x416; pad any smaller
non-square image without resizing and update its labels. The inherited RT-DETR
loader otherwise stretches non-square images to the requested square size.

## Evaluation

```bash
python val.py \
  --weights /path/to/best.pt \
  --data configs/pcdd.yaml \
  --device 0
```

Use `--split test` to evaluate the test split and `--save-json` to export COCO-style predictions.

`val.py` reports the inherited training-validator metrics. These are not
identical to an independent pycocotools evaluation. The model's 300 queries
are an architectural setting; COCO `maxDets` is a separate evaluation limit.
Passing `--max-det 300` here does not establish COCO AP@300. State the evaluator,
IoU thresholds, original-coordinate area bins and maxDets when reporting results.
Do not call a validation split an independent test split if both paths alias.

## Inference

```bash
python predict.py \
  --weights /path/to/best.pt \
  --source /path/to/images \
  --device 0
```

## Parameters and FLOPs

```bash
python profile_model.py --imgsz 768 --device cpu
```

The reported FLOPs depend on input resolution, so comparisons should use the same `--imgsz` value.
The profiler runs THOP on the actual input shape and reports 2 operations per MAC.
This is an estimate: some functional/attention operations are not counted.

## Checkpoints

`val.py`, `predict.py`, and training with `--pretrained` build the canonical
ACDS-DETR model and load parameter keys strictly. Use checkpoints saved by
this implementation or compatible parameter dictionaries. Architecture
mismatches are rejected. Full-object `.pt` files use pickle and
must only be loaded from trusted sources. Dataset/checkpoint files are not
bundled with this code release.

## Verification

```bash
python tools/verify_release.py
```

This checks the stage structure, 300 queries, three decoder levels, finite
forward outputs at 128/416/768, normalized scale weights, and a synthetic
training-loss backward pass. It does not launch a training experiment.

## Code map

- `ultralytics/nn/extra_modules/acds.py`: Spatial-SGE and RCR implementation.
- `configs/ACDS_DETR.yaml`: model definition used by the public entry points.
- `ultralytics/nn/tasks.py`: parser registration for the ACDS-DETR modules.
- `release_utils.py`: strict loading into the canonical architecture.
- `train.py`, `val.py`, and `predict.py`: minimal command-line entry points.

## Acknowledgements and license

This implementation is built on the RT-DETR support in [Ultralytics](https://github.com/ultralytics/ultralytics). The code is released under the GNU Affero General Public License v3.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
