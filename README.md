# ACDS-DETR

An RT-DETR-based framework for dense cross-scale crater detection in planetary imagery.

ACDS-DETR combines **Spatial-SGE** for location-wise receptive-field selection with **RCR** for context-conditioned shallow-feature routing.

## Installation

Use Python 3.10+ with a compatible PyTorch and torchvision installation. Install this repository in a separate environment.

```bash
git clone https://github.com/Chuan-lz/ACDS-DETR.git
cd ACDS-DETR
pip install -e .
```

## Dataset

Organize images and labels into matching `images/train`, `images/val`, `labels/train`, and `labels/val` directories. Each label row uses normalized coordinates:

```text
class_id center_x center_y width height
```

Copy [configs/dataset_example.yaml](configs/dataset_example.yaml) to `configs/pcdd.yaml` and update the dataset path and split paths. Datasets and pretrained weights are not included.

## Usage

Run the following commands from the repository root.

### Training

```bash
python train.py --data configs/pcdd.yaml --imgsz 768 --epochs 150 --batch 8 --device 0
```

### Evaluation

```bash
python val.py --weights /path/to/best.pt --data configs/pcdd.yaml --device 0
```

### Inference

```bash
python predict.py --weights /path/to/best.pt --source /path/to/images --device 0
```

Use a compatible ACDS-DETR checkpoint from a trusted source. Results are saved under `runs/`. Additional options are available with `--help`.

## Model

- [Model configuration](configs/ACDS_DETR.yaml)
- [Spatial-SGE and RCR implementation](ultralytics/nn/extra_modules/acds.py)

## Acknowledgements and License

Built on [Ultralytics](https://github.com/ultralytics/ultralytics) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR). Released under the GNU AGPL-3.0 license. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
