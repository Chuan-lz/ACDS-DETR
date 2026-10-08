#!/usr/bin/env python3
"""Train ACDS-DETR for crater detection."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--model", default="configs/ACDS_DETR.yaml", help="Model YAML path.")
    parser.add_argument("--recipe", choices=("pcdd", "lunar-stable"), default="pcdd")
    parser.add_argument("--pretrained", default=None, help="Optional compatible checkpoint used for initialization.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="acds_detr")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import RTDETR

    model = RTDETR(str(resolve_path(args.model)))
    if args.pretrained:
        from release_utils import load_weights_strict
        load_weights_strict(model, resolve_path(args.pretrained))

    model.train(
        data=str(resolve_path(args.data)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        max_det=args.max_det,
        project=str(resolve_path(args.project)),
        name=args.name,
        exist_ok=args.exist_ok,
        patience=0,
        seed=args.seed,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.1 if args.recipe == "lunar-stable" else 1.0,
        cos_lr=args.recipe == "lunar-stable",
        momentum=0.9,
        weight_decay=1e-4,
        # This inherited trainer interprets warmup_epochs as iterations.
        warmup_epochs=125 if args.recipe == "lunar-stable" else 2000,
        warmup_bias_lr=0.0 if args.recipe == "lunar-stable" else 0.1,
        amp=args.amp,
        augment=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        random_crop=0.0,
        nwd_gain=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )


if __name__ == "__main__":
    main()
