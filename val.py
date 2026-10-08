#!/usr/bin/env python3
"""Evaluate an ACDS-DETR checkpoint with the RT-DETR validator."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Checkpoint path.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="acds_detr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from release_utils import load_detector
    model = load_detector(Path(args.weights).expanduser())
    model.val(
        data=str(Path(args.data).expanduser()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        max_det=args.max_det,
        save_json=args.save_json,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
