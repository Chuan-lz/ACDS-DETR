#!/usr/bin/env python3
"""Run ACDS-DETR inference on images, directories, or videos."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Checkpoint path.")
    parser.add_argument("--source", required=True, help="Image, directory, glob, or video source.")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="acds_detr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from release_utils import load_detector
    model = load_detector(Path(args.weights).expanduser())
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
    )


if __name__ == "__main__":
    main()
