#!/usr/bin/env python3
"""Report ACDS-DETR parameters and direct-input THOP operation estimates."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="configs/ACDS_DETR.yaml", help="Model YAML or checkpoint path.")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import RTDETR
    import torch
    from copy import deepcopy
    from thop import profile
    from ultralytics.utils.torch_utils import get_num_params

    source = Path(args.model).expanduser()
    if not source.is_absolute():
        source = ROOT / source

    if source.suffix == '.pt':
        from release_utils import load_detector
        wrapper = load_detector(source)
    else:
        wrapper = RTDETR(str(source))
    wrapper.model.to(args.device)
    params = get_num_params(wrapper.model)
    sample = torch.zeros(1, 3, args.imgsz, args.imgsz, device=args.device)
    with torch.inference_mode():
        macs, _ = profile(deepcopy(wrapper.model).eval(), inputs=(sample,), verbose=False)
    gflops = 2 * macs / 1e9

    print(f"Parameters: {params / 1e6:.3f} M")
    print(f"GFLOPs at {args.imgsz}x{args.imgsz}: {gflops:.3f}")
    print("THOP estimate, 2 operations per MAC; uncounted attention/functional operations may remain.")


if __name__ == "__main__":
    main()
