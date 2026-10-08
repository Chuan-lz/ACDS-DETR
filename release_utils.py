"""Strict checkpoint loading for ACDS-DETR.

Full-object .pt files are pickle-based: ONLY load trusted checkpoints.
This helper never silently drops mismatched parameter keys.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_weights_strict(detector, path):
    import torch
    from torch import nn
    from ultralytics.nn.extra_modules.acds import SpatialScaleSelection, ACDSCueRouter

    checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a checkpoint dictionary or a parameter dictionary.')
    source = checkpoint.get('ema')
    if source is None:
        source = checkpoint.get('model')
    if isinstance(source, nn.Module):
        if source.model[-1].num_queries != 300 or source.model[-1].nl != 3:
            raise ValueError('ACDS-DETR requires 300 queries and three decoder feature levels.')
        selectors = [m for m in source.modules() if isinstance(m, SpatialScaleSelection)]
        routers = [m for m in source.modules() if isinstance(m, ACDSCueRouter)]
        if len(selectors) != 6 or len(routers) != 1:
            raise ValueError('The checkpoint architecture does not match ACDS-DETR.')
        for module in selectors:
            if [branch.conv.dilation for branch in module.branches] != [(3,3),(5,5),(7,7)]:
                raise ValueError('ACDS-DETR uses dilation rates {3,5,7}.')
            if any(score.out_channels != 1 for score in module.score):
                raise ValueError('ACDS-DETR uses channel-shared scale selection.')
        state = source.float().state_dict()
        detector.model.names = source.names
        detector.model.args = dict(getattr(source, 'args', {}))
    else:
        state = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError('Expected model/ema or state_dict containing ACDS-DETR parameters.')
        detector.model.args = dict(getattr(detector.model, 'args', {}))
    detector.model.load_state_dict(state, strict=True)
    detector.model.args.update(model=str(ROOT/'configs/ACDS_DETR.yaml'), task='detect', nwd_gain=0.0)
    detector.overrides.update(model=str(ROOT/'configs/ACDS_DETR.yaml'),task='detect')
    return detector


def load_detector(path):
    from ultralytics import RTDETR
    return load_weights_strict(RTDETR(str(ROOT/'configs/ACDS_DETR.yaml')), path)
