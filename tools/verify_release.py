"""CPU architecture/backward checks; optional numerical comparison to a trusted reference.

Run from the release root. This does not launch a training experiment.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from ultralytics import RTDETR
    from ultralytics.nn.extra_modules.acds import ACDSSGEEncoder, ACDSCueRouter, SpatialScaleSelection
    from ultralytics.utils import DEFAULT_CFG_DICT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', help='Trusted ACDS-DETR checkpoint.')
    parser.add_argument('--reference', help='Trusted tensor archive from the original implementation.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    wrapper = RTDETR(str(ROOT/'configs/ACDS_DETR.yaml'))
    if args.weights:
        from release_utils import load_weights_strict
        load_weights_strict(wrapper, args.weights)
    net = wrapper.model.float().cpu().eval()
    stages = [m for m in net.modules() if isinstance(m, ACDSSGEEncoder)]
    assert [m.cv3.conv.out_channels for m in stages] == [128,256,384,384]
    assert [len(m.m) for m in stages] == [1,1,1,3]
    assert sum(isinstance(m, ACDSCueRouter) for m in net.modules()) == 1
    assert sum(p.numel() for p in net.parameters()) == 13781709
    assert net.model[-1].num_queries == 300 and net.model[-1].nl == 3
    reference = torch.load(args.reference, map_location='cpu', weights_only=False) if args.reference else None
    differences = {}
    for size in [128,416,768]:
        shapes = []
        handle = net.model[-1].register_forward_pre_hook(lambda m,x: shapes.extend([list(t.shape) for t in x[0]]))
        torch.manual_seed(20261008+size)
        with torch.inference_mode():
            output = net(torch.rand(1,3,size,size))[0]
        handle.remove()
        assert list(output.shape) == [1,300,5] and torch.isfinite(output).all()
        assert shapes == [[1,256,size//s,size//s] for s in (8,16,32)]
        if reference:
            torch.testing.assert_close(output, reference['outputs'][size],rtol=1e-5,atol=1e-6)
            differences[size] = float((output-reference['outputs'][size]).abs().max())
        for module in net.modules():
            if isinstance(module,SpatialScaleSelection):
                weights = module.last_scale_weights
                assert weights.shape[1:3] == (3,1)
                torch.testing.assert_close(weights.sum(1),torch.ones_like(weights[:,0]))
        print('FORWARD PASS',size,shapes,flush=True)
    net.train()
    net.nc = 1
    net.args = {**DEFAULT_CFG_DICT,'imgsz':128,'nwd_gain':0.0}
    if hasattr(net,'criterion'):
        del net.criterion
    torch.manual_seed(42)
    batch = dict(img=torch.rand(2,3,128,128),batch_idx=torch.tensor([0,1]),
                 cls=torch.zeros(2,1),bboxes=torch.tensor([[.5,.5,.2,.2],[.4,.6,.15,.15]]))
    loss, items = net(batch)
    assert torch.isfinite(loss) and torch.isfinite(items).all()
    loss.backward()
    for parameter in net.parameters():
        if parameter.grad is not None: assert torch.isfinite(parameter.grad).all()
    assert stages[0].m[0].context.score[0].weight.grad is not None
    router = next(m for m in net.modules() if isinstance(m,ACDSCueRouter))
    assert router.reliability_gate.gate[1].weight.grad is not None
    print(json.dumps(dict(status='PASS',parameters=13781709,queries=300,
              loss=float(loss.detach()),reference_max_absolute_errors=differences,
              source_sha256=reference['source_sha256'] if reference else None),indent=2))


if __name__ == '__main__':
    main()
