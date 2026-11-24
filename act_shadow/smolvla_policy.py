"""Local CUDA worker for the four-position SmolVLA. All coordinates use LeRobot units."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from run_shadow import take_image, file_hash

JOINTS = [name + ".pos" for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]

def fixture(path):
    with np.load(path, allow_pickle=False) as case:
        state, front, wrist, noise = [torch.from_numpy(case[key]).cuda() for key in ('state', 'front', 'wrist', 'noise')]
        return (state, front.unsqueeze(0).float().div_(255), wrist.unsqueeze(0).float().div_(255), noise), case['ref']

def run(bundle, benchmark):
    meta = json.loads((bundle / 'manifest.json').read_text())
    assert meta['joints'] == JOINTS and meta['shape'] == [1,40,6] and meta['fps'] == 20
    assert meta['language_layers'] == meta['expert_layers'] == 16 and meta['num_steps'] == 10
    for name, digest in meta['artifact_sha256'].items():
        if file_hash(bundle / name) != digest:
            raise ValueError(f'模型校验失败 {name}')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = torch.jit.load(str(bundle / 'smolvla-core.pt'), map_location='cuda').eval()
    static_args, _ = fixture(bundle / 'case-0.npz')
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        def predict(args):
            return model(*args)
        for _ in range(2):
            predict(static_args)
        torch.cuda.synchronize()
        errors = []
        for n in range(3):
            inputs, ref = fixture(bundle / f'case-{n}.npz')
            prediction = predict(inputs).cpu().numpy()
            delta = np.abs(prediction - ref)
            if prediction.shape != (1,40,6) or not np.isfinite(delta).all() or delta.max() > 1.5:
                raise ValueError(f'4080 参考动作不匹配 {n}: {delta.max()}')
            errors.append(dict(max=float(delta.max()), mean=float(delta.mean())))
        if benchmark:
            times = []
            for _ in range(10):
                start = time.perf_counter()
                predict(inputs).cpu()
                times.append((time.perf_counter()-start)*1000)
            print(json.dumps(dict(parity=errors, times_ms=times, cuda=torch.cuda.get_device_name()),indent=2))
            return
        stream, out = sys.stdin.buffer, sys.stdout.buffer
        out.write(b'R')
        out.flush()
        while raw := stream.read(24):
            if len(raw) != 24:
                raise EOFError('关节状态不完整')
            state = torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy()).cuda().unsqueeze(0)
            args = (state, take_image(stream), take_image(stream), torch.randn((1,40,32),device='cuda'))
            output = predict(args).cpu().numpy()
            if output.shape != (1,40,6) or not np.isfinite(output).all():
                raise ValueError('模型输出无效')
            out.write(output.astype('<f4',copy=False).tobytes())
            out.flush()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    run(args.bundle,args.benchmark)
