"""Compare all three TensorRT engines on the same Jetson, without opening hardware."""
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import tensorrt as trt
import torch

ROOT=Path(__file__).resolve().parent
LEGACY=Path.home()/'cie6007-edge/act-shadow/cube46-v03-20261123-step-005000'
LABELS=('baseline','student','student-w8a8')
LOG=trt.Logger(trt.Logger.ERROR)


class Policy:
    def __init__(self,label,inputs):
        self.path=ROOT/(label+'.engine')
        self.runtime=trt.Runtime(LOG)
        self.engine=self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:raise ValueError(f'Cannot load {label}')
        self.ctx=self.engine.create_execution_context()
        self.output=torch.empty((1,40,6),device='cuda',dtype=torch.float32)
        assert self.engine.num_io_tensors==4
        for name,tensor in dict(**inputs,actions=self.output).items():
            assert tuple(self.engine.get_tensor_shape(name))==tuple(tensor.shape)
            assert self.engine.get_tensor_dtype(name)==trt.DataType.FLOAT
            assert self.ctx.set_tensor_address(name,tensor.data_ptr())
        inspector=self.engine.create_engine_inspector()
        inspector.execution_context=self.ctx
        info=json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        (ROOT/(label+'-layers.json')).write_text(json.dumps(info))
        layers=info['Layers'] if isinstance(info,dict) else info
        def int8_io(layer):
            io=layer.get('Inputs',[])+layer.get('Outputs',[])
            return any(isinstance(x,dict) and 'int8' in x.get('Format/Datatype','').lower() for x in io)
        self.profile=dict(engine_bytes=self.path.stat().st_size,context_bytes=self.engine.device_memory_size_v2,
            layer_count=len(layers),int8_io_layers=sum(int8_io(x) for x in layers),
            sample_int8_layers=[x.get('Name','')[:110] for x in layers if int8_io(x)][:5])

    def run(self):
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT failed')


@contextmanager
def frames(path):
    ffmpeg=Path.home()/'miniconda3/envs/lerobot/bin/ffmpeg'
    proc=subprocess.Popen([str(ffmpeg),'-nostdin','-v','error','-threads','1','-i',str(path),'-map','0:v:0',
                           '-pix_fmt','rgb24','-threads','1','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE)
    try:
        yield proc.stdout
        if proc.stdout.read(1):raise ValueError(f'extra frames in {path}')
        if proc.wait(timeout=12):raise RuntimeError(f'decode failed: {path}')
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate();proc.wait()


def metrics(values):
    return {k:float(v) for k,v in zip(('p50','p95'),np.percentile(values,[50,95]))}


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    inputs={name:torch.empty(shape,device='cuda',dtype=torch.float32) for name,shape in
            [('state',(1,6)),('front',(1,3,480,640)),('wrist',(1,3,480,640))]}
    policies={k:Policy(k,inputs) for k in LABELS}
    manifest=json.loads((ROOT/'edge-test/test.json').read_text())
    scales=np.asarray(manifest['action_std'],dtype=np.float64)
    assert scales.shape==(6,) and np.all(scales>0)
    checks={}
    for file in sorted(LEGACY.glob('act-input-*.npz')):
        with np.load(file) as case:
            for k in inputs:inputs[k].copy_(torch.from_numpy(case[k]))
            policies['baseline'].run()
            err=float(np.max(np.abs(policies['baseline'].output.cpu().numpy()-case['ref'])))
            assert err<.003,err
            checks[file.name]=err
    # Warm all engines, then alternate order on real dual-camera data. Exclude I/O and host transfers.
    for i in range(24):
        for p in policies.values():p.run()
    torch.cuda.synchronize()
    timed={k:[] for k in LABELS};gpu_timed={k:[] for k in LABELS}
    start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
    n=valid_targets=0
    total={k:np.zeros(6,np.float64) for k in LABELS}
    regions={}
    for row in manifest['episodes']:
        ep,zone,count=row['ep'],row['zone'],row['length']
        target=np.load(ROOT/f'edge-test/ep-{ep}.npz')
        region={k:np.zeros(6,np.float64) for k in LABELS};region_count=0
        with ExitStack() as stack:
            readers=[stack.enter_context(frames(ROOT/f'edge-test/videos/chunk-000/observation.images.camera{cam}/episode_{ep:06}.mp4')) for cam in [1,2]]
            for i in range(count):
                inputs['state'].copy_(torch.from_numpy(target['state'][i].copy())[None])
                for key,reader in zip(('front','wrist'),readers):
                    data=reader.read(480*640*3)
                    if len(data)!=480*640*3:raise EOFError(f'{ep}/{i}/{key}')
                    rgb=torch.from_numpy(np.frombuffer(data,np.uint8).reshape(480,640,3).copy()).cuda()
                    inputs[key].copy_(rgb.permute(2,0,1).unsqueeze(0).float()/255)
                torch.cuda.synchronize()
                horizon=min(40,count-i)
                truth=target['action'][i:i+horizon]
                # Rotate the starting model so every group sees every test frame once.
                for key in LABELS[n%3:]+LABELS[:n%3]:
                    p=policies[key];start=time.perf_counter();start_event.record()
                    p.run();end_event.record();end_event.synchronize()
                    elapsed=(time.perf_counter()-start)*1000
                    timed[key].append(elapsed);gpu_timed[key].append(start_event.elapsed_time(end_event))
                    pred=p.output.cpu().numpy()[0]
                    if not np.isfinite(pred).all():raise ValueError(f'{key} produced nonfinite actions')
                    error=np.abs(pred[:horizon].astype(np.float64)-truth).sum(axis=0)
                    total[key]+=error;region[key]+=error
                valid_targets+=horizon;region_count+=horizon;n+=1
        regions[zone]={k:float((v/region_count/scales).mean()) for k,v in region.items()}
        print('finished',zone,n,flush=True)
    assert n==858
    report=dict(frames=n,valid_targets=valid_targets,calibration_source='training split only',
                baseline_fixture_max_error=checks,normalized_mae_by_zone=regions,models={})
    for k,p in policies.items():
        report['models'][k]=dict(**p.profile,normalized_mae=float((total[k]/valid_targets/scales).mean()),
            mae_per_joint=(total[k]/valid_targets).tolist(),latency_ms=metrics(timed[k]),kernel_ms=metrics(gpu_timed[k]),latency_n=n)
    target=ROOT/'edge-results.json'
    target.with_suffix('.tmp').write_text(json.dumps(report,indent=2));target.with_suffix('.tmp').replace(target)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
