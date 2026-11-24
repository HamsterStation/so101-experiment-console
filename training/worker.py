"""One 4080 inference session, connected over a private SSH byte stream."""
import contextlib
import hashlib
import json
import socket
import struct
import zlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

RUN = Path.home() / 'robotlab/outputs/smolvla-cube46-v03-20261123'
BUNDLE = RUN / 'deployment'
FRONT, WRIST = ('observation.images.camera1','observation.images.camera2')
WEIGHTS = 'ce5ccdafd87f6ee5087f7a6a1e1848af6c12e3634fcefd912f2c6993bf4485c5'

def fixture(i):
    with np.load(BUNDLE / f'case-{i}.npz', allow_pickle=False) as case:
        images = [torch.from_numpy(case[k]).cuda().unsqueeze(0).float()/255 for k in ('front','wrist')]
        return (torch.from_numpy(case['state']).cuda(), *images, torch.from_numpy(case['noise']).cuda()), case['ref']

def initialize():
    ck=(RUN/'best-policy').resolve()
    with (ck/'model.safetensors').open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=WEIGHTS: raise ValueError('Unexpected checkpoint')
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    assert manifest['checkpoint_sha256']==WEIGHTS and manifest['shape']==[1,40,6]
    policy=SmolVLAPolicy.from_pretrained(str(ck)).eval().cuda()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=str(ck))
    args,_=fixture(0)
    batch=pre({FRONT:args[1],WRIST:args[2],'observation.state':args[0],'task':[manifest['task']]})
    tokens,mask=batch[OBS_LANGUAGE_TOKENS],batch[OBS_LANGUAGE_ATTENTION_MASK]
    normal=pre.steps[-1].state_dict()
    inverse=post.steps[0].state_dict()
    mean,std=(normal['observation.state.'+k].cuda() for k in ('mean','std'))
    out_mean,out_std=(inverse['action.'+k].cuda() for k in ('mean','std'))
    def core(state,front,wrist,noise):
        images,masks=policy.prepare_images({FRONT:front,WRIST:wrist})
        padded=policy.prepare_state({'observation.state':(state-mean)/(std+1e-8)})
        pred=policy.model.sample_actions(images,masks,tokens,mask,padded,noise=noise)
        return pred[...,:6].float()*out_std+out_mean

    # Both inputs use the same fixed 512-square image grid with every patch present.
    # Reuse its positional encoding; pixel convolutions still run for every image.
    embed=policy.model.vlm_with_expert.get_vlm_model().vision_model.embeddings
    positions=[]
    hook=embed.position_embedding.register_forward_hook(lambda _,args,out: positions.append(out.detach().clone()))
    baseline=core(*args)
    hook.remove()
    assert len(positions)==2 and torch.equal(positions[0],positions[1])
    def fixed_grid(pixel_values,patch_attention_mask=None):
        return embed.patch_embedding(pixel_values).flatten(2).transpose(1,2)+positions[0]
    embed.forward=fixed_grid
    vision=policy.model.vlm_with_expert.get_vlm_model().vision_model
    side=policy.config.resize_imgs_with_padding[0]//vision.patch_size
    live_grid=torch.ones((1,side,side),device='cuda',dtype=torch.bool)
    def supply_grid(module, args, kwargs):
        kwargs['patch_attention_mask']=live_grid
        return args,kwargs
    vision.register_forward_pre_hook(supply_grid,with_kwargs=True)
    out=core(*args)
    torch.testing.assert_close(baseline,out,rtol=0,atol=0)
    # LeRobot constructs literal attention masks and Euler timesteps from Python.
    # Materialize those constants once before graph capture to avoid host copies.
    from unittest.mock import patch
    constants={}
    tensor_factory=torch.tensor
    def device_constant(value,*args,**kwargs):
        if not str(kwargs.get('device','')).startswith('cuda') or torch.is_tensor(value):
            return tensor_factory(value,*args,**kwargs)
        key=(repr(value),repr(args),repr(sorted(kwargs.items())))
        if key not in constants:
            constants[key]=tensor_factory(value,*args,**kwargs)
        return constants[key]
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with patch.object(torch,"tensor",device_constant),torch.cuda.stream(stream):
        for _ in range(3): core(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with patch.object(torch,"tensor",device_constant),torch.cuda.graph(graph):
        static_out=core(*args)
    def predict(inputs):
        for dest,src in zip(args,inputs): dest.copy_(src)
        graph.replay()
        return static_out
    errors=[]
    for i in range(3):
        case,ref=fixture(i)
        result=predict(case).cpu().numpy()
        delta=np.abs(result-ref)
        if result.shape!=(1,40,6) or not np.isfinite(delta).all() or delta.max()>1.5:
            raise ValueError(f'4080 fixture mismatch {i}: {delta.max()}')
        errors.append(float(delta.max()))
    times=[]
    for _ in range(10):
        start=time.perf_counter(); predict(case).cpu(); times.append(1000*(time.perf_counter()-start))
    print(json.dumps(dict(max_abs=errors,core_ms=times)),file=sys.stderr,flush=True)
    # Captured kernels hold device addresses. Keep their source tensors alive for the SSH session.
    predict.capture_owners=(core, constants, policy, tokens, mask, positions, live_grid)
    return predict

def exact(stream,n):
    buf=stream.read(n)
    if len(buf)!=n: raise EOFError('Incomplete observation')
    return buf

def serve(predict,wire):
    wire.write(b'SM46'+bytes.fromhex(WEIGHTS));wire.flush()
    while header:=wire.read(4):
        if len(header)!=4: raise EOFError('Incomplete header')
        req=struct.unpack('!I',header)[0]
        state=torch.from_numpy(np.frombuffer(exact(wire,24),dtype='<f4').copy()).unsqueeze(0).cuda()
        images=[]
        for _ in range(2):
            n,checksum=struct.unpack('!II',exact(wire,8))
            if not 0<n<=2000000: raise ValueError('Invalid image length')
            data=exact(wire,n)
            img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            if img is None or img.shape!=(480,640,3): raise ValueError('Invalid camera image')
            rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if zlib.crc32(memoryview(rgb))!=checksum: raise ValueError('双摄 RGB 校验失败')
            images.append(torch.from_numpy(rgb).cuda().permute(2,0,1).unsqueeze(0).contiguous().float()/255)
        noise=torch.randn((1,40,32),device='cuda')
        out=predict((state,*images,noise)).cpu().numpy()
        if not np.isfinite(out).all(): raise ValueError('Invalid action')
        wire.write(header+out.astype('<f4',copy=False).tobytes());wire.flush()

if __name__=='__main__':
    import os
    # LeRobot prints model-loading details to stdout. Reserve the SSH stdout for binary data.
    out=os.fdopen(os.dup(1),'wb');os.dup2(2,1)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=True):
        predict=initialize()
        if '--benchmark' not in sys.argv:
            class Wire:
                read=sys.stdin.buffer.read
                write=out.write
                flush=out.flush
            serve(predict,Wire)
