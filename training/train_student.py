import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

ROOT = Path.home() / 'robotlab'
DATA = ROOT / 'cie6007-recordings/run-cube46-v03-20261123'
BASE = ROOT / 'outputs/act-cube46-v03-20261123'
OUT = ROOT / 'outputs/act-cube46-comparison-v1'
CHECKPOINT = BASE / 'best-policy'
TEACHER_HASH = '2ded4e80934293e90ab6d1e40191280bda84e19c22a0bf7e0258f179d5051dd9'
WIDTH = 1600
from export_so101_act import LiveACT, CAMERAS


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def load(path):
    p = ACTPolicy.from_pretrained(str(path)).cuda().eval()
    pre,post = make_pre_post_processors(p.config, pretrained_path=str(path))
    return p, LiveACT(p,pre.steps[-1],post.steps[0]).cuda().eval()


def samples(part, stride):
    root = DATA / (part + '_old')
    info = json.loads((root/'meta/info.json').read_text())
    for episode in range(info['total_episodes']):
        keys = dict(episode_index=episode,episode_chunk=0)
        tab = pq.read_table(root/info['data_path'].format(**keys))
        state = np.array(tab['observation.state'].to_pylist(),dtype=np.float32)
        action = np.array(tab['action'].to_pylist(),dtype=np.float32)
        assert np.isfinite(state).all() and np.isfinite(action).all()
        paths = [root/info['video_path'].format(**keys,video_key=k) for k in CAMERAS]
        with av.open(str(paths[0])) as top, av.open(str(paths[1])) as wrist:
            num = 0
            for i,(x,y) in enumerate(zip(top.decode(video=0),wrist.decode(video=0),strict=True)):
                num += 1
                if i % stride:
                    continue
                indices = np.arange(i,i+40)
                yield (episode,i,state[i],x.to_ndarray(format='rgb24'),y.to_ndarray(format='rgb24'),
                       action[indices.clip(max=len(action)-1)],indices<len(action))
            assert num==len(state)


def batches(part,stride,size):
    batch=[]
    for row in samples(part,stride):
        batch.append(row)
        if len(batch)==size:
            yield batch
            batch=[]
    if batch:
        yield batch


def batch_values(batch):
    ep,frame,state,top,wrist,truth,valid=zip(*batch)
    dev=lambda x:torch.from_numpy(np.stack(x)).cuda()
    return (dev(state),*(dev(v).permute(0,3,1,2).contiguous().float()/255 for v in (top,wrist))), dev(truth),dev(valid)


class FeatureInput(nn.Module):
    def forward(self, image):
        return {'feature_map':image}


def record_features(model,part,stride,score):
    rows=[[],[],[],[],[],[]]
    features=[]
    hooks=[model.core.backbone.register_forward_hook(
        lambda module,args,result:features.append(result['feature_map'].detach().half().cpu()))]
    def record(layer):
        def hook(module,args):
            value=args[0].detach().reshape(-1,args[0].shape[-1])
            score[layer][0].add_(value.sum(0))
            score[layer][1] += value.shape[0]
        return hook
    if score is not None:
        for key,layer in layers(model.core):
            hooks.append(layer.linear2.register_forward_pre_hook(record(key)))
    try:
        with torch.no_grad():
            for batch in batches(part,stride,6):
                inputs,truth,valid=batch_values(batch)
                features.clear()
                pred=model(*inputs)
                assert len(features)==2
                for dest,val in zip(rows,((inputs[0]-model.state_mean)/(model.state_std+model.eps),
                        *features,(truth-model.action_mean)/(model.action_std+model.eps),valid,
                        (pred-model.action_mean)/(model.action_std+model.eps))):
                    dest.append(val.cpu())
    finally:
        for hook in hooks:hook.remove()
    return [torch.cat(xs) for xs in rows]


def layers(core):
    for group in ('encoder','decoder'):
        for i,layer in enumerate(getattr(core,group).layers):
            yield f'{group}.{i}',layer


@torch.no_grad()
def prune(layer,mean):
    a,b=layer.linear1,layer.linear2
    score=b.weight.norm(dim=0)*(mean if mean is not None else a.weight.norm(dim=1))
    order=score.argsort(descending=True)
    keep,cut=order[:WIDTH].sort().values,order[WIDTH:]
    first=nn.Linear(a.in_features,WIDTH,device=a.weight.device)
    last=nn.Linear(WIDTH,b.out_features,device=b.weight.device)
    first.weight.copy_(a.weight[keep]); first.bias.copy_(a.bias[keep])
    last.weight.copy_(b.weight[:,keep]); last.bias.copy_(b.bias)
    if mean is not None:last.bias.add_(b.weight[:,cut] @ mean[cut])
    layer.linear1,layer.linear2=first,last
    return keep.cpu().tolist()


def objective(core,cache,idx):
    state,front,wrist,truth,valid,teacher=[v[idx].cuda() for v in cache]
    pred=core({OBS_STATE:state,OBS_IMAGES:[front.float(),wrist.float()]})[0]
    abs_demo=(pred-truth).abs().mean(-1)
    abs_kd=(pred-teacher).abs().mean(-1)
    return abs_demo[valid], abs_kd[valid]


@torch.no_grad()
def validate(core,cache):
    total=count=0
    for idx in torch.arange(len(cache[0])).split(8):
        e,_=objective(core,cache,idx);total+=float(e.sum());count+=e.numel()
    return total/count


def main():
    assert sha(CHECKPOINT/'model.safetensors')==TEACHER_HASH
    assert not (OUT/'best_student').exists()
    torch.manual_seed(42); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    teacher,wrapped=load(CHECKPOINT)
    scores={k:[torch.zeros(3200,device='cuda'),0] for k,_ in layers(teacher.model)}
    print('Caching training features and teacher labels',flush=True)
    train=record_features(wrapped,'dataset',4,scores)
    print('Training frames',len(train[0]),flush=True)
    val=record_features(wrapped,'validation',4,None)
    student=copy.deepcopy(teacher).eval()
    keep={}
    for k,layer in layers(student.model):
        sums,n=scores[k];keep[k]=prune(layer,sums/n)
    # The training-only VAE shares dim_feedforward in the LeRobot config.
    for layer in student.model.vae_encoder.layers:prune(layer,None)
    student.config.dim_feedforward=WIDTH
    student.model.config.dim_feedforward=WIDTH
    backbone=student.model.backbone
    student.model.backbone=FeatureInput()
    student.model.vae_encoder.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in student.model.parameters() if p.requires_grad],lr=8e-6)
    base_param=sum(p.numel() for n,p in teacher.model.named_parameters() if not n.startswith('vae_'))
    compact_param=sum(p.numel() for n,p in student.model.named_parameters() if not n.startswith('vae_'))+sum(p.numel() for p in backbone.parameters())
    results={0:validate(student.model,val)}
    print('Pruned validation',results[0],flush=True)
    best=float('inf');best_step=None
    for step in range(1,501):
        ids=torch.randint(len(train[0]),(8,))
        demo,kd=objective(student.model,train,ids)
        loss=.5*demo.mean()+.5*kd.mean()
        if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),1.0)
        optimizer.step()
        if step%50==0:print('step',step,'loss',float(loss.detach()),flush=True)
        if step in (100,250,500):
            results[step]=validate(student.model,val)
            print('validation',step,results[step],flush=True)
            if results[step]<best:
                best,best_step=results[step],step
                student.model.backbone=backbone
                stage=OUT/f'candidate-{step:03}'
                student.save_pretrained(stage)
                for f in CHECKPOINT.glob('policy_*'):shutil.copy2(f,stage/f.name)
                student.model.backbone=FeatureInput()
    (OUT/'best_student').symlink_to(f'candidate-{best_step:03}')
    report=dict(teacher_sha256=TEACHER_HASH,student_sha256=sha(OUT/'best_student/model.safetensors'),split_sha256=sha(DATA/'split.json'),
        original_width=3200,student_width=WIDTH,inference_params=[base_param,compact_param],steps=500,best_step=best_step,validation_stride4=results,
        train_frames=len(train[0]),val_frames=len(val[0]),seed=42,demo_weight=.5,kd_weight=.5,batch=8,lr=8e-6,calibration='train only',
        teacher='zero-latent eval mode',backbone='frozen, FP16 cached feature maps; deployed backbone weights FP32',
        pruning='mean training ReLU activity times outgoing channel norm, omitted mean folded into bias')
    (OUT/'train_report.json').write_text(json.dumps(report,indent=2))
    (OUT/'retained_channels.json').write_text(json.dumps(keep))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
