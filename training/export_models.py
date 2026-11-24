import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from safetensors.torch import load_file
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from train_student import BASE, CHECKPOINT, OUT, LiveACT, sha


class Stats:
    def __init__(self,ck,kind):
        stages=json.loads((ck/f'policy_{kind}.json').read_text())['steps']
        s=next(s for s in stages if 'state_file' in s)
        self.values=load_file(str(ck/s['state_file']))
        self.eps=s['config']['eps']

    def state_dict(self):
        return self.values


class StaticACT(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.m=original
        c=original.core
        # Both physical cameras always deliver the same spatial grid.
        feature=c.backbone(torch.zeros(1,3,480,640))['feature_map']
        pos=c.encoder_cam_feat_pos_embed(feature).flatten(2).permute(2,0,1)
        token_pos=c.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
        self.register_buffer('pos',torch.cat((token_pos,pos,pos)).detach())
        self.register_buffer('latent',c.encoder_latent_input_proj(torch.zeros(1,c.config.latent_dim)).detach().unsqueeze(0))
        self.register_buffer('decoder_in',torch.zeros(40,1,c.config.dim_model))

    def forward(self,state,front,wrist):
        m=self.m
        c=m.core
        s=c.encoder_robot_state_input_proj((state-m.state_mean)/(m.state_std+m.eps)).unsqueeze(0)
        tokens=[self.latent,s]
        for x,mean,std in ((front,m.front_mean,m.front_std),(wrist,m.wrist_mean,m.wrist_std)):
            features=c.backbone((x-mean)/(std+m.eps))['feature_map']
            tokens.append(c.encoder_img_feat_input_proj(features).flatten(2).permute(2,0,1))
        enc=c.encoder(torch.cat(tokens,dim=0),pos_embed=self.pos)
        dec=c.decoder(self.decoder_in,enc,encoder_pos_embed=self.pos,decoder_pos_embed=c.decoder_pos_embed.weight.unsqueeze(1))
        return c.action_head(dec.transpose(0,1))*m.action_std+m.action_mean


def build(label):
    ck=CHECKPOINT if label=='baseline' else OUT/'best_student'
    cfg=ACTConfig.from_pretrained(str(ck));cfg.device='cpu';cfg.pretrained_backbone_weights=None
    policy=ACTPolicy.from_pretrained(str(ck),config=cfg).cpu().eval()
    original=LiveACT(policy,Stats(ck,'preprocessor'),Stats(ck,'postprocessor')).eval()
    simple=StaticACT(original).eval()
    xs=[]
    for path in sorted((BASE/'deployment/step-005000').glob('act-input-*.npz')):
        with np.load(path) as d:
            inputs=tuple(torch.from_numpy(d[k]) for k in ('state','front','wrist'))
            ref=original(*inputs)
            if label=='baseline': np.testing.assert_allclose(ref.numpy(),d['ref'],atol=.002,rtol=0)
            assert (simple(*inputs)-ref).abs().max().item()<.002
            xs.append((inputs,ref.numpy()))
    assert len(xs)==2
    dest=OUT/f'{label}.onnx'
    torch.onnx.export(simple,xs[0][0],str(dest),input_names=['state','front','wrist'],output_names=['actions'],
        dynamo=False,opset_version=17,do_constant_folding=True)
    graph=onnx.load(dest)
    onnx.checker.check_model(graph)
    opts=ort.SessionOptions();opts.intra_op_num_threads=3;opts.inter_op_num_threads=1
    sess=ort.InferenceSession(str(dest),opts,providers=['CPUExecutionProvider'])
    errors=[]
    for inputs,ref in xs:
        y=sess.run(None,dict(zip(('state','front','wrist'),(x.numpy() for x in inputs))))[0]
        errors.append(float(np.max(np.abs(y-ref))))
    assert max(errors)<.003, errors
    print(json.dumps(dict(model=label,sha256=sha(dest),fixture_errors=errors,ops=Counter(n.op_type for n in graph.graph.node))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('label',choices=['student','baseline'])
    torch.set_num_threads(3);torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad(): build(parser.parse_args().label)
