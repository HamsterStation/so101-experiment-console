"""Check metadata before conversion and decode both cameras at representative frames afterwards."""
from pathlib import Path
import json, sys
root=Path('dataset').resolve(); info=json.loads((root/'meta/info.json').read_text()); phase=sys.argv[1]
keys=sorted(k for k,v in info['features'].items() if v['dtype']=='video'); assert len(keys)==2, keys
counts=(info['total_episodes'],info['total_frames']); assert all(x>0 for x in counts)
for k in ('action','observation.state'): assert info['features'][k]['shape']==[6], (k,info['features'][k])
if phase=='before':
    assert info['codebase_version']=='v2.1',info['codebase_version']; print('SOURCE',counts,keys)
elif phase=='after':
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torch
    old=json.loads((root.parent/'dataset_old/meta/info.json').read_text()); assert info['codebase_version']=='v3.0'
    assert [old[x] for x in ('total_episodes','total_frames','fps')]==[info[x] for x in ('total_episodes','total_frames','fps')]
    for k in [*keys,'action','observation.state']:assert old['features'][k]['shape']==info['features'][k]['shape'] and old['features'][k].get('names')==info['features'][k].get('names')
    ds=LeRobotDataset('local/so101_recording',root=root,video_backend='pyav'); n=0
    for ep in range(counts[0]):
        meta=ds.meta.episodes[ep];first,last=int(meta['dataset_from_index']),int(meta['dataset_to_index'])-1
        for i in sorted({first,(first+last)//2,last}):
            row=ds[i]; assert int(row['episode_index'])==ep
            for k in keys:assert row[k].ndim==3 and row[k].shape[0]==3 and torch.isfinite(row[k]).all(),(i,k)
            for k in ('action','observation.state'):assert tuple(row[k].shape)==(6,) and torch.isfinite(row[k]).all(),(i,k)
            n+=1
    report=dict(episodes=ds.num_episodes,frames=len(ds),cameras=keys,positions_checked=n,check='first/middle/last per episode'); print('CONVERSION_OK',report); (root.parent/'recording-check.json').write_text(json.dumps(report,indent=2)+'\n')
else: raise ValueError(phase)
