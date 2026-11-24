from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
from train_student import DATA,OUT
root=DATA/'test_old'
info=json.loads((root/'meta/info.json').read_text())
split=json.loads((DATA/'split.json').read_text())
dest=OUT/'edge-test';dest.mkdir(exist_ok=True)
episodes=[]
for rec in sorted((x for x in split['episodes'] if x['split']=='test'), key=lambda x:x['split_index']):
    ep=rec['split_index']
    tab=pq.read_table(root/info['data_path'].format(episode_chunk=0,episode_index=ep))
    action=np.array(tab['action'].to_pylist(),np.float32)
    state=np.array(tab['observation.state'].to_pylist(),np.float32)
    np.savez(dest/f'ep-{ep}.npz',state=state,action=action)
    episodes.append(dict(ep=ep,zone=rec.get('zone',rec.get('batch')),length=len(state)))
assert sum(x['length'] for x in episodes)==858 and len(episodes)==4
scale=json.loads((DATA/'dataset/meta/stats.json').read_text())['action']['std']
(dest/'test.json').write_text(json.dumps(dict(episodes=episodes,action_std=scale),indent=2))
print(episodes)
