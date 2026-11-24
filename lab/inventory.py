"""Runs read-only via SSH. Sources are scoped to the experiment directories."""
import base64,json,os,shutil,subprocess,sys
from pathlib import Path
h=Path.home()
jet=sys.argv[1]=='jetson'
fixture=h/('cie6007-edge/checks/synthetic' if jet else 'robotlab/cie6007-recordings/run-synthetic/dataset')
parent=h/('cie6007-edge/lab/recordings' if jet else 'robotlab/cie6007-recordings')
roots=[fixture,*sorted(parent.glob('*' if jet else 'run-*/dataset'))]
roots=[p for p in dict.fromkeys(roots) if p.is_dir() and not p.is_symlink() and (p.resolve()==fixture or parent in p.resolve().parents)]
if len(sys.argv)>2:
    ds=Path(sys.argv[2])
    if ds not in roots: raise SystemExit('数据集不在已登记目录')
    m=json.loads((ds/'meta/info.json').read_text())
    images=[]
    for k,v in sorted(m['features'].items()):
        if v.get('dtype')!='video' or not k.startswith('observation.images.'): continue
        vid=next((f for f in sorted((ds/'videos').glob('**/*.mp4')) if k in f.parts and ds.resolve() in f.resolve().parents),None)
        if not vid: continue
        ff=str(h/'miniconda3/envs/lerobot/bin/ffmpeg') if jet else '/usr/bin/ffmpeg'
        image=subprocess.run([ff,'-v','error','-nostdin','-threads','1','-i',str(vid),'-frames:v','1','-vf','scale=320:-2','-c:v','mjpeg','-threads','1','-f','image2pipe','pipe:1'],capture_output=True,timeout=20)
        if image.returncode:raise RuntimeError('无法解码 '+k+': '+image.stderr.decode(errors='replace')[-200:])
        if image.stdout: images.append({'name':k,'src':'data:image/jpeg;base64,'+base64.b64encode(image.stdout).decode()})
    print(json.dumps({'images':images,'detail':f'首段录像的真实首帧 · {len(images)} 个视角'})); sys.exit()
datasets=[]
for p in roots:
    try:
        m=json.loads((p/'meta/info.json').read_text()); n=m['total_episodes']; frames=m['total_frames']
        if n<1 or frames<1:continue
        # Do not offer transfer/training for a session that is still collecting.
        if jet and p!=fixture:
            session=h/'cie6007-edge/lab/sessions'/p.name/'state.json'
            if not session.exists() or json.loads(session.read_text()).get('phase') not in ('closed','failed'): continue
        synt=p==fixture
        if not jet and not synt:
            tasks=p/'meta/tasks.parquet'
            # Limit read to the task index (not images); identifies copies of the test fixture.
            if tasks.exists() and b'Synthetic pipeline check only.' in tasks.read_bytes(): synt=True
        datasets.append({'path':str(p),'name':('双摄合成演练' if synt else p.name if jet else p.parent.name),'synthetic':synt,'detail':f"{n} 条演示 · {frames} 帧 · {m['fps']} FPS · {m['codebase_version']}",'episodes':n,'frames':frames})
    except (OSError,KeyError,ValueError,TypeError):continue
o={'datasets':datasets,'free':round(shutil.disk_usage(h).free/2**30)}
if jet:
    nodes=[]
    for p in sorted(Path('/sys/class/video4linux').glob('video*')):
        if (p/'index').read_text().strip()!='0':continue
        a=next((a for a in p.resolve().parents if (a/'idVendor').exists()),None)
        if a:nodes.append(dict(node='/dev/'+p.name,port=a.name,product=((a/'product').read_text().strip() if (a/'product').exists() else p.name)))
    o['devices']=nodes;o['arms']=len(list(Path('/dev/serial/by-id').glob('usb-1a86_USB_Single_Serial_*-if00')))
else:
    r=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi' if Path('/usr/lib/wsl/lib/nvidia-smi').exists() else 'nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=6);o['gpu']=r.stdout.strip() or 'GPU 状态未确认'
print(json.dumps(o,ensure_ascii=False))
