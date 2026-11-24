"""Loopback control room for the two fixed, SSH-authenticated lab machines."""
import concurrent.futures, contextlib, copy, datetime, hashlib, json, mimetypes, os, re, secrets, shlex, signal, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
import record_jobs as rec
import trial_jobs as trials
from public_datasets import inspect_repo

HERE=Path(__file__).resolve().parent
ART = Path(os.environ.get('SO101_CONSOLE_DATA', HERE.parent / 'local-data')).expanduser()
DB=ART/'lab-state.json'
REFERENCE=ART/'act-cube49-real-004000'/'真机试跑-双摄.mp4'
PORT = int(os.environ.get('SO101_CONSOLE_PORT', '8797'))
TOKEN=secrets.token_urlsafe(32)
HOSTS={'127.0.0.1:'+str(PORT),'localhost:'+str(PORT)}
L=threading.RLock()
hosts=('jetson','gpu4080')
dev={k:{'online':False,'detail':'正在连接…'} for k in hosts}
cams={'ready':False,'status':'unknown','detail':'等待检测','devices':[]}
catalog={}; jobs={}; live={}; leases={}; camera_at=0.; labelmap={}; previews=threading.BoundedSemaphore(2)
recipes=[{'id':'act','name':'ACT · 基线','detail':'使用双摄、六关节状态和动作进行模仿学习。'},{'id':'smolvla','name':'SmolVLA · 基线','detail':'用本机缓存的预训练权重，对选定数据集进行微调。'}]
def now():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def persist():
    with L:
        tmp=DB.with_suffix('.new');tmp.write_text(json.dumps({'jobs':jobs,'labels':labelmap},ensure_ascii=False));tmp.replace(DB)
def init():
    ART.mkdir(mode=0o700, parents=True, exist_ok=True)
    if DB.exists():
        d=json.loads(DB.read_text());jobs.update(d['jobs']);labelmap.update(d.get('labels',{}))
        for j in jobs.values():
            if j['status'] in ('running','stopping','interrupted') and j['kind'] in ('train','reproduce'):
                j.update(status='running',detail='正在重连 4080 上原有运行')
                threading.Thread(target=runner,args=(j,[sys.executable,'-u',str(HERE/'train_jobs.py'),'watch','--run-id',j['id']]),daemon=True).start()
            elif j['status'] in ('running','stopping'):
                detail='平台已重启，请到现场托稳并核对从臂供电状态' if j['kind']=='trial' else '平台已重启，请核对远端状态'
                j.update(status='interrupted',detail=detail,ended=now())
            elif j['kind']=='record' and j['status']=='failed' and j['detail'].startswith('已请求结束；'):
                # Older versions mixed control acknowledgments with recorder output.
                for line in reversed(j['log']):
                    if re.match(r'^\w*(?:Error|Exception): ',line):
                        j['detail']=line[:300]
                        break
        persist()
def sh(args,timeout=35):
    p=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,errors='replace',timeout=timeout)
    if p.returncode:raise ValueError((p.stderr or p.stdout or '远程命令未完成').strip()[-750:])
    return p.stdout
def ssh(host,args):
    assert host in hosts
    return ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=6','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',host,shlex.join(args)]
def survey(host,path=None):
    return json.loads(sh(ssh(host,['/usr/bin/python3','-c',(HERE/'inventory.py').read_text(),host,*([path] if path else [])]),timeout=55))
def dsid(host,path):return hashlib.sha256((host+':'+path).encode()).hexdigest()[:20]
def refresh_host(host):
    global camera_at
    try:d=survey(host)
    except Exception as e:
        with L:
            dev[host].update(online=False,detail='未连通：'+str(e)[:150],checked=now())
            if host=='jetson':cams.update(ready=False,status='unknown',detail='Jetson 未连通',devices=[]);camera_at=0
        return
    with L:
        items={}
        for x in d['datasets']:
            i=dsid(host,x['path']);x.update(id=i,location=host)
            x.update(labelmap.get(i,{}));items[i]=x
        for i in [i for i,k in catalog.items() if k['location']==host and i not in items]: catalog.pop(i)
        catalog.update(items)
        detail=(f"两臂串口 {d['arms']}/2" if host=='jetson' else d['gpu'])
        dev[host].update(online=True,detail=f"{detail} · 剩余 {d['free']} GiB",checked=now())
        if host=='jetson':
            nodes=d['devices']
            fingerprint=lambda ls: sorted((x.get('node'),x.get('port')) for x in ls)
            if not nodes or fingerprint(nodes)!=fingerprint(cams['devices']) or time.time()-camera_at>300:
                cams.update(ready=False,status='missing' if len(nodes)<2 else 'unchecked',devices=nodes,detail=f'识别到 {len(nodes)} 路 USB 视频设备'+('；等待双摄接入' if len(nodes)<2 else '；请点击双摄检查'));camera_at=0.
def refresh():
    with concurrent.futures.ThreadPoolExecutor(2) as pool: list(pool.map(refresh_host,hosts))
def evidence():
    out=[]
    for key,name in [('act','原版 ACT'),('act2','两层 ACT'),('smol6','六层 SmolVLA')]:
        p=ART/f'{key}-edge.json'
        if p.is_file():
            d=json.loads(p.read_text());out.append({'name':name,'detail':f"Jetson GPU · {d['median_core_ms']} ms · 8 次预热后核心推理中位数，合成输入"})
    return out
def snapshot():
    with L:
        jj=list(reversed(jobs.values()))
        runs=[dict(id=j['id'],name=j['name'],detail=j['detail'],status=j['status'],recipe_id=j['recipe_id'],steps=j['steps'],seed=j['seed'],reproducible=j['status']=='succeeded') for j in jj if j['kind'] in ('train','reproduce')]
        for r in runs:
            source=jobs[r['id']]
            line=next((s[11:] for s in reversed(source['log']) if s.startswith('LAB_RESULT ')),None)
            if line:
                with contextlib.suppress(ValueError):
                    res=json.loads(line)
                    if 'same_checkpoint_bytes' in res:r['detail']=('复现权重逐字节一致' if res['same_checkpoint_bytes'] else '复现已完成，权重字节有差异')+' · '+r['detail']
        return copy.deepcopy(dict(devices=dev,cameras=cams,datasets=list(catalog.values()),jobs=jj,runs=runs,recipes=recipes,evidence=evidence(),trial_reference=REFERENCE.is_file()))
def append(j,s):
    s=s.strip()[-1600:]
    if not s:return
    with L:
        j['log'].append(s);j['log']=j['log'][-100:]
        m=re.fullmatch(r'RECORD_PHASE (\w+) (\d+)',s)
        if m:j['phase']=m[1];j['episode']=int(m[2]);j['detail']=s
def number(d,key,default,lo,hi):
    v=d.get(key,default)
    if type(v)!=int or not lo<=v<=hi:raise ValueError(f'{key} 超出范围')
    return v
def dataset(key,location=None):
    d=catalog.get(key)
    if not d or (location and d['location']!=location):raise ValueError('请选择已登记且位置正确的数据集')
    return d
def active(kind):
    return any(j['status'] in ('running','stopping') and j['kind'] in kind for j in jobs.values())
def runner(j,argv=None,cb=None):
    try:
        k=j['kind']
        if k=='refresh':
            refresh()
            for h in hosts:append(j,h+'：'+dev[h]['detail'])
        elif k=='camera_check':
            global camera_at
            argv=ssh('jetson',['/usr/bin/python3','/home/nvidia/cie6007-edge/checks/check_cameras.py'])
            p=subprocess.run(argv,capture_output=True,text=True,timeout=40)
            for s in (p.stdout+p.stderr).splitlines():append(j,s)
            with L:
                cams.update(ready=False,status='missing',detail=(p.stdout or p.stderr or '无法检查').strip().splitlines()[-1][:180])
                if p.stdout.lstrip().startswith('['):
                    nodes,_=json.JSONDecoder().raw_decode(p.stdout.lstrip());good=sum(x['ok'] for x in nodes)
                    cams.update(devices=nodes,ready=good>=2,status='ready' if good>=2 else 'failed',detail=f'{good}/{len(nodes)} 路通过连续取帧检查')
                else:cams['devices']=[]
                camera_at=time.time()
                if not cams['ready']:raise ValueError(cams['detail'])
        else:
            p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
            with L:live[j['id']]=p
            last_output=''
            for line in p.stdout:
                if line.strip():last_output=line.strip()[-1600:]
                append(j,line)
            if p.wait():raise ValueError(last_output or '命令未完成')
            if cb:cb(j)
        with L:j.update(status='stopped' if j.get('stop_requested') else 'succeeded',detail=j.get('result') or ('已停止' if j.get('stop_requested') else '已完成'))
    except Exception as e:
        with L:
            if not j['log'] or j['log'][-1]!=str(e):append(j,str(e))
            j.update(status='failed',detail=str(e)[:300])
    finally:
        with L:j.update(ended=now());leases.pop(j['id'],None);live.pop(j['id'],None);persist()
        if j['kind'] in ('record','stage'):refresh_host('jetson' if j['kind']=='record' else 'gpu4080')
def complete_stage(j):
    out=next((s.split(': ',1)[1] for s in reversed(j['log']) if s.startswith('4080 dataset: ')),None)
    if not out:raise ValueError('未返回校验后的 4080 数据集路径，请核对日志')
    with L:labelmap[dsid('gpu4080',out)]={'name':j['dataset_name'],'synthetic':j['synthetic']};j['result']='已校验并入库：'+out
def complete_record(j):
    with L:labelmap[dsid('jetson',rec.record_path(j['id']))]={'name':j['name'],'synthetic':False};j['result']='已保存：'+rec.record_path(j['id'])
trial_owners = {}


def trial_input(j, token):
    p = live.get(j['id'])
    if not p or p.poll() is not None:
        raise ValueError('Jetson 控制连接尚未建立或已断开')
    try:
        p.stdin.write(token + '\n')
        p.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        raise ValueError('Jetson 控制连接断开，请托稳后核对从臂供电') from exc


def trial_snapshot(owner, heartbeat_id):
    with L:
        current = next((j for j in reversed(jobs.values()) if j['kind']=='trial'), None)
        if not current:
            return {}
        ident = current['id']
        owned = trial_owners.get(ident) == owner
        if owned and heartbeat_id == ident and ident in live and current.get('phase') in ('checking','ready','arming','running'):
            trial_input(current,'PING')
        keys = ('id','name','zone','model','phase','detail','status','fault','replay','commands','created','armed_possible','task_complete')
        return {**{k:current[k] for k in keys if k in current}, 'owned':owned}


def run_trial(j):
    trace = camera = error = ''
    try:
        with subprocess.Popen(trials.command(j["model"]), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1) as p:
            with L:
                live[j['id']] = p
                if j.get('cancelled'):
                    trial_input(j,'STOP')
            for line in p.stdout:
                line = line.strip()
                if not line:
                    continue
                with L:
                    append(j,line)
                    if line.startswith('TRIAL_PHASE '):
                        phase = line.split()[-1]
                        if phase in trials.PHASES and not (j.get('cancelled') and phase in ('ready','arming','running')):
                            j.update(phase=phase,detail=trials.PHASES[phase])
                    elif line == 'TRIAL_SUCCESS cube_in_box':
                        j['task_complete'] = True
                    elif line.startswith('试跑记录 '):
                        trace = line.split()[-1]
                    elif line.startswith('现场双摄保存到 '):
                        camera = line.split()[-1]
                    elif line.startswith('TRIAL_ERROR '):
                        j['fault'] = line.removeprefix('TRIAL_ERROR ')
                    elif re.match(r'^\w*(?:Error|Exception):',line):
                        error = line
            code = p.wait()
        with L:
            live.pop(j['id'],None)
            if code:
                j['fault'] = error or j.get('fault') or 'Jetson 连接中断或试跑程序退出'
            if j['phase']=='released':
                j.update(detail='从臂已卸力，正在整理双摄回放')
        if trace:
            try:
                dest = ART/'trials'/j['id']
                replay = trials.save_evidence(dest,trace,camera,j["model"])
                telemetry = json.loads((dest/'trace.json').read_text())
                with L:
                    j.update(replay=replay,commands=telemetry.get('commands_sent',0))
                    if telemetry.get('completion',{}).get('reason') == 'cube_in_box':
                        j['task_complete'] = True
                    if telemetry.get('phase') in ('failed','finished') and not telemetry.get('torque_enable_attempted'):
                        j['armed_possible']=False
            except Exception as exc:
                with L:
                    j['evidence_error'] = '试跑记录或回放未取回'
                    append(j,f"回放：{exc}")
        with L:
            if j.get('armed_possible') and j['phase'] != 'released':
                j['fault'] = (j.get('fault') or '试跑已退出') + '。卸力未确认，请托稳后核对从臂电源。'
            detail = j.get('fault') or ('从臂已卸力' if j['phase']=='released' else '已取消接管')
            if j.get('evidence_error'): detail += ' · ' + j['evidence_error']
            j.update(status='failed' if j.get('fault') else 'stopped' if j.get('cancelled') or j['phase']!='released' else 'succeeded',detail=detail)
    except Exception as exc:
        with L:
            j.update(status='failed',detail=str(exc)+('。卸力未确认，请在现场托稳后核对电源。' if j.get('armed_possible') else ''))
    finally:
        with L:
            live.pop(j['id'],None);trial_owners.pop(j['id'],None)
            j.update(ended=now());persist()


def trial_action(d,owner):
    action = d['action']
    with L:
        if action == 'trial_check':
            if not re.fullmatch(r'[A-Za-z0-9-]{12,80}',owner):
                raise ValueError('请刷新操作页面')
            if active(('trial','camera_check','record')):
                raise ValueError('Jetson 正在录制、检查双摄或试跑')
            zone = d.get('zone')
            model = d.get('model', 'cube49')
            if not isinstance(model, str) or model not in trials.MODELS:
                raise ValueError('请选择已部署的模型')
            if zone not in trials.MODELS[model][3]:
                raise ValueError('请选择方块起点')
            ident = 'trial-' + uuid.uuid4().hex[:12]
            j = dict(id=ident,kind='trial',zone=zone,model=model,name=trials.MODELS[model][0]+' · '+zone,
                     phase='checking',status='running',detail=trials.PHASES['checking'],created=now(),log=[])
            jobs[ident]=j;trial_owners[ident]=owner;persist()
            threading.Thread(target=run_trial,args=(j,),daemon=True).start()
            return {'ok':True,'job_id':ident}
        j = jobs.get(d.get('job_id'))
        if not j or j['kind']!='trial' or j['status']!='running':
            raise ValueError('这轮试跑已结束')
        phase = j['phase']
        if action=='trial_start' and phase=='ready' and trial_owners.get(j['id'])==owner and d.get('onsite') is True:
            trial_input(j,'PING')
            trial_input(j,'START')
            j.update(phase='arming',armed_possible=True)
        elif action=='trial_stop' and phase in ('checking','ready','arming','running','stopping'):
            if j['id'] in live: trial_input(j,'STOP')
            j.update(cancelled=True,phase='stopping')
        elif action=='trial_release' and phase=='holding' and d.get('supported') is True:
            trial_input(j,'RELEASE')
            j['phase']='releasing'
        else:
            raise ValueError('当前阶段不接受该操作，请核对页面状态')
        j['detail']=trials.PHASES[j['phase']]
        return {'ok':True}


def submit(d,owner):
    global camera_at
    kind=d.get('action')
    if kind in ('trial_check','trial_start','trial_stop','trial_release'):
        return trial_action(d,owner)
    if kind in ('stop','finish_episode'):
        with L:
            j=jobs.get(d.get('job_id'))
            if not j or j['kind']!='record' or j['status'] not in ('running','stopping'):raise ValueError('没有对应的进行中录制')
            if d.get('onsite') is not True:raise ValueError('请由现场操作人确认')
            cmd=rec.record_stop_command if kind=='stop' else rec.record_finish_command
        msg=sh(cmd(j['id']),15)
        with L:
            if j['status'] not in ('running','stopping'):
                return {'ok':True,'message':j['detail']}
            append(j,msg)
            if kind=='stop' and j['status']=='running':j.update(status='stopping',stop_requested=True)
            persist()
        return {'ok':True,'message':msg.strip()}
    if kind not in ('refresh','camera_check','record','stage','train','reproduce'):raise ValueError('不支持该操作')
    with L:
        group={'record':('record','camera_check','trial'),'camera_check':('record','camera_check','trial'),'train':('train','reproduce'),'reproduce':('train','reproduce')}.get(kind,(kind,))
        if active(group):raise ValueError('同一设备已有任务运行，请等待完成')
        ident=kind+'-'+uuid.uuid4().hex[:12]; argv=None;cb=None
        names={'refresh':'检查设备连接','camera_check':'双摄连续取帧','stage':'转存数据集','train':'模型训练','reproduce':'复现实验','record':'遥操录制'}
        j=dict(id=ident,kind=kind,name=names[kind],status='running',detail='正在运行',log=[],created=now())
        if kind=='record':
            if not cams['ready'] or time.time()-camera_at>300:raise ValueError('请先通过最近 5 分钟内的双摄检查')
            safe_nodes={x['node'] for x in cams['devices'] if x.get('ok')};
            if not {d.get('front'),d.get('wrist')}<=safe_nodes:raise ValueError('选择的两路摄像头未通过检查')
            if not re.fullmatch(r'[A-Za-z0-9-]{12,80}',owner):raise ValueError('没有有效的操作页')
            argv=rec.record_command(d,ident);name=d.get('name') or names[kind]
            if not isinstance(name,str) or not name.isprintable() or len(name)>90:raise ValueError('名称须为最多 90 字的单行文字')
            j.update(name=name);leases[ident]=(owner,time.monotonic());cb=complete_record
        if kind=='stage':
            source=dataset(d.get('dataset_id'),'jetson')
            argv=[str(HERE.parent/'stage_recording.sh'),source['path']];cb=complete_stage
            j.update(dataset_name=source['name'],synthetic=source['synthetic'],name='转存 · '+source['name'])
        if kind in ('train','reproduce'):
            script=HERE/'train_jobs.py'
            if not script.exists():raise ValueError('训练服务尚未就绪')
            if kind=='train':
                source=dataset(d.get('dataset_id'),'gpu4080');model=d.get('recipe_id')
                if model not in ('act','smolvla'):raise ValueError('未知训练配方')
                steps=number(d,'steps',1000,1,100000);seed=number(d,'seed',42,0,2147483647)
                mode=d.get('split','all');n=source.get('episodes',0)
                if mode not in ('all','manual','episode80') or (mode!='all' and n<2):raise ValueError('该数据集不支持所选划分')
                args=['train','--dataset',source['path'],'--model',model,'--seed',str(seed),'--steps',str(steps),'--split',mode]
                if mode=='manual':
                    ids=d.get('validation_episodes')
                    if not isinstance(ids,list) or not 0<len(ids)<n or any(type(v)!=int or not 0<=v<n for v in ids) or len(set(ids))!=len(ids):raise ValueError('验证演示编号无效、重复或覆盖全部数据')
                    args+=['--validation-episodes',','.join(str(v) for v in sorted(ids))]
                j['split']=mode
            else:
                orig=jobs.get(d.get('run_id'))
                if not orig or orig['kind'] not in ('train','reproduce') or orig['status']!='succeeded':raise ValueError('请选择已完成的训练实验')
                model=orig['recipe_id'];steps=orig['steps'];seed=orig['seed'];source={'name':orig['dataset_name'],'synthetic':orig['synthetic']};args=['reproduce','--source-id',orig['id']]
            j.update(recipe_id=model,steps=steps,seed=seed,synthetic=source['synthetic'],dataset_name=source['name'],name=('复现' if kind=='reproduce' else '训练')+' · '+model.upper()+(' · 合成演练' if source['synthetic'] else ' · '+source['name']))
            argv=[sys.executable,'-u',str(script),*args,'--run-id',ident]
            j['result']=f"4080：~/robotlab/cie6007-lab/runs/{ident}/"
        jobs[ident]=j;persist();threading.Thread(target=runner,args=(j,argv,cb),daemon=True).start()
    return {'ok':True,'job_id':ident}
def watchdog():
    while True:
        time.sleep(2)
        with L:
            expired=[i for i,(_,t) in leases.items() if time.monotonic()-t>25]
            for i in expired:
                if i in live:live[i].terminate()
                else:continue
                leases.pop(i,None);jobs[i].update(status='stopping',stop_requested=True);append(jobs[i],'操作页连接超时，已撤销录制租约')
class Handler(BaseHTTPRequestHandler):
    def handle_one_request(self):
        try:return super().handle_one_request()
        except (BrokenPipeError,ConnectionResetError):return
    def log_message(self,*args): pass
    def send(self,code,data,typ='application/json; charset=utf-8'):
        if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(code);self.send_header('Content-Type',typ);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'");self.end_headers();self.wfile.write(data)
    def local(self):
        return self.headers.get('Host') in HOSTS and self.headers.get('Sec-Fetch-Site')!='cross-site'
    def do_GET(self):
        if not self.local():return self.send(403,{'error':'仅限本机页面访问'})
        u=urlsplit(self.path)
        if u.path=='/api/public-dataset':
            try:return self.send(200,inspect_repo(parse_qs(u.query).get('repo',[''])[0]))
            except ValueError as e:return self.send(400,{'error':str(e)})
        if u.path=='/api/session':return self.send(200,{'token':TOKEN})
        if u.path=='/api/state':return self.send(200,snapshot())
        if u.path=='/api/manifest':
            ident=parse_qs(u.query).get('id',[''])[0]
            with L:j=jobs.get(ident)
            if not j or j['kind'] not in ('train','reproduce'):return self.send(404,{'error':'没有该训练记录'})
            try:return self.send(200,json.loads(sh(ssh('gpu4080',['/usr/bin/python3','-c',"from pathlib import Path; import sys; print((Path.home()/'robotlab/cie6007-lab/runs'/sys.argv[1]/'manifest.json').read_text())",ident]))))
            except Exception as e:return self.send(400,{'error':str(e)})
        if u.path=='/api/trial-replay':
            ident=parse_qs(u.query).get('id',[''])[0]
            if ident=='reference':
                f=REFERENCE
            else:
                with L:
                    if not re.fullmatch(r'trial-[a-f0-9]{12}',ident) or not jobs.get(ident,{}).get('replay'):
                        return self.send(404,{'error':'这轮没有可播放的回放'})
                f=ART/'trials'/ident/'replay.mp4'
            if not f.is_file():return self.send(404,{'error':'回放文件不存在'})
            size=f.stat().st_size
            start,end=0,size-1
            header=self.headers.get('Range')
            if header:
                m=re.fullmatch(r'bytes=(\d+)-(\d*)',header)
                if not m:return self.send(416,{'error':'无效的视频区间'})
                start=int(m[1]);end=min(int(m[2]),end) if m[2] else end
                if start>end:return self.send(416,{'error':'无效的视频区间'})
            self.send_response(206 if header else 200)
            self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes')
            if header:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.send_header('Content-Length',str(end-start+1));self.end_headers()
            with f.open('rb') as video:
                video.seek(start);self.wfile.write(video.read(end-start+1))
            return
        if u.path=='/api/preview':
            try:
                with L:d=dataset(parse_qs(u.query).get('id',[''])[0])
                if not previews.acquire(blocking=False):return self.send(429,{'error':'正在读取其他预览，请稍候'})
                try:out=survey(d['location'],d['path'])
                finally:previews.release()
                return self.send(200,out)
            except Exception as e:return self.send(400,{'error':str(e)})
        static={'/':'index.html','/app.css':'app.css','/app.js':'app.js','/trial.js':'trial.js','/space.js':'space.js','/space.css':'space.css','/data-workbench.js':'data-workbench.js','/data-workbench.css':'data-workbench.css'}
        if u.path not in static:return self.send(404,{'error':'页面不存在'})
        f=HERE/'static'/static[u.path];return self.send(200,f.read_bytes(),mimetypes.guess_type(f)[0]+'; charset=utf-8')
    def do_POST(self):
        origin=self.headers.get('Origin')
        if not self.local() or (origin and origin not in {'http://'+h for h in HOSTS}) or not secrets.compare_digest(self.headers.get('X-Lab-Token',''),TOKEN):return self.send(403,{'error':'会话已失效，请刷新页面'})
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n<2 or n>8192 or self.headers.get_content_type()!='application/json':raise ValueError('无效请求')
            d=json.loads(self.rfile.read(n));owner=self.headers.get('X-Lab-Operator','')
            if not isinstance(d,dict):raise ValueError('无效请求')
            if self.path=='/api/trial-status':
                return self.send(200,trial_snapshot(owner,d.get('heartbeat')))
            if self.path=='/api/heartbeat':
                with L:
                    for i,(o,t) in list(leases.items()):
                        if secrets.compare_digest(owner,o):leases[i]=(o,time.monotonic())
                return self.send(200,{'ok':True})
            if self.path!='/api/action':return self.send(404,{'error':'页面不存在'})
            return self.send(200,submit(d,owner))
        except (ValueError,TypeError,subprocess.SubprocessError) as e:return self.send(400,{'ok':False,'error':str(e)[:600]})
if __name__=='__main__':
    os.umask(0o077);init()
    http=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    threading.Thread(target=refresh,daemon=True).start();threading.Thread(target=watchdog,daemon=True).start()
    print(f'本机实验台：http://127.0.0.1:{PORT}',flush=True);http.serve_forever()
