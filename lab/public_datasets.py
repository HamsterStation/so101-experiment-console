"""Read public Hub metadata only. Never download weights, samples, or repository code."""
import json,re,urllib.request, urllib.error
from urllib.parse import urlsplit
import time, threading
CACHE={};LOCK=threading.Lock()
class HubRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        p=urlsplit(newurl)
        if p.scheme!='https' or p.netloc!='huggingface.co':raise ValueError('元数据跳转到其他域名，已停止读取')
        return super().redirect_request(req,fp,code,msg,headers,newurl)
def read_json(path):
    req=urllib.request.Request('https://huggingface.co/'+path,headers={'User-Agent':'local-so101-console/1.0'})
    with urllib.request.build_opener(HubRedirect()).open(req,timeout=6) as r:data=r.read(1048577)
    if len(data)>1048576: raise ValueError('元数据超过大小限制')
    return json.loads(data)
def inspect_repo(repo):
    if not isinstance(repo,str) or len(repo)>160 or not re.fullmatch(r'[\w-]+/[\w][\w.-]*',repo,re.ASCII) or '..' in repo:raise ValueError('请输入公开数据集编号，例如 lerobot/svla_so101_pickplace')
    with LOCK:
        if repo in CACHE and time.monotonic()-CACHE[repo][0]<1800:return CACHE[repo][1]
    try:
        card=read_json('api/datasets/'+repo+'?expand=sha&expand=cardData')
        sha=card.get('sha','')
        if not re.fullmatch('[0-9a-f]{40}',sha):raise ValueError('未能确定数据集版本')
        meta=read_json('datasets/'+repo+'/raw/'+sha+'/meta/info.json')
        f=meta['features'];cams=[k for k,v in f.items() if v.get('dtype') in ('image','video') and k.startswith('observation.images.')]
        target=['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']
        state=f.get('observation.state',{});action=f.get('action',{})
        same=state.get('shape')==action.get('shape')==[6] and state.get('names')==action.get('names')==target
        out={'repo':repo,'revision':sha,'version':meta.get('codebase_version'),'episodes':meta.get('total_episodes'),'fps':meta.get('fps'),'robot':meta.get('robot_type'),'license':(card.get('cardData') or {}).get('license') or '未声明','cameras':cams,'joints_ok':same,'meta_url':'https://huggingface.co/datasets/'+repo+'/blob/'+sha+'/meta/info.json'}
    except urllib.error.HTTPError as e:raise ValueError('仓库或 meta/info.json 不公开、不存在或当前请求受限（HTTP '+str(e.code)+'）') from None
    except (OSError,KeyError,TypeError,json.JSONDecodeError) as e:raise ValueError('未取得有效的 LeRobot 元数据。可手动打开数据卡核对；本机数据不受影响。') from None
    with LOCK:
        if len(CACHE)>=64:CACHE.pop(next(iter(CACHE)))
        CACHE[repo]=(time.monotonic(),out)
    return out
