#!/usr/bin/python3
"""On-demand camera-only test; no serial access or reset. No images are saved."""
import argparse,concurrent.futures as cf,json,os,subprocess,sys,time
from pathlib import Path

def get(p):
    try:return p.read_text().strip()
    except OSError:return ''

def cameras():
    result=[]
    for dev in sorted(Path('/sys/class/video4linux').glob('video*')):
        if get(dev/'index')!='0':continue
        u=next((a for a in dev.resolve().parents if (a/'idVendor').is_file()),None)
        if not u:continue
        result.append(dict(node='/dev/'+dev.name,port=u.name,product=get(u/'product'),serial=get(u/'serial'),usb_id=get(u/'idVendor')+':'+get(u/'idProduct')))
    return result

def read_camera(path,seconds):
    import cv2
    v=cv2.VideoCapture(path,cv2.CAP_V4L2)
    try:
        if not v.isOpened():return {'ok':False,'issue':'未能打开设备，也可能正被其他程序占用'}
        for key,value in [(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')),(cv2.CAP_PROP_FRAME_WIDTH,1280),(cv2.CAP_PROP_FRAME_HEIGHT,720),(cv2.CAP_PROP_FPS,30)]:v.set(key,value)
        if not v.read()[0]:return {'ok':False,'issue':'没有收到首帧'}
        t0=previous=time.monotonic();n=failed=0;maxgap=0;sample=None;largest_std=0
        while time.monotonic()-t0<seconds:
            ok,img=v.read();now=time.monotonic()
            if not ok:failed+=1;time.sleep(.03);continue
            n+=1;maxgap=max(maxgap,now-previous);previous=now;sample=img
            largest_std=max(largest_std,float(img[::16,::16].std()))
        duration=time.monotonic()-t0
        return dict(ok=(n>0 and not failed and n/duration>=10 and maxgap<1 and largest_std>2),frames=n,read_failures=failed,fps=round(n/duration,1),max_frame_gap_ms=round(maxgap*1000),width=sample.shape[1] if n else 0,height=sample.shape[0] if n else 0,max_image_std=round(largest_std,1))
    finally:v.release()

def test(device):
    try:
        r=subprocess.run([sys.executable,__file__,'--node',device['node']],capture_output=True,text=True,timeout=20); data=json.loads(r.stdout)
    except (ValueError,subprocess.TimeoutExpired):data={'ok':False,'issue':'未能在20秒内完成取帧'}
    return {**device,**data}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node');a=p.parse_args()
    if a.node:print(json.dumps(read_camera(a.node,8),ensure_ascii=False));sys.exit(0)
    nodes=cameras()
    if not nodes:
        print('没有 USB 视频设备。尚未开始取帧。');j=subprocess.run(['journalctl','-k','-b','--since','5 min ago','--no-pager','-o','cat','-g','unable to enumerate USB device','-n','1'],capture_output=True,text=True,timeout=8); print(j.stdout.strip());sys.exit(2)
    with cf.ThreadPoolExecutor(max_workers=min(len(nodes),4)) as pool: results=list(pool.map(test,nodes))
    print(json.dumps(results,ensure_ascii=False,indent=2));passed=sum(c['ok'] for c in results);print(f'同时出有效画面的摄像头：{passed}/{len(nodes)}；双摄检查：'+('通过' if passed>=2 else '未通过'));sys.exit(0 if passed>=2 else 1)
