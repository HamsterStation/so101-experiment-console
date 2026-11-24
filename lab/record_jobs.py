"""Mac adapter: only these predefined operations reach the Jetson over SSH."""
import base64
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = '/home/nvidia/cie6007-edge/lab'
PYTHON = '/home/nvidia/miniconda3/envs/lerobot/bin/python'
SSH = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=6','-o','ServerAliveInterval=3','-o','ServerAliveCountMax=2','jetson']
DEFAULT_TASK = 'Pick up the cube and place it in the target tray.'


def _id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',value):
        raise ValueError('无效的录制编号')
    return value


def record_path(job_id):
    return f'{BASE}/recordings/{_id(job_id)}'


def _config(c):
    if c.get('onsite') is not True:
        raise ValueError('需要现场操作人员确认')
    d = {'onsite': True}
    for name in ('front','wrist'):
        value = c.get(name)
        if not isinstance(value,str) or not re.fullmatch(r'/dev/video[0-9]{1,3}',value):
            raise ValueError('请选择两只已识别的摄像头')
        d[name] = value
    if d['front'] == d['wrist']:
        raise ValueError('外部和腕部不能选择同一只摄像头')
    for k,default,lo,hi in [('episodes',1,1,100),('episode_s',20,5,180),('reset_s',10,3,120)]:
        value = c.get(k, default)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f'{k} 应为 {lo} 至 {hi} 的整数')
        d[k] = value
    task = c.get('task',DEFAULT_TASK)
    if not isinstance(task,str) or not 1 <= len(task.strip()) <= 200 or not task.isprintable():
        raise ValueError('请填写 1–200 字符的单行任务描述')
    d['task'] = task.strip()
    return d


def record_command(config,job_id):
    return [sys.executable,'-u',str(HERE/'record_jobs.py'),'run',_id(job_id),json.dumps(_config(config),ensure_ascii=False)]


def _remote(mode, job_id='', cfg=None):
    # Read our installed local worker; the request cannot supply a command or path.
    source = (HERE/'jetson_recorder.py').read_text()
    args = [PYTHON if mode=='record' else '/usr/bin/python3','-u','-c',source,mode,job_id]
    if cfg is not None:
        args.append(base64.urlsafe_b64encode(json.dumps(cfg).encode()).decode())
    return [*SSH, shlex.join(['env','-u','PYTHONPATH','PYTHONNOUSERSITE=1','HF_HUB_OFFLINE=1','PATH=/home/nvidia/miniconda3/envs/lerobot/bin:/usr/local/bin:/usr/bin:/bin',*args])]


def camera_probe_command():
    return _remote('inventory')


def record_stop_command(job_id):
    return _remote('stop',_id(job_id))


def record_finish_command(job_id):
    return _remote('finish',_id(job_id))


def stage_command(job_id):
    return [str(HERE.parent/'stage_recording.sh'),record_path(job_id)]


def _bridge(job_id,cfg):
    owner = os.getppid()
    child = subprocess.Popen(_remote('record',_id(job_id),_config(cfg)),stdin=subprocess.PIPE)
    alive = True
    def stop(*args):
        nonlocal alive
        alive = False
    for sig in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,stop)
    try:
        while alive and child.poll() is None and os.getppid()==owner:
            child.stdin.write(b'live\n')
            child.stdin.flush()
            time.sleep(.7)
    except (BrokenPipeError, OSError):
        pass
    finally:
        try: child.stdin.close()
        except OSError: pass
    return child.wait()


if __name__ == '__main__':
    if len(sys.argv)!=4 or sys.argv[1]!='run':
        raise SystemExit('Start from the local Lab')
    sys.exit(_bridge(sys.argv[2],json.loads(sys.argv[3])))
