"""Run one Jetson trial and its private RTX worker through the Mac's two existing SSH sessions."""
import argparse
import os
from pathlib import Path
import shlex
import socket
import subprocess
import tempfile
import uuid

SSH = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=6','-o','ServerAliveInterval=3','-o','ServerAliveCountMax=2']
EDGE='/home/nvidia/cie6007-edge/act-shadow'
GPU_COMMAND='cd ~/robotlab && exec env HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -u outputs/smolvla-cube46-v03-20261123/diagnostic-4080/worker.py'

def main():
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    for name in ('transport','check-start','measure-loop','onsite'): mode.add_argument('--'+name,action='store_true')
    p.add_argument('--seconds',type=int,default=15,choices=range(1,31))
    a=p.parse_args()
    trial_mode=next(flag for flag in ('transport','check-start','measure-loop','onsite') if getattr(a,flag.replace('-','_')))
    remote='/home/nvidia/cie6007-edge/lab/sessions/smol4080-'+uuid.uuid4().hex[:16]+'.sock'
    with tempfile.TemporaryDirectory(prefix='smol4080-',dir='/tmp') as tmp, socket.socket(socket.AF_UNIX) as listener:
        local=str(Path(tmp)/'infer')
        listener.bind(local); listener.listen(1); listener.settimeout(.5)
        flags=['--'+trial_mode]
        if a.onsite: flags+=['--console']
        if not a.transport: flags+=['--action-mode','chunk','--seconds',str(a.seconds)]
        argv=['env','-u','PYTHONPATH','PYTHONNOUSERSITE=1','SMOL46_SOCKET='+remote,
              '/home/nvidia/miniconda3/envs/lerobot/bin/python','-u','smol46_5k.py','--gpu4080',*flags]
        cmd='cd '+EDGE+" && trap 'rm -f "+remote+"' EXIT && "+shlex.join(argv)
        # The two SSH channels carry robot operator commands and inference data independently.
        with subprocess.Popen(SSH+['-o','ControlPath=none','-o','ExitOnForwardFailure=yes','-R',
                                   remote+':'+local,'jetson-lan',cmd]) as jetson:
            gpu=None
            try:
                while jetson.poll() is None:
                    try: stream,_=listener.accept()
                    except socket.timeout: continue
                    with stream:
                        gpu=subprocess.Popen(SSH+['gpu4080',GPU_COMMAND],stdin=stream,stdout=stream)
                    break
                return jetson.wait()
            finally:
                if gpu is not None:
                    try: gpu.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        gpu.terminate();gpu.wait(timeout=6)
if __name__=='__main__':
    raise SystemExit(main())
