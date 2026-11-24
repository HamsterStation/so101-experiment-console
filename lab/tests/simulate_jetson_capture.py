"""Target-environment integration exercise. Dummy arms+cameras ONLY; real writer/recorder."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import json
import tempfile
import time
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jetson_recorder as jr
import lerobot.teleoperators.so101_leader.so101_leader as lm
import lerobot.robots.so101_follower.so101_follower as fm

names='shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper'.split()
class Bus:
    is_calibrated=True
    def __init__(self):
        self.motors={n:NS(id=i+1) for i,n in enumerate(names)}
        self.is_connected=False
    def connect(self):self.is_connected=True
    def disconnect(self,disable_torque):self.is_connected=False
    def sync_read(self,key,normalize=False):
        return {n:(0 if key=='Torque_Enable' else 2048) for n in names}
    def sync_write(self,*a,**kw): pass
    def read(self,*a,**kw):return 0
    def _normalize(self,d):return dict.fromkeys(d,0.0)
class Camera:
    def __init__(self):
        self.is_connected=False;self.videocapture=NS(set=lambda *a:True)
        self.frame=np.broadcast_to((np.arange(640)%256).astype('uint8')[None,:,None],(480,640,3)).copy()
    def connect(self,warmup):self.is_connected=True
    def disconnect(self):self.is_connected=False
    def async_read(self,**kw):return self.frame
class FakeLeader(lm.SO101Leader):
    def __init__(self,cfg):
        self.id=cfg.id;self.bus=Bus();self.calibration={n:NS(range_min=800,range_max=3200) for n in names}
    def configure(self):pass
    def get_action(self):return {n+'.pos':0.0 for n in names}
class FakeFollower(fm.SO101Follower):
    def __init__(self,cfg):
        self.id=cfg.id;self.bus=Bus();self.config=cfg;self.calibration={n:NS(range_min=800,range_max=3200) for n in names}
        self.cameras={n:Camera() for n in cfg.cameras}
    def configure(self):pass
    def send_action(self,action):return action
    def get_observation(self):return {**{n+'.pos':0.0 for n in names},**{k:v.frame for k,v in self.cameras.items()}}
class Operator:
    def __init__(self,home):self.home=home
    def stopped(self):return False
    def checkpoint(self):pass

def main():
    inventory=[dict(node=f'/dev/video{i}',port=f'sim-{i}') for i in (0,2)]
    with tempfile.TemporaryDirectory(prefix='sim-lab-') as folder:
        tmp=Path(folder); tmpfile=tmp/'fake-port';tmpfile.touch()
        with patch.object(jr,'BASE',tmp), patch.object(jr,'devices',return_value=inventory), patch.dict(jr.PORTS,leader=str(tmpfile),follower=str(tmpfile)),patch.object(jr,'Operator',Operator), patch.object(lm,'SO101Leader',FakeLeader), patch.object(fm,'SO101Follower',FakeFollower):
            # fuser checks this file, no serial port is ever opened; tests use camera IDs intentionally absent in real sessions.
            jr.capture('sim_only',dict(front='/dev/video0',wrist='/dev/video2',onsite=True,episodes=2,episode_s=.23,reset_s=.1,task='Synthetic integration test'))
        info=json.loads((tmp/'recordings/sim_only/meta/info.json').read_text())
        assert info['total_episodes']==2 and info['total_frames']>=4 and info['total_videos']==4, info
        print('RECORD_INTEGRATION_OK',info['total_frames'],'frames, 4 video segments, no hardware')

if __name__=='__main__':main()
