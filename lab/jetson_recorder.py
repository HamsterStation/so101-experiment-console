"""Executed over an authenticated SSH channel. No hardware access on import/inventory."""
import base64
import fcntl
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
import time

BASE = Path('/home/nvidia/cie6007-edge/lab')
PORTS = {k:'/dev/serial/by-id/usb-1a86_USB_Single_Serial_'+s+'-if00'
         for k,s in [('leader','YOUR_LEADER_SERIAL'),('follower','YOUR_FOLLOWER_SERIAL')]}
# The unloaded follower repeatedly settles three encoder counts past its taught
# elbow stop. Allow one extra count of measurement variation (4/4096 of a turn).
ENDPOINT_TOLERANCE_TICKS = 4


def read(path):
    try: return path.read_text().strip()
    except OSError: return ''


def devices():
    result = []
    for video in sorted(Path('/sys/class/video4linux').glob('video*')):
        if read(video/'index')!='0': continue
        usb = next((x for x in video.resolve().parents if (x/'idVendor').is_file()),None)
        if usb:
            result.append({'node':'/dev/'+video.name,'product':read(usb/'product'),
                           'usb_id':read(usb/'idVendor')+':'+read(usb/'idProduct'),
                           'serial':read(usb/'serial'),'port':usb.name})
    return result


def _atomic(path,obj):
    t=path.with_suffix('.tmp'); t.write_text(json.dumps(obj,ensure_ascii=False)); t.replace(path)


def startup_check(leader, follower, cold=True):
    positions=[]
    for arm in (leader,follower):
        bus = arm.bus
        if not arm.calibration or not bus.is_calibrated: raise RuntimeError(f'{arm.id} 校准不一致；请现场检查')
        if cold and any(bus.sync_read('Torque_Enable',normalize=False).values()): raise RuntimeError(f'{arm.id} 已上力，拒绝占用')
        if any(bus.read('Status',n,normalize=False) for n in bus.motors): raise RuntimeError(f'{arm.id} 有电机报警')
        raw=bus.sync_read('Present_Position',normalize=False)
        hold={}
        for joint,value in raw.items():
            cal=arm.calibration[joint]
            if not 0<=value<=4095:
                raise RuntimeError(f'{arm.id}/{joint} 位置 {value} 超出单圈编码范围 0–4095，取消跟随')
            target=min(cal.range_max,max(cal.range_min,value))
            if abs(value-target)>ENDPOINT_TOLERANCE_TICKS:
                raise RuntimeError(f'{arm.id}/{joint} 位置 {value} 超出校准范围 {cal.range_min}–{cal.range_max}，取消跟随')
            hold[joint]=target
        normalized=bus._normalize({bus.motors[n].id:v for n,v in raw.items()})
        positions.append((hold,normalized))
    for joint,motor in follower.bus.motors.items():
        leader_value=positions[0][1][motor.id]
        follower_value=positions[1][1][motor.id]
        if abs(leader_value-follower_value)>15:
            raise RuntimeError(f'两臂 {joint} 的校准读数相差过大（主臂 {leader_value:.1f}，从臂 {follower_value:.1f}），请托稳并对齐后重试')
    return positions[1][0]


class Operator:
    """No remote scheduler. Continuation requires a live Mac process/SSH channel."""
    def __init__(self,home):
        self.home=home; self.lost=False; self.last=time.monotonic()
        def revoke(*_): self.lost=True
        for sig in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT): signal.signal(sig,revoke)
        def listen():
            while not self.lost:
                if select.select([0],[],[],1)[0]:
                    if not os.read(0,8192):
                        self.lost=True; return
                    self.last=time.monotonic()
        threading.Thread(target=listen,daemon=True).start()

    def stopped(self):
        return self.lost or time.monotonic()-self.last>6 or (self.home/'stop').exists()

    def checkpoint(self):
        if self.stopped(): raise InterruptedError('已取消或与操作平台失联')


class Events(dict):
    # LeRobot's official loop polls exit_early before every new observation/action.
    def __init__(self,op,ep):
        super().__init__(exit_early=False,stop_recording=False,rerecord_episode=False)
        self.op=op; self.ep=ep

    def __getitem__(self,key):
        if key=='exit_early':
            return self.op.stopped() or (self.ep and (self.op.home/f'finish-{self.ep}').exists())
        return super().__getitem__(key)


def capture(job,cfg):
    base=BASE/'sessions'; base.mkdir(parents=True,exist_ok=True,mode=0o700)
    with open(base/'capture.lock','a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('已有录制占用设备')
        sess=base/job
        sess.mkdir(exist_ok=True,mode=0o700)
        # A stop arriving before startup is also honored.
        if (sess/'state.json').exists(): raise RuntimeError('录制编号已经使用，请新建任务')
        try: _session(job,cfg,sess)
        except BaseException:
            _atomic(sess/'state.json',{'phase':'failed','episode':0})
            raise


def _session(job,cfg,sess):
    op=Operator(sess)
    def phase(s,ep=0):
        _atomic(sess/'state.json',{'phase':s,'episode':ep})
        print('RECORD_PHASE',s,ep,flush=True)
    phase('preflight')
    inventory={d['node']:d for d in devices()}
    if any(c not in inventory for c in (cfg['front'],cfg['wrist'])) or inventory[cfg['front']]['port']==inventory[cfg['wrist']]['port']:
        raise RuntimeError('没有两只选定的独立 USB 摄像头；未启动机械臂')
    if not all(Path(p).exists() for p in PORTS.values()): raise RuntimeError('没有找到两只已配对的主从臂串口')
    if subprocess.run(['/usr/bin/fuser',*PORTS.values(),cfg['front'],cfg['wrist']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('摄像头或串口已被其他程序占用（请先关闭其他预览、遥操）')
    if shutil.disk_usage(BASE).free < 2*1024**3:
        raise RuntimeError('Jetson 可用空间低于 2 GiB')
    from lerobot.record import LeRobotDataset,hw_to_dataset_features,record_loop
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so101_follower.so101_follower import SO101Follower,SO101FollowerConfig
    from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader,SO101LeaderConfig
    # Use the very same camera instances for preflight and recording. The fallback
    # mode works without modifying the vendor's OpenCV adapter or opening a third stream.
    cams={'camera1':OpenCVCameraConfig(index_or_path=int(cfg['front'][10:]),fps=30,width=640,height=480),
          'camera2':OpenCVCameraConfig(index_or_path=int(cfg['wrist'][10:]),fps=30,width=640,height=480)}
    robot=SO101Follower(SO101FollowerConfig(id='so101_follower',port=PORTS['follower'],cameras=cams))
    leader=SO101Leader(SO101LeaderConfig(id='so101_leader',port=PORTS['leader']))
    root=BASE/'recordings'/job; root.parent.mkdir(parents=True,exist_ok=True)
    if root.exists(): raise RuntimeError('数据集目录已经存在，不会覆盖')
    ds=None; saved=0; robot_configuring=False
    try:
        op.checkpoint()
        import cv2
        for cam in robot.cameras.values():
            cam.fps=None  # apply requested FPS after negotiating MJPG below
            cam.connect(warmup=False)
            # Set the wire format before reading (the vendor config has no fourcc field).
            feed=cam.videocapture
            feed.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
            feed.set(cv2.CAP_PROP_FRAME_WIDTH,640); feed.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            feed.set(cv2.CAP_PROP_FPS,30)
        def read_pair():
            op.checkpoint()
            for key,cam in robot.cameras.items():
                image=cam.async_read(timeout_ms=1000)
                if image.shape!=(480,640,3) or image[::16,::16].std()<2:
                    raise RuntimeError(f'{key} 没有有效画面；未启动遥操')

        # The vendor adapter starts each USB stream on its first async_read.
        # Exclude first-frame latency and exposure settling from the rate test.
        read_pair()
        warm_until=time.monotonic()+1
        while time.monotonic()<warm_until:
            read_pair()
        t0=time.monotonic(); counts=0
        while time.monotonic()-t0 < 4:
            read_pair()
            counts+=1
        fps=counts/(time.monotonic()-t0)
        print(f'双摄预热后同采 {counts} 对画面，{fps:.1f} FPS',flush=True)
        if fps < 17: raise RuntimeError(f'双摄预热后仅 {fps:.1f} FPS，低于录制要求 17 FPS；未启动遥操')
        camera_map={k:{'role':v,**inventory[cfg[v]]} for k,v in [('camera1','front'),('camera2','wrist')]}
        print('CAMERA_PAIR_OK',json.dumps(camera_map),flush=True)
        op.checkpoint()
        for arm in (leader,robot): arm.bus.connect()
        startup_check(leader,robot)
        feat={**hw_to_dataset_features(robot.action_features,'action',True),
              **hw_to_dataset_features(robot.observation_features,'observation',True)}
        ds=LeRobotDataset.create(f'local/{job}',20,feat,root=root,robot_type=robot.name,use_videos=True,image_writer_threads=4)
        _atomic(root/'meta/lab-capture.json',{'job':job,'started_unix':time.time(),'fps':20,
                'config':cfg,'cameras':camera_map,'capture_test_fps':round(fps,2),'paired_ports':PORTS})
        op.checkpoint()
        # Seed the powered-off follower's target with its current position before
        # calling the original configure(). No altered gains or overload thresholds.
        leader.configure()
        pose=startup_check(leader,robot)
        robot.bus.sync_write('Goal_Position',pose,normalize=False)
        if robot.bus.sync_read('Goal_Position',normalize=False)!=pose:
            raise RuntimeError('初始保持目标未通过回读，拒绝上力')
        op.checkpoint()
        robot_configuring=True
        robot.configure()
        for ep in range(1,cfg['episodes']+1):
            if op.stopped(): break
            startup_check(leader,robot,cold=False)
            phase('recording',ep)
            record_loop(robot,Events(op,ep),20,dataset=ds,teleop=leader,
                        control_time_s=cfg['episode_s'],single_task=cfg['task'])
            if op.stopped(): break  # Stop cancels the incomplete episode.
            if ds.episode_buffer['size'] < 2: raise RuntimeError('本条片段太短，未保存')
            phase('saving',ep)
            ds.save_episode()
            saved+=1
            print('EPISODE_SAVED',saved,flush=True)
            if ep<cfg['episodes'] and not op.stopped():
                startup_check(leader,robot,cold=False)
                phase('resetting',ep)
                record_loop(robot,Events(op,0),20,teleop=leader,control_time_s=cfg['reset_s'])
    finally:
        errs=[]
        for arm in (robot,leader):
            if arm.bus.is_connected:
                try: arm.bus.disconnect(disable_torque=(arm is robot and robot_configuring))
                except Exception as exc: errs.append(f'{arm.id} 卸力/断开未确认：{exc}')
        for cam in robot.cameras.values():
            if cam.is_connected:
                try: cam.disconnect()
                except Exception as exc: errs.append(str(exc))
        if ds is not None:
            try:
                if ds.episode_buffer and ds.episode_buffer.get('size',0):
                    if not getattr(ds.image_writer,'_stopped',False): ds._wait_image_writer()
                    ds.clear_episode_buffer()
            finally:
                ds.stop_image_writer()
        if errs: raise RuntimeError('；'.join(errs))
    if not saved: raise RuntimeError('未保存任何完整演示')
    phase('closed')
    print('RECORDING_SAVED',root,saved,flush=True)


def main():
    mode=sys.argv[1]; job=sys.argv[2]
    if mode=='inventory':
        print(json.dumps(dict(devices=devices(),arms={k:Path(p).exists() for k,p in PORTS.items()}))); return
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',job): raise ValueError('无效任务编号')
    sess=BASE/'sessions'/job
    if mode=='stop':
        sess.mkdir(parents=True,exist_ok=True,mode=0o700)
        (sess/'stop').touch(); print('已请求结束；未完成片段不保存，支持住从臂'); return
    if mode=='finish':
        text=read(sess/'state.json'); state=json.loads(text) if text else {}
        if state.get('phase')!='recording': raise ValueError('当前未在采集片段')
        (sess/f"finish-{int(state['episode'])}").touch()
        print('已请求保存第',state['episode'],'条'); return
    if mode!='record': raise ValueError('未知操作')
    cfg=json.loads(base64.urlsafe_b64decode(sys.argv[3]))
    if cfg.get('onsite') is not True: raise RuntimeError('未确认有人在场')
    capture(job,cfg)


if __name__=='__main__': main()
