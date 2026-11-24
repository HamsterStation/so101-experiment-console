"""Request a fresh 40×6 action chunk over a per-trial SSH stream."""
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time
import zlib

import cv2
import numpy as np

class RemotePredictor:
    inference_deadline = .40
    prefetch_age = 1.68
    switch_deadline = .47

    def __init__(self,bundle,stack):
        manifest=json.loads((bundle/'manifest.json').read_text())
        self.sock=socket.socket(socket.AF_UNIX)
        self.sock.settimeout(85)
        self.seq=0
        self.wire=None
        stack.callback(self.close)
        self.sock.connect(os.environ['SMOL46_SOCKET'])
        self.wire=self.sock.makefile('rwb',buffering=131072)
        expected=b'SM46'+bytes.fromhex(manifest['checkpoint_sha256'])
        if self.wire.read(len(expected))!=expected:
            raise RuntimeError('4080 权重或参考样本校验失败')
        self.sock.settimeout(2.)

    def close(self):
        if self.sock.fileno()<0: return
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError: pass
        if self.wire is not None: self.wire.close()
        self.sock.close()

    def infer(self,state,front,wrist):
        self.seq+=1
        seq=struct.pack('!I',self.seq)
        self.wire.write(seq+struct.pack('<6f',*state))
        for image in (front,wrist):
            if image.shape!=(480,640,3) or image.dtype!=np.uint8: raise ValueError('双摄格式错误')
            data=getattr(image,'mjpeg',None)
            if data is None:
                bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                ok,data=cv2.imencode('.png',bgr,[cv2.IMWRITE_PNG_COMPRESSION,1])
                if not ok: raise RuntimeError('双摄无损编码失败')
            self.wire.write(struct.pack('!II',len(data),zlib.crc32(memoryview(image))))
            self.wire.write(memoryview(data))
        self.wire.flush()
        packet=self.wire.read(964)
        if len(packet)!=964 or packet[:4]!=seq: raise RuntimeError('4080 动作回复中断或序号不一致')
        result=np.frombuffer(packet[4:],dtype='<f4').reshape(40,6)
        if not np.isfinite(result).all(): raise RuntimeError('4080 动作含非有限数值')
        return result

def transport_test(bundle):
    from contextlib import ExitStack
    with np.load(bundle/'case-0.npz') as case:
        state=case['state'][0]
        frames=[np.ascontiguousarray(case[k].transpose(1,2,0)) for k in ('front','wrist')]
    with ExitStack() as stack:
        predictor=RemotePredictor(bundle,stack)
        elapsed=[]
        for _ in range(12):
            start=time.perf_counter()
            predictor.infer(state,*frames)
            elapsed.append(round((time.perf_counter()-start)*1000,1))
        print(json.dumps({'transport_and_model_ms':elapsed,'motor_writes':0}))

# Preserve the camera's original MJPEG payload. Local checks and the GPU decode the
# same packet; the RGB checksum in every request verifies pixel-for-pixel identity.
from observations import CameraFeed
from smol46_trial import MAX_NEW_AGE

class CameraRGB(np.ndarray):
    pass

class PacketCamera(CameraFeed):
    @staticmethod
    def decode_packet(packet):
        data = packet.tobytes()
        # Some devices pad after EOI. An incomplete payload can decode to a
        # full-sized image with a corrupt lower half, so check both markers.
        if not data.startswith(b'\xff\xd8') or data.rfind(b'\xff\xd9') < 2:
            return None
        try:
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            return None
        if bgr is None or bgr.shape != (480, 640, 3):
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).view(CameraRGB)
        rgb.mjpeg = data
        return rgb

    def _run(self):
        cap=cv2.VideoCapture(self.node,cv2.CAP_V4L2)
        last_valid = None
        first_bad = None
        dropped = 0
        try:
            if not cap.isOpened(): raise OSError(f'无法打开 {self.node}')
            for key,value in ((cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')),
                              (cv2.CAP_PROP_FRAME_WIDTH,640),(cv2.CAP_PROP_FRAME_HEIGHT,480),
                              (cv2.CAP_PROP_FPS,30),(cv2.CAP_PROP_BUFFERSIZE,2),
                              (cv2.CAP_PROP_CONVERT_RGB,0)):
                cap.set(key,value)
            while not self.stop.is_set():
                ok,packet=cap.read()
                received=time.monotonic()
                rgb = self.decode_packet(packet) if ok and packet is not None else None
                if rgb is None:
                    dropped += 1
                    if first_bad is None:
                        first_bad = received
                    age = received - (last_valid if last_valid is not None else first_bad)
                    if age >= MAX_NEW_AGE:
                        raise TimeoutError(f'{self.node} 已有 {age*1000:.0f} ms 没有可用的 MJPEG 帧')
                    if not ok:
                        self.stop.wait(.005)
                    continue
                stamp=cap.get(cv2.CAP_PROP_POS_MSEC)*.001
                if not 0 <= received-stamp < 1:
                    raise RuntimeError(f'{self.node} 原始画面时间戳无效')
                if last_valid is not None and stamp <= last_valid:
                    continue
                last_valid = stamp
                first_bad = None
                with self.changed:
                    self.frames.append((stamp,rgb))
                    self.changed.notify_all()
        except Exception as exc:
            with self.changed:
                self.error=exc;self.changed.notify_all()
        finally:
            cap.release()
            if dropped:
                print(f'{self.node} 本轮丢弃 {dropped} 个无效相机帧', file=sys.stderr, flush=True)

RemotePredictor.camera_type=PacketCamera
