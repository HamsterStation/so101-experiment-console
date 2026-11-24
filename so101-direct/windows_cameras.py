"""Native DirectShow capture on the same laptop as the WSL controller."""
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import time

import av
import numpy as np
from smol46_local import PacketCamera

BIN = Path('/mnt/c/so101-direct-tools')
FFMPEG = BIN / 'capture-ffmpeg.exe'


@lru_cache(maxsize=1)
def camera_devices():
    probe = subprocess.run([FFMPEG, '-hide_banner', '-list_devices', 'true',
                            '-f', 'dshow', '-i', 'dummy'], stdin=subprocess.DEVNULL, capture_output=True, timeout=12)
    alternatives = re.findall(r'Alternative name "([^"]+)"', probe.stderr.decode(errors='replace'))
    ordered = []
    for label, pid in [('顶摄', '9221'), ('腕摄', '9251')]:
        matches = [name for name in alternatives if 'vid_32e6&pid_' + pid + '&' in name.lower()]
        if len(matches) != 1:
            raise RuntimeError(f'Windows 未能唯一识别{label} (32e6:{pid})')
        ordered.append(matches[0])
    return ordered


class CaptureClock:
    """Translate DirectShow timestamps through independent native-clock readings."""
    def __init__(self, process):
        self.process = process
        self.anchor = None
        self.rate = 1.0

    def translate(self, packet_time):
        start = time.monotonic()
        self.process.stdin.write(b'now\n')
        self.process.stdin.flush()
        host = int(self.process.stdout.readline()) / 10_000_000
        end = time.monotonic()
        self.last_roundtrip_ms = round((end - start) * 1000, 1)
        self.last_packet_age_ms = round((host - packet_time) * 1000, 1)
        if end - start > .010:
            return None
        middle = (start + end) * .5
        if self.anchor is None:
            self.anchor = host, middle
        elif host - self.anchor[0] >= .5:
            self.rate = (middle - self.anchor[1]) / (host - self.anchor[0])
            self.anchor = host, middle
        if not .85 <= self.rate <= 1.15:
            raise RuntimeError('Windows 与 WSL 时钟速率异常')
        native_age = host - packet_time
        # DirectShow and the MJPEG mux round clock ticks. Bound that ambiguity;
        # an old packet keeps its original capture age, even after a stalled pipe.
        if native_age < -.020 or native_age > .30:
            return None
        return start - (max(0., native_age) + .010) * self.rate


class WindowsCamera(PacketCamera):
    def __init__(self, node):
        super().__init__(node)
        self.process = None
        self.stats = dict(packets=0, accepted=0, clock_rejects=0, stale_or_duplicate=0, bad_jpeg=0)

    def _run(self):
        # Timestamp at the native capture callback, before FFmpeg queues or writes the packet.
        # The top camera's driver synthesizes a nominal 30 fps sample clock in low light.
        flags = [FFMPEG, '-nostdin', '-hide_banner', '-loglevel', 'error',
                 '-f', 'dshow', '-use_video_device_timestamps', '0',
                 '-video_size', '640x480', '-vcodec', 'mjpeg',
                 '-probesize', '32', '-analyzeduration', '1', '-i', 'video=' + self.node,
                 '-map', '0:v:0', '-c:v', 'copy', '-copyts', '-copytb', '1',
                 '-avoid_negative_ts', 'disabled', '-flush_packets', '1', '-f', 'nut', 'pipe:1']
        try:
            with subprocess.Popen([BIN/'HostClock.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as helper, \
                    subprocess.Popen(flags, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, bufsize=0) as proc:
                clock = CaptureClock(helper)
                self.process = proc
                try:
                    with av.open(proc.stdout, format='nut', options={'analyzeduration':'1','probesize':'32'}) as video:
                        last_stamp = float('-inf')
                        for pkt in video.demux(video=0):
                            if self.stop.is_set():
                                return
                            if not pkt.size or pkt.pts is None:
                                continue
                            self.stats["packets"] += 1
                            stamp = clock.translate(float(pkt.pts * pkt.time_base))
                            if stamp is None:
                                self.stats["clock_rejects"] += 1
                                self.stats["last_clock_rtt_ms"] = clock.last_roundtrip_ms
                                self.stats["last_rejected_packet_age_ms"] = clock.last_packet_age_ms
                                continue
                            now = time.monotonic()
                            # The DirectShow graph clock advances in milliseconds.
                            if stamp <= last_stamp or not 0 <= now - stamp <= .30:
                                self.stats["stale_or_duplicate"] += 1
                                continue
                            rgb = self.decode_packet(np.frombuffer(pkt, dtype=np.uint8))
                            if rgb is None:
                                self.stats["bad_jpeg"] += 1
                                continue
                            last_stamp = stamp
                            with self.changed:
                                self.stats["accepted"] += 1
                                self.frames.append((stamp, rgb))
                                self.changed.notify_all()
                    if not self.stop.is_set():
                        raise RuntimeError('Windows 摄像头采集流已结束')
                finally:
                    helper.stdin.close()
                    proc.stdout.close()
                    if proc.poll() is None:
                        proc.terminate()
        except Exception as exc:
            if not self.stop.is_set():
                with self.changed:
                    self.error = exc
                    self.changed.notify_all()

    def close(self):
        self.stop.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.thread.join(timeout=3)
