"""Serve paired RGB observations with SO101 follower state. No motor write opcodes are allowed."""

import argparse
from collections import Counter, deque
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from threading import Condition, Event, Thread
import time

import cv2
import numpy as np

FRONT = ("32e6:9221", "YOUR_FRONT_CAMERA_SERIAL")
WRIST = ("32e6:9251", "")
PORT = Path("/dev/serial/by-id/usb-1a86_USB_Single_Serial_YOUR_FOLLOWER_SERIAL-if00")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LOCK = Path.home() / "cie6007-edge/lab/sessions/capture.lock"


def camera_devices():
    identities = {}
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        if (node / "index").read_text().strip() != "0":
            continue
        usb = next((p for p in node.resolve().parents if (p / "idVendor").exists()), None)
        if usb is None:
            continue
        device = ":".join((usb / k).read_text().strip() for k in ("idVendor", "idProduct"))
        serial = (usb / "serial").read_text().strip() if (usb / "serial").exists() else ""
        identities.setdefault((device, serial), []).append("/dev/" + node.name)
    selected = []
    for identity in (FRONT, WRIST):
        matches = identities.get(identity, [])
        if len(matches) != 1:
            raise RuntimeError(f"Cannot identify exactly one camera for {identity}: {matches}")
        selected.append(matches[0])
    return selected


class CameraFeed:
    def __init__(self, node):
        self.node = node
        self.stop = Event()
        self.changed = Condition()
        self.frames = deque(maxlen=6)
        self.error = None
        self.thread = Thread(target=self._run, daemon=True)

    def _run(self):
        cap = cv2.VideoCapture(self.node, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise OSError(f"Cannot open {self.node}")
            for key, value in (
                (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
                (cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                (cv2.CAP_PROP_FPS, 30), (cv2.CAP_PROP_BUFFERSIZE, 2),
            ):
                cap.set(key, value)
            while not self.stop.is_set():
                ok, bgr = cap.read()
                received = time.monotonic()
                # V4L2 reports DQBUF timestamps in the monotonic clock domain for these UVC cameras.
                frame_stamp = cap.get(cv2.CAP_PROP_POS_MSEC) * 0.001
                if not ok or bgr is None or bgr.shape != (480, 640, 3):
                    raise RuntimeError(f"Invalid frame at {self.node}")
                if not 0 <= received - frame_stamp < 1:
                    raise RuntimeError(f"{self.node}: unavailable monotonic V4L2 frame timestamp")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with self.changed:
                    self.frames.append((frame_stamp, rgb))
                    self.changed.notify_all()
        except Exception as exc:
            with self.changed:
                self.error = exc
                self.changed.notify_all()
        finally:
            cap.release()

    def after(self, timestamp):
        with self.changed:
            self.changed.wait_for(lambda: self.error or
                                  (self.frames and self.frames[-1][0] > timestamp), timeout=1.2)
            if self.error:
                raise self.error
            newer = [x for x in self.frames if x[0] > timestamp]
            if not newer:
                raise TimeoutError(f"No new frame on {self.node}")
            return newer

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def open_follower(stack):
    from lerobot.robots.so101_follower.so101_follower import SO101Follower, SO101FollowerConfig
    from scservo_sdk.scservo_def import INST_PING, INST_READ, INST_SYNC_READ

    robot = SO101Follower(SO101FollowerConfig(port=str(PORT), id="so101_follower"))
    if tuple(robot.bus.motors) != JOINTS or not robot.calibration:
        raise RuntimeError("Follower joint order or calibration is missing")
    bus = robot.bus
    opcode_counts = Counter()
    original_tx = bus.packet_handler.txPacket

    def checked_tx(port, packet):
        instruction = packet[4]
        if instruction not in (INST_PING, INST_READ, INST_SYNC_READ):
            raise RuntimeError(f"Blocked non-read servo instruction {instruction}")
        opcode_counts[instruction] += 1
        return original_tx(port, packet)

    bus.packet_handler.txPacket = checked_tx
    stack.callback(lambda: bus.port_handler.closePort() if bus.is_connected else None)
    bus.connect()
    if not bus.is_calibrated:
        raise RuntimeError("Follower EEPROM disagrees with calibration file")
    torque = bus.sync_read("Torque_Enable", normalize=False)
    status = {joint: bus.read("Status", joint, normalize=False) for joint in JOINTS}
    if any(torque.values()) or any(status.values()):
        raise RuntimeError(f"Unexpected active torque or servo alarm: torque={torque}, status={status}")
    cal_sha = hashlib.sha256(robot.calibration_fpath.read_bytes()).hexdigest()
    return bus, {"joints": list(JOINTS), "calibration_sha256": cal_sha,
                 "torque": torque, "status": status, "read_opcodes": opcode_counts}


class JointFeed:
    """Timestamp the actual synchronous bus transaction in the cameras' monotonic clock."""
    def __init__(self, bus):
        self.bus = bus
        self.stop = Event()
        self.changed = Condition()
        self.history = deque(maxlen=24)
        self.error = None
        self.thread = Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            while not self.stop.is_set():
                before = time.monotonic()
                raw = self.bus.sync_read("Present_Position", normalize=False)
                after = time.monotonic()
                for joint, value in raw.items():
                    cal = self.bus.calibration[joint]
                    if not 0 <= value <= 4095 or not cal.range_min - 4 <= value <= cal.range_max + 4:
                        raise RuntimeError(f"{joint} raw reading outside calibrated single turn: {value}")
                normalized = self.bus._normalize({self.bus.motors[j].id: raw[j] for j in JOINTS})
                state = [normalized[self.bus.motors[j].id] for j in JOINTS]
                with self.changed:
                    self.history.append(((before + after) / 2, 1000 * (after - before), raw, state))
                    self.changed.notify_all()
                self.stop.wait(max(0, 0.02 - (time.monotonic() - before)))
        except Exception as exc:
            with self.changed:
                self.error = exc
                self.changed.notify_all()

    def near(self, when):
        with self.changed:
            self.changed.wait_for(lambda: self.error or (self.history and self.history[-1][0] >= when),
                                  timeout=0.1)
            if self.error:
                raise self.error
            if not self.history:
                return None
            return min(self.history, key=lambda entry: abs(entry[0] - when))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def sample(feeds, joints, previous, *, try_older=False):
    front, wrist = (feed.after(t) for feed, t in zip(feeds, previous))
    available = [(f, w) for f in front for w in wrist if abs(f[0] - w[0]) <= 0.025]
    available.sort(key=lambda p: (min(p[0][0], p[1][0]), -abs(p[0][0] - p[1][0])), reverse=True)
    for f, w in (available if try_older else available[:1]):
        if time.monotonic() - min(f[0], w[0]) > 0.12:
            break
        joint_sample = joints.near((f[0] + w[0]) / 2)
        if joint_sample is None:
            continue
        state_time, read_ms, raw, state = joint_sample
        state_skew_ms = 1000 * max(abs(state_time - f[0]), abs(state_time - w[0]))
        if state_skew_ms > 25:
            continue
        # A blocking state source may have waited since we selected the frames.
        age = time.monotonic() - min(f[0], w[0])
        if not 0 <= age <= 0.12:
            continue
        for label, frame in (("front", f[1]), ("wrist", w[1])):
            if float(frame[::16, ::16].std()) < 2:
                raise RuntimeError(f"{label}: uniform or dark frame")
        header = {"state": state, "raw": raw, "frame_time": [f[0], w[0]], "state_time": state_time,
                  "pair_skew_ms": 1000*abs(f[0]-w[0]), "state_skew_ms": state_skew_ms,
                  "state_read_ms": read_ms}
        return header, f[1], w[1]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serve", action="store_true", help="Binary output for the inference client")
    p.add_argument("--count", type=int, default=20)
    args = p.parse_args()
    stream = sys.stdout.buffer
    with ExitStack() as stack:
        lock = stack.enter_context(LOCK.open("rb"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        nodes = camera_devices()
        if not PORT.exists():
            raise RuntimeError("Follower serial port is unavailable")
        if subprocess.run(["/usr/bin/fuser", str(PORT.resolve()), *nodes],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 1:
            raise RuntimeError("Capture device in use or device check failed")
        bus, setup = open_follower(stack)
        joints = JointFeed(bus)
        joints.thread.start()
        stack.callback(joints.close)
        feeds = [CameraFeed(node) for node in nodes]
        for feed in feeds:
            feed.thread.start()
            stack.callback(feed.close)
        time.sleep(1.2)
        previous = [time.monotonic()-0.1] * 2
        for index in range(args.count):
            if args.serve and sys.stdin.buffer.read(1) != b"N":
                break
            sample_deadline = time.monotonic() + 2
            obs = None
            while obs is None and time.monotonic() < sample_deadline:
                obs = sample(feeds, joints, previous)
                if obs is None:
                    time.sleep(.004)
            if obs is None:
                raise TimeoutError("No fresh frame pair met age and skew limits")
            header, front, wrist = obs
            previous = header["frame_time"]
            header.update(index=index, devices=nodes, **setup)
            encoded = json.dumps(header).encode()
            if args.serve:
                stream.write(struct.pack("<I", len(encoded)))
                stream.write(encoded)
                for rgb in (front, wrist):
                    stream.write(memoryview(rgb))
                stream.flush()
            else:
                print(encoded.decode(), flush=True)


if __name__ == "__main__":
    main()
