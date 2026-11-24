"""Supervised SO101 ACT trial. The default evaluates saved examples and never opens hardware."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LOW = np.array([-100.] * 5 + [0.])
HIGH = np.full(6, 100.)
# Calibrated control units, not degrees. A first trial limits each 50-ms target increment.
STEP = np.array([2., 5., 5., 3., 2., 2.])
START_DELTA = np.array([8., 12., 12., 8., 8., 8.])
FOLLOW_DELTA = np.array([12., 20., 20., 12., 10., 12.])
CALIBRATION_SHA = "3efa4045ab8f84f55a7d5ad6e4db17d5c2987393e067c3dc3d541ea54c4e567e"
STARTUP_MARGIN_TICKS = 10
RUN_MARGIN_TICKS = 4
DT = .05
# FP32 camera-to-output P95 was 194 ms. Discard inference arriving after 300 ms.
# A previously accepted plan expires at 400 ms (at most eight steps from capture).
MAX_NEW_AGE = .30
MAX_PLAN_AGE = .40
# At 20 Hz, query near h34, switch near h38 with a short overlap, expire after h39.
CHUNK_HORIZON_S = 40 * DT
CHUNK_PREFETCH_S = 34 * DT
CHUNK_SWITCH_S = 38 * DT
LAUNCH_DISPLACEMENT = 15.  # Calibrated units; sustained across two real encoder polls.


def check_target(target, state, limit=None, *, label="目标"):
    if target.shape != (6,) or state.shape != (6,) or not (np.isfinite(target).all() and np.isfinite(state).all()):
        raise ValueError("动作或位置数据无效")
    for i, joint in enumerate(JOINTS):
        if not LOW[i] <= target[i] <= HIGH[i]:
            raise ValueError(f"{joint} 目标 {target[i]:.2f} 超出机械范围 [{LOW[i]:g}, {HIGH[i]:g}]")
        if limit is not None and abs(target[i]-state[i]) > limit[i]:
            raise ValueError(f"{joint} {label}与实际位置相差 {target[i]-state[i]:.2f}，超过 {limit[i]:g}")


def command(chunk, state, age, previous, first=False, max_plan_age=MAX_PLAN_AGE, *, stepwise=False):
    if not (0 <= age < max_plan_age):
        raise TimeoutError(f"动作块已过期，图像年龄 {age*1000:.0f} ms")
    if chunk.shape != (40,6) or not np.isfinite(chunk).all():
        raise ValueError("动作块包含无效目标")
    limit = START_DELTA if first else FOLLOW_DELTA
    prediction_limit = None if stepwise else limit
    if first:
        check_target(chunk[0], state, prediction_limit)
    index = int(age / DT)
    target = chunk[index]
    check_target(target, state, prediction_limit)
    setpoint = previous + np.clip(target - previous, -STEP, STEP)
    if stepwise:
        check_target(setpoint, state, limit, label="本拍分步指令")
    return setpoint, index


class Predictor:
    def __init__(self, bundle, stack):
        env = dict(os.environ, PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)
        self.process = subprocess.Popen(
            ["/usr/bin/python3", "-u", str(Path(__file__).with_name("trial_policy.py")), str(bundle)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, start_new_session=True)
        stack.callback(self.close)
        if not select.select([self.process.stdout], [], [], 90)[0] or self.process.stdout.read(1) != b"R":
            raise RuntimeError("Jetson CUDA 模型加载或参考样本验证失败")

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()

    def infer(self, state, front, wrist):
        self.process.stdin.write(struct.pack("<6f", *state))
        for image in (front, wrist):
            if image.shape != (480,640,3) or image.dtype != np.uint8:
                raise ValueError("需要 640×480 RGB 图像")
            self.process.stdin.write(memoryview(np.ascontiguousarray(image)))
        self.process.stdin.flush()
        data = self.process.stdout.read(40*6*4)
        if len(data) != 40*6*4:
            raise RuntimeError("推理进程中断")
        return np.frombuffer(data, dtype="<f4").reshape(40,6)


def offline(bundle, predictor, *, stepwise=False):
    manifest = json.loads((bundle/"manifest.json").read_text())
    error_key = "max_error_vs_trtexec" if "tensorrt_engine" in manifest else "max_error_vs_4080"
    results = []
    for name in ("act-input-0.npz", "act-input-1.npz"):
        with np.load(bundle/name) as case:
            state = case["state"][0]
            images = [np.rint(case[k][0].transpose(1,2,0)*255).astype(np.uint8) for k in ("front", "wrist")]
            chunk = predictor.infer(state, *images)
            error = float(np.abs(chunk-case["ref"][0]).max())
            if error > .002:
                raise RuntimeError(f"{name} 输入输送后误差 {error} 超过阈值")
            try:
                _, index = command(chunk, state, .20, state, first=True, stepwise=stepwise)
                gate = f"允许模拟首步（索引 {index}）"
            except ValueError as exc:
                gate = str(exc)
            results.append({"source": name, error_key: error, "startup_gate": gate})
    result = dict(mode="offline", motor_writes=0, validation=results)
    captured = bundle/"shadow-fp32.json"
    if captured.is_file():
        shadow = json.loads(captured.read_text())
        if shadow["model_sha256"] != manifest["artifact_sha256"]["act-core.pt"]:
            raise ValueError("历史现场影子样本对应另一版本模型")
        rejected = 0
        outside = np.zeros(6, dtype=int)
        for obs in shadow["samples"]:
            target = np.array(obs["first_action"])
            state = np.array(obs["state"])
            outside += ((target < LOW) | (target > HIGH)).astype(int)
            try:
                command(np.tile(target, (40, 1)), state, 0., state, first=True, stepwise=stepwise)
            except ValueError:
                rejected += 1
        result["recorded_live_start_checks"] = dict(
            samples=len(shadow["samples"]), rejected=rejected,
            outside_calibrated_limits=dict(zip(JOINTS, outside.tolist())),
            camera_and_motor_reads_this_run=0)
    return result


class History(deque):
    def near(self, timestamp):
        if self:
            return min(self, key=lambda row: abs(row[0]-timestamp))


class BufferedFrames:
    """Nonblocking view for the 20-Hz controller; camera acquisition uses its own threads."""
    def __init__(self, feed):
        self.feed = feed

    def after(self, timestamp):
        with self.feed.changed:
            if self.feed.error:
                raise self.feed.error
            return [x for x in self.feed.frames if x[0] > timestamp]


def startup_edges(bus, raw):
    # Only joints at a calibrated endpoint at handover can carry its read tolerance.
    edges = {}
    for joint in JOINTS:
        cal = bus.calibration[joint]
        if raw[joint] <= cal.range_min:
            edges[joint] = "lower"
        elif raw[joint] >= cal.range_max:
            edges[joint] = "upper"
    return edges


def read_state(bus, history, edges=None, *, margin_ticks=RUN_MARGIN_TICKS):
    start = time.monotonic()
    raw = bus.sync_read("Present_Position", normalize=False)
    end = time.monotonic()
    for j in JOINTS:
        cal = bus.calibration[j]
        edge = edges.get(j) if edges else None
        lower = max(0, cal.range_min - (STARTUP_MARGIN_TICKS if edge == "lower" else margin_ticks))
        upper = min(4095, cal.range_max + (STARTUP_MARGIN_TICKS if edge == "upper" else margin_ticks))
        if not lower <= raw[j] <= upper:
            raise ValueError(f"{j} 编码器读数 {raw[j]} 超出允许范围 [{lower}, {upper}]"
                             f"（标定范围 [{cal.range_min}, {cal.range_max}]）")
        if (edge == "upper" and raw[j] < cal.range_max) or (edge == "lower" and raw[j] > cal.range_min):
            del edges[j]
    positions = bus._normalize({bus.motors[j].id:raw[j] for j in JOINTS})
    state = np.array([positions[bus.motors[j].id] for j in JOINTS])
    history.append(((start+end)/2, (end-start)*1000, raw, state))
    return raw, state


def check_health(bus, powered):
    torque = bus.sync_read("Torque_Enable", normalize=False)
    if any(torque[j] != int(powered) for j in JOINTS):
        raise RuntimeError(f"上力状态变化 {torque}")
    for joint in JOINTS:
        alarm = bus.read("Status", joint, normalize=False)
        if alarm:
            raise RuntimeError(f"{joint} 伺服报警 {alarm}")


def check_settings(robot):
    bus = robot.bus
    if not bus.is_calibrated or hashlib.sha256(robot.calibration_fpath.read_bytes()).hexdigest() != CALIBRATION_SHA:
        raise RuntimeError("从臂校准与采集版本不符")
    if tuple(bus.motors) != JOINTS or any(bus.read("Operating_Mode", j) != 0 for j in JOINTS):
        raise RuntimeError("伺服顺序或位置模式不符")
    for reg, expected in (("P_Coefficient",16), ("I_Coefficient",0), ("D_Coefficient",32)):
        if any(bus.read(reg,j,normalize=False) != expected for j in JOINTS):
            raise RuntimeError(f"从臂 {reg} 与官方采集配置不一致")
    for reg, expected in (("Max_Torque_Limit",500), ("Protection_Current",250), ("Overload_Torque",25)):
        if bus.read(reg, "gripper", normalize=False) != expected:
            raise RuntimeError(f"夹爪 {reg} 与官方采集配置不一致")
    check_health(bus, False)


def notice(message):
    # A disconnected SSH pseudo-terminal can already be returning EIO while stopping.
    try:
        print(message, flush=True)
    except OSError:
        pass


class TrialTrace:
    """Buffer at most 30 seconds of numeric telemetry, write outside the control loop."""
    def __init__(self, directory, bundle, mode):
        directory.mkdir(parents=True, exist_ok=True)
        created = datetime.now(timezone.utc)
        self.path = directory / f"{created:%Y%m%dT%H%M%SZ}-{mode}-{uuid4().hex[:12]}.json"
        self.commands_sent = 0
        self.rows = []
        self.plans = []
        self.data = dict(mode=mode, bundle=str(bundle.resolve()), created_utc=created.isoformat(),
                         joint_names=JOINTS, clock="monotonic seconds on Jetson",
                         units="calibrated LeRobot joint position", ticks=self.rows, plans=self.plans,
                         returned_setpoint_scope="Local return from send_action; not a bus acknowledgement",
                         goal_register_scope="Sampled every 10 ticks after any send; calibrated values and raw encoder counts",
                         phase="initializing")
        # Reserve a unique name and verify writability before any hardware access.
        with self.path.open("x") as output:
            json.dump(self.data, output)
        notice(f"试跑记录 {self.path}")

    def plan(self, chunk, stamp):
        self.plans.append(dict(exposure_s=stamp, actions=chunk.tolist()))
        return len(self.plans)-1

    def read_goal_register(self, row, bus):
        try:
            raw = bus.sync_read("Goal_Position", normalize=False)
            values = bus._normalize({bus.motors[j].id: raw[j] for j in JOINTS})
            row.update(goal_read_s=time.monotonic(), goal_register=[float(values[bus.motors[j].id]) for j in JOINTS],
                       goal_register_ticks=[int(raw[j]) for j in JOINTS],
                       goal_read_after_commands=self.commands_sent)
        except Exception as exc:
            row["goal_read_error"] = f"{type(exc).__name__}: {exc}"

    def save(self, phase):
        self.data.update(phase=phase, commands_sent=self.commands_sent)
        summary = {}
        for key in ("measured", "raw_target", "bounded_target", "returned_setpoint", "goal_register"):
            rows = [row[key] for row in self.rows if key in row]
            if rows:
                a = np.asarray(rows)
                summary[key] = dict(first=a[0].tolist(), last=a[-1].tolist(),
                                    span=np.ptp(a, axis=0).tolist())
        self.data["summary"] = summary
        partial = self.path.with_suffix(".tmp")
        try:
            with partial.open("w") as output:
                json.dump(self.data, output, separators=(",", ":"))
            partial.replace(self.path)
        except OSError as exc:
            # Telemetry failures must never interrupt holding or releasing the arm.
            notice(f"写入试跑记录失败 {self.path}：{exc}")


def release(bus, trace=None, operator=None):
    # Freeze healthy powered joints at their current positions. Never reactivate a faulted joint.
    for j in JOINTS:
        try:
            if bus.read("Status", j, normalize=False):
                raise RuntimeError("伺服有报警")
            if bus.read("Torque_Enable", j, normalize=False) == 1:
                raw = bus.read("Present_Position",j,normalize=False)
                c = bus.calibration[j]
                if c.range_min <= raw <= c.range_max:
                    bus.write("Goal_Position",j,raw,normalize=False)
        except Exception as exc:
            notice(f"{j} 保持未确认：{exc}")
            if j == "gripper":
                try:
                    bus.disable_torque(j)
                    if bus.read("Torque_Enable",j,normalize=False) != 0:
                        raise RuntimeError("夹爪仍在上力")
                    notice("故障夹爪已卸力")
                except Exception as gripper_error:
                    notice(f"夹爪卸力未确认，请托稳后切断从臂外接电源：{gripper_error}")
    if trace is not None:
        trace.save("waiting_for_support")
    try:
        if operator is None:
            input("跟随已停止。托稳从臂，放稳物体，再按 Enter 卸力。")
        else:
            operator.confirm_support()
    except (EOFError, OSError, KeyboardInterrupt):
        if trace is not None:
            trace.data["release"] = "support_not_confirmed"
        notice("未收到支撑确认。请在现场托稳后断开从臂外接电源。")
        return
    uncertain = []
    for j in JOINTS:
        try:
            bus.disable_torque(j)
            if bus.read("Torque_Enable",j,normalize=False) != 0:
                uncertain.append(j)
        except Exception:
            uncertain.append(j)
    if trace is not None:
        trace.data["release"] = {"unconfirmed_joints": uncertain}
    if uncertain:
        raise RuntimeError(f"下列关节卸力未确认 {uncertain}。请保持支撑并断开从臂外接电源。")
    if operator is not None:
        operator.transition("released")


def reserve_hardware(stack):
    from observations import LOCK
    lock = stack.enter_context(LOCK.open("rb"))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Jetson 正在录制或试跑，请先结束原会话") from exc


def onsite(bundle, predictor, seconds, stack, check_only=False, timing=False, trace=None, action_mode="replan", operator=None, reserved=False, *, stepwise=False, goal=None, align_inputs=False):
    from observations import CameraFeed, PORT, camera_devices, sample
    from lerobot.robots.so101_follower.so101_follower import SO101Follower, SO101FollowerConfig
    if not reserved:
        reserve_hardware(stack)
    nodes = camera_devices()
    if subprocess.run(["/usr/bin/fuser", str(PORT.resolve()), *nodes], capture_output=True).returncode != 1:
        raise RuntimeError("录制、遥操或预览正在占用设备")
    robot = SO101Follower(SO101FollowerConfig(port=str(PORT), id="so101_follower"))
    bus = robot.bus
    stack.callback(lambda: bus.port_handler.closePort() if bus.is_connected else None)
    # PING, READ, SYNC_READ are the only instructions before on-site confirmation.
    opcodes = Counter()
    from scservo_sdk.scservo_def import INST_PING, INST_READ, INST_SYNC_READ
    transmit = bus.packet_handler.txPacket
    def read_only(port, packet):
        if packet[4] not in (INST_PING, INST_READ, INST_SYNC_READ):
            raise RuntimeError("预检禁止写入舵机")
        opcodes[packet[4]] += 1
        return transmit(port,packet)
    bus.packet_handler.txPacket = read_only
    bus.connect()  # Official bus connects without configuring or enabling the motors.
    check_settings(robot)
    feeds = [CameraFeed(n) for n in nodes]
    for f in feeds:
        f.thread.start()
        stack.callback(f.close)
    views = [BufferedFrames(f) for f in feeds]
    history = History(maxlen=24)
    previous_frames = [-1.,-1.]
    def latest_exposures():
        stamps = []
        for feed in feeds:
            with feed.changed:
                if feed.error:
                    raise feed.error
                stamps.append(feed.frames[-1][0] if feed.frames else None)
        return stamps
    camera_times = latest_exposures if align_inputs else None
    def observation():
        if align_inputs:
            value = sample(views, history, previous_frames, try_older=True)
        else:
            value = sample(views, history, previous_frames)
        if value:
            previous_frames[:] = value[0]["frame_time"]
        return value

    start_check = {}
    def fresh_start(validate=True):
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            raw, state = read_state(bus,history,margin_ticks=STARTUP_MARGIN_TICKS)
            value = observation()
            if value is not None:
                obs,f,w = value
                stamp = min(obs["frame_time"])
                chunk = predictor.infer(obs["state"],f,w)
                _, state = read_state(bus,history,margin_ticks=STARTUP_MARGIN_TICKS)
                age = time.monotonic()-stamp
                if age > MAX_NEW_AGE:
                    raise TimeoutError(f"首个动作块延迟 {age*1000:.0f} ms")
                if validate:
                    first, index = command(chunk,state,age,state,first=True,stepwise=stepwise)
                    start_check.update(image_age_ms=age*1000, action_index=index, state=state.tolist(),
                                       raw_first_action=chunk[0].tolist(), aligned_action=chunk[index].tolist(),
                                       first_bounded_setpoint=first.tolist())
                return chunk, stamp
            time.sleep(.01)
        raise TimeoutError("双摄和关节未产生对时的新输入")

    time.sleep(1.2)
    if timing:
        chunk, stamp = fresh_start(validate=False)
        _, state = read_state(bus,history,margin_ticks=STARTUP_MARGIN_TICKS)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            measured = control_loop(robot,predictor,pool,history,observation,chunk,stamp,state,seconds,timing=True,trace=trace,action_mode=action_mode,stepwise=stepwise,camera_times=camera_times)
            check_health(bus,False)
            measured["read_opcodes"] = dict(opcodes)
            return measured
        finally:
            predictor.close()
            pool.shutdown(wait=True, cancel_futures=True)
    fresh_start()
    if check_only:
        return {"mode":"read-only-start-check","startup_passed":True,"motor_writes":0,
                "read_opcodes":dict(opcodes),"check":start_check}
    confirmed = operator.confirm_start() if operator is not None else input("现场已清空、方块已放好、从臂有支撑且可立即断电。我本人在从臂旁并能立刻断电。输入 START 确认：").strip() == "START"
    if not confirmed:
        return {"mode":"cancelled","motor_writes":0}
    previous_frames[:] = [time.monotonic()-.08]*2
    chunk, stamp = fresh_start()
    check_health(bus, False)
    raw, state = read_state(bus,history,margin_ticks=STARTUP_MARGIN_TICKS)
    command(chunk,state,time.monotonic()-stamp,state,first=True,stepwise=stepwise)
    hold = {j:min(bus.calibration[j].range_max,max(bus.calibration[j].range_min,raw[j])) for j in JOINTS}
    if operator is not None and operator.stop_requested():
        return {"mode":"cancelled","motor_writes":0}
    bus.packet_handler.txPacket = transmit
    bus.sync_write("Goal_Position",hold,normalize=False)
    if bus.sync_read("Goal_Position",normalize=False) != hold:
        raise RuntimeError("上力前保持目标回读不一致")
    # Only this branch can power the follower; protect against a partial enable on an exception.
    fresh_raw, _ = read_state(bus,history,margin_ticks=STARTUP_MARGIN_TICKS)
    if any(abs(fresh_raw[j] - raw[j]) > 12 for j in JOINTS):
        raise RuntimeError("上力前从臂发生移动，请托稳后重试")
    if time.monotonic()-stamp > MAX_NEW_AGE:
        raise TimeoutError("上力前图像过期")
    if operator is not None and operator.stop_requested():
        return {"mode":"cancelled","powered":False}
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        if trace is not None:
            trace.data["torque_enable_attempted"] = True
        bus.enable_torque()
        check_health(bus,True)
        if operator is not None:
            operator.transition("running")
        return control_loop(robot, predictor, pool, history, observation, chunk, stamp, state, seconds, timing=False,trace=trace,action_mode=action_mode,operator=operator,stepwise=stepwise,goal=goal,camera_times=camera_times)
    except BaseException as exc:
        if trace is not None:
            trace.data["control_error"] = f"{type(exc).__name__}: {exc}"
        if operator is not None:
            notice(f"TRIAL_ERROR {exc}")
        raise
    finally:
        try:
            release(bus, trace, operator)
        finally:
            predictor.close()
            pool.shutdown(wait=True, cancel_futures=True)


def control_loop(robot, predictor, pool, history, observation, chunk, stamp, state, seconds, timing, trace=None, action_mode="replan", operator=None, *, stepwise=False, goal=None, camera_times=None):
    # The latest feedback is the final unpowered measurement for this session.
    edges = startup_edges(robot.bus, history[-1][2]) if history else {}
    previous = state.copy()
    pending = None
    queued = None
    sequential = action_mode == "chunk"
    launching = action_mode == "launch"
    watch_cameras = sequential or launching
    initial = state.copy()
    advanced_ticks = 0
    stalled_ticks = 0
    expired_plans = 0
    last_camera_stamp = stamp
    switches = 0
    ticks = []
    bus_ms = []
    budget_ms = []
    ages_ms = []
    command_errors = Counter()
    late_ticks = 0
    misses = 0
    replan_ages_ms = []
    start = time.monotonic()
    if trace is not None:
        plan_id = trace.plan(chunk,stamp)
        trace.data.update(loop_start_s=start, initial_state=state.tolist(), action_mode=action_mode)
    sent = 0
    completion = None
    for i in range(int(np.ceil(seconds / DT))):
        tick = start + i * DT
        if operator is not None and operator.stop_requested():
            break
        if operator is None and not timing and select.select([sys.stdin],[],[],0)[0]:
            sys.stdin.readline()
            break
        now = time.monotonic()
        ticks.append(now)
        if trace is not None:
            row = dict(i=i, tick_s=now)
            trace.rows.append(row)
        if now-tick > DT*.5:
            late_ticks += 1
            if not timing:
                raise TimeoutError("20 Hz 控制循环超期")
        raw,state = read_state(robot.bus,history,edges)
        if trace is not None:
            row.update(measurement_s=history[-1][0], measured=state.tolist(),
                       encoder_ticks=[int(raw[j]) for j in JOINTS])
        check_health(robot.bus, not timing)
        bus_ms.append(1000*(time.monotonic()-now))
        if pending is not None and pending.done():
            new_chunk = pending.result()
            latency = time.monotonic()-new_stamp
            replan_ages_ms.append(1000*latency)
            if latency > MAX_NEW_AGE and (not timing or watch_cameras):
                raise TimeoutError("新动作块迟到")
            queued = (new_chunk, new_stamp)
            pending = None
        now = time.monotonic()
        if launching:
            shoulder_delta = float(state[1]-initial[1])
            elbow_delta = float(initial[2]-state[2])
            progress = max(shoulder_delta, elbow_delta)
            advanced_ticks = advanced_ticks+1 if progress >= LAUNCH_DISPLACEMENT else 0
            demanded = max(previous[1]-initial[1], initial[2]-previous[2])
            stalled_ticks = stalled_ticks+1 if demanded >= 8 and progress < 2 else 0
            if trace is not None:
                row.update(shoulder_lift_from_start=shoulder_delta, elbow_extension_from_start=elbow_delta,
                           progress_confirmations=advanced_ticks, launch_stall_ticks=stalled_ticks)
            if stalled_ticks >= 6:
                raise RuntimeError("起步目标已前移，但连续 6 拍从臂几乎未动，停止试跑")
            if now-stamp >= CHUNK_HORIZON_S:
                raise TimeoutError("首个动作块已到期，从臂尚未确认完成起步，停止试跑")
        use_candidate = advanced_ticks >= 2 if launching else not sequential or now-stamp >= CHUNK_SWITCH_S
        if queued is not None and use_candidate:
            next_chunk, next_stamp = queued
            if launching and not 0 <= now-next_stamp <= MAX_NEW_AGE:
                raise TimeoutError("起步后没有足够新鲜的预测可切换")
            if sequential and not 0 <= now-next_stamp < MAX_PLAN_AGE:
                raise TimeoutError("切换时下一动作块的观测已过期")
            if launching:
                launching = False
                if trace is not None:
                    trace.data["launch_handover"] = dict(tick=i, preceding_plan=plan_id,
                                                       new_exposure_s=next_stamp, state=state.tolist())
            chunk, stamp = next_chunk, next_stamp
            switches += 1
            queued = None
            if trace is not None:
                plan_id = trace.plan(chunk, stamp)
        value = None
        if watch_cameras or pending is None or goal is not None:
            value = observation()
            if value is not None:
                obs,f,w = value
                last_camera_stamp = min(obs["frame_time"])
                if pending is None and (launching or queued is None) and (not sequential or now-stamp >= CHUNK_PREFETCH_S):
                    new_stamp = last_camera_stamp
                    pending = pool.submit(predictor.infer, obs["state"], f,w)
        if watch_cameras:
            now = time.monotonic()
            age_ms = (now-last_camera_stamp)*1000
            if trace is not None:
                row.update(camera_exposure_s=last_camera_stamp, camera_age_ms=age_ms)
                if camera_times is not None and value is not None:
                    row.update(pair_skew_ms=obs["pair_skew_ms"], state_skew_ms=obs["state_skew_ms"])
            if not 0 <= now-last_camera_stamp <= MAX_NEW_AGE:
                if camera_times is not None:
                    stamps = camera_times()
                    ages = [None if t is None else round(1000*(now-t), 1) for t in stamps]
                    if trace is not None:
                        row["front_age_ms"], row["wrist_age_ms"] = ages
                    raise TimeoutError(f"对时观测距今 {age_ms:.0f} ms，停止播放动作块"
                                       f"（顶摄最新帧 {ages[0]} ms，腕摄最新帧 {ages[1]} ms）")
                raise TimeoutError("双摄实时观测已超时，停止播放动作块")
            if pending is not None and now-new_stamp > MAX_NEW_AGE:
                raise TimeoutError("后续动作块推理超时")
            # Reject an expired plan in read-only measurement as well as the powered trial.
            if now-stamp >= CHUNK_HORIZON_S:
                raise TimeoutError("预测动作块的 40 步已用完，下一块尚未就绪")
        if goal is not None and not timing:
            completion = goal.check(time.monotonic(), value)
            if completion is not None:
                if trace is not None:
                    trace.data["completion"] = completion
                notice("TRIAL_SUCCESS cube_in_box")
                notice("方块已入盒，任务完成")
                break
        # Include wake jitter, six status reads, state, RGB pairing and starting GPU work.
        elapsed = time.monotonic()-tick
        budget_ms.append(1000*elapsed)
        if elapsed > DT*.5:
            misses += 1
            if not timing:
                raise TimeoutError("传感器查询超过控制周期预算")
        age = time.monotonic()-stamp
        ages_ms.append(1000*age)
        max_plan_age = CHUNK_HORIZON_S if sequential or launching else MAX_PLAN_AGE
        expired_plans += int(age >= max_plan_age)
        if trace is not None:
            index = int(age / DT)
            row.update(plan=plan_id, phase="launch" if launching else ("chunk" if sequential else "replan"),
                       age_ms=age*1000, action_index=index,
                       first_action=chunk[0].tolist(), previous_target=previous.tolist())
            if 0 <= index < len(chunk):
                row["raw_target"] = chunk[index].tolist()
        try:
            target,_ = command(chunk,state,age,previous,first=(i==0),max_plan_age=max_plan_age,stepwise=stepwise)
        except (ValueError, TimeoutError) as exc:
            if trace is not None:
                row["gate_error"] = str(exc)
            if not timing or action_mode == "launch":
                raise
            command_errors[str(exc).split(" ")[0]] += 1
        else:
            if trace is not None:
                row["bounded_target"] = target.tolist()
            if not timing:
                if trace is not None:
                    row["send_started_s"] = time.monotonic()
                returned = robot.send_action({f"{j}.pos":float(v) for j,v in zip(JOINTS,target)})
                sent += 1
                if trace is not None:
                    trace.commands_sent = sent
                    row["send_returned_s"] = time.monotonic()
                    if isinstance(returned,dict):
                        row["returned_setpoint"] = [float(returned[f"{j}.pos"]) for j in JOINTS]
            previous = target
        if trace is not None and i % 10 == 0:
            trace.read_goal_register(row, robot.bus)
        if camera_times is not None and watch_cameras and time.monotonic() < tick + DT/2:
            # Read actual joint feedback between consecutive 20 Hz command ticks.
            time.sleep(max(0., tick + DT/2 - time.monotonic()))
            if time.monotonic() < tick + DT - .008:
                read_state(robot.bus, history, edges)
        time.sleep(max(0.,tick+DT-time.monotonic()))
    if not timing:
        result = {"mode":"onsite","action_mode":action_mode,"commands_sent":sent,"plan_switches":switches}
        if completion is not None:
            result["stop_reason"] = completion["reason"]
        return result

    def summary(values):
        if len(values) == 0:
            return None
        return {key: round(float(np.percentile(values, q)),3)
                for key,q in (("p50",50),("p95",95),("max",100))}
    return dict(mode="read-only-cadence",action_mode=action_mode,plan_switches=switches,
                last_plan_window_ms=1000*max_plan_age, expired_plans=expired_plans,
                ticks=len(ticks),motor_writes=0,
                effective_hz=((len(ticks)-1)/(ticks[-1]-ticks[0])) if len(ticks)>1 else None,
                control_interval_ms=summary(np.diff(ticks)*1000), state_and_six_alarms_ms=summary(bus_ms),
                full_precommand_budget_ms=summary(budget_ms), missed_25ms_budget=misses,
                wake_over_25ms=late_ticks, replans=len(replan_ages_ms), selected_plan_age_ms=summary(ages_ms),
                replan_age_at_tick_ms=summary(replan_ages_ms),
                new_plan_over_300ms=sum(age>300 for age in replan_ages_ms),
                **({} if sequential else {"plan_over_400ms":sum(age>=400 for age in ages_ms)}),
                discarded_action_gate_errors=dict(command_errors))



def main(*, stepwise=False, goal=None, align_inputs=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-start", action="store_true", help="Check the present pose and paired cameras with read-only servo instructions")
    mode.add_argument("--measure-loop", action="store_true", help="Measure real 20-Hz observation and inference scheduling with all actuator writes forbidden")
    mode.add_argument("--onsite", action="store_true", help="Supervised experiment from an interactive foreground terminal, including SSH")
    parser.add_argument("--action-mode", choices=("replan","chunk","launch"), default="replan",
                        help="Immediate replanning, continuous chunks, or first-chunk launch then immediate replanning")
    parser.add_argument("--seconds", type=int, default=15, choices=range(1,31), metavar="1..30")
    parser.add_argument("--console", action="store_true", help="现场控制台的独立开始、停止及卸力输入")
    args = parser.parse_args()
    operator = None
    if args.console:
        if not args.onsite or sys.stdin.isatty():
            parser.error("控制台需要独立的现场会话")
        from trial_operator import ConsoleOperator
        operator = ConsoleOperator(os.fdopen(os.dup(sys.stdin.fileno()), "r"), notice)
    if args.onsite:
        if operator is None and (not sys.stdin.isatty() or not sys.stdout.isatty() or os.tcgetpgrp(sys.stdin.fileno()) != os.getpgrp()):
            parser.error("真机试跑须在前台交互终端启动；在机械臂旁使用 Mac 时可用 ssh -t")
        stop_requested = False
        def stop(*_):
            nonlocal stop_requested
            if not stop_requested:
                stop_requested = True
                raise InterruptedError("控制进程收到停止信号")
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig,stop)
        if operator is None:
            print("现场试跑。按 Enter 或 Ctrl+C 停止；遇到碰撞趋势立即断开从臂电源。", flush=True)
    trace = None
    if args.onsite or args.measure_loop:
        trace = TrialTrace(Path(__file__).resolve().parent/"traces", args.bundle,
                           "onsite" if args.onsite else "read-only-cadence")
        trace.data["tracking_check"] = "step_limited_setpoint" if stepwise else "raw_prediction"
    with ExitStack() as stack:
        if trace is not None:
            def persist(exception_type, exc, _tb):
                if exc is not None:
                    trace.data["session_error"] = f"{exception_type.__name__}: {exc}"
                trace.save("failed" if exc is not None else "finished")
            stack.push(persist)
        hardware = args.onsite or args.check_start or args.measure_loop
        if hardware:
            reserve_hardware(stack)
        predictor = Predictor(args.bundle,stack)
        if hardware:
            result = onsite(args.bundle,predictor,args.seconds,stack,check_only=args.check_start,timing=args.measure_loop, trace=trace, action_mode=args.action_mode, operator=operator, reserved=True,stepwise=stepwise,goal=goal,align_inputs=align_inputs)
        else:
            result = offline(args.bundle,predictor,stepwise=stepwise)
        if trace is not None:
            result["trace"] = str(trace.path)
            trace.data["result"] = result
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
