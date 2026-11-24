"""Exercise real controller scheduling and startup decisions without devices, using the calibrated control units."""
from contextlib import ExitStack
from unittest import TestCase, main, mock
from types import SimpleNamespace
import sys
import json
import tempfile
import time
from pathlib import Path
import numpy as np

from trial import TrialTrace, History, control_loop, command, check_health, onsite, release, JOINTS, STEP


class SafetyTests(TestCase):
    def setUp(self):
        self.state = np.array([0.,0.,0.,0.,0.,35.])
        self.chunk = np.tile(self.state,(40,1))

    def test_crossing_first_row_cannot_be_skipped(self):
        self.chunk[0,1] = -116.6
        with self.assertRaisesRegex(ValueError,"shoulder_lift"):
            command(self.chunk,self.state,.20,self.state,first=True)

    def test_reject_invalid_lag_compensated_row(self):
        self.chunk[4,2] = 107.
        with self.assertRaisesRegex(ValueError,"elbow_flex"):
            command(self.chunk,self.state,.20,self.state,first=True)

    def test_first_motion_and_stale_rejected(self):
        self.chunk[0,2] += 24
        with self.assertRaisesRegex(ValueError,"elbow"):
            command(self.chunk,self.state,0,self.state,first=True)
        with self.assertRaises(TimeoutError):
            command(self.chunk,self.state,.40,self.state)
        self.chunk[8,4] = np.nan
        with self.assertRaises(ValueError):
            command(self.chunk,self.state,.22,self.state)

    def test_absolute_units_rate_limited_after_validation(self):
        self.chunk[4] += STEP*1.5
        target,index = command(self.chunk,self.state,.202,self.state,first=True)
        np.testing.assert_array_equal(target, self.state+STEP)
        self.assertEqual(index,4)
        self.chunk[4,5] = -1.
        with self.assertRaisesRegex(ValueError,"gripper"):
            command(self.chunk,self.state,.20,self.state)

    def test_alarm_aborts_and_no_torque_enable_on_release(self):
        bus = mock.Mock()
        bus.sync_read.return_value = dict.fromkeys(JOINTS,1)
        def read(reg,joint,**_):
            if reg=="Status":
                return 32 if joint=="gripper" else 0
            return 0
        bus.read.side_effect = read
        with self.assertRaisesRegex(RuntimeError,"gripper"):
            check_health(bus,True)
        with mock.patch("builtins.input",return_value=""):
            release(bus)
        bus.enable_torque.assert_not_called()
        bus.write.assert_not_called()
        self.assertEqual(bus.disable_torque.call_count,7)
        self.assertEqual(bus.disable_torque.call_args_list[0],mock.call("gripper"))


    def test_disconnected_terminal_does_not_drop_supported_axes(self):
        bus = mock.Mock()
        bus.calibration = {j:SimpleNamespace(range_min=900,range_max=3200) for j in JOINTS}
        def read(reg,j,**kw):
            return {"Status":0, "Torque_Enable":1, "Present_Position":2000}[reg]
        bus.read.side_effect = read
        with mock.patch("builtins.input", side_effect=EOFError), mock.patch("builtins.print", side_effect=OSError):
            release(bus)
        self.assertEqual(bus.write.call_count,6)
        bus.enable_torque.assert_not_called()
        bus.disable_torque.assert_not_called()

    def test_freeze_healthy_joints_release_after_support(self):
        timeline = []
        bus = mock.Mock()
        powered = set(JOINTS)
        bus.calibration = {j:SimpleNamespace(range_min=900,range_max=3200) for j in JOINTS}
        def read(reg,j,**kw):
            if reg=="Status" and j=="gripper":
                raise RuntimeError("RxPacketError Overload")
            if reg=="Torque_Enable":
                return int(j in powered)
            if reg=="Present_Position":
                return 1923
            return 0
        bus.read.side_effect = read
        def disable(j):
            powered.discard(j)
            timeline.append(("disable",j))
        bus.disable_torque.side_effect = disable
        bus.write.side_effect = lambda reg,j,val,**kw: timeline.append(("hold",j,val))
        with mock.patch("builtins.input", side_effect=lambda _: timeline.append(("supported",))):
            release(bus)
        marker = timeline.index(("supported",))
        self.assertEqual(timeline[:marker], [("hold",j,1923) for j in JOINTS[:-1]]+[("disable","gripper")])
        self.assertEqual(timeline[marker+1:], [("disable",j) for j in JOINTS])
        self.assertFalse(powered)
        bus.enable_torque.assert_not_called()


class TraceTests(TestCase):
    def exercise(self, directory, read_only=False, fail_second=False):
        state = np.array([0.,0.,0.,0.,0.,35.])
        pred = np.tile(state + 1,(40,1))
        bus = mock.Mock()
        bus.motors = {j:SimpleNamespace(id=i+1) for i,j in enumerate(JOINTS)}
        bus.sync_read.return_value = dict.fromkeys(JOINTS, 1999)
        bus._normalize.return_value = dict(zip(range(1,7),state+0.8))
        robot = SimpleNamespace(bus=bus,send_action=mock.Mock())
        returned = {f"{j}.pos":v for j,v in zip(JOINTS,state+0.5)}
        robot.send_action.side_effect = ([returned,RuntimeError("device gone")]
                                         if fail_second else [returned]*4)
        history = History(maxlen=24)
        def read(*_):
            now = time.monotonic()
            raw = dict.fromkeys(JOINTS, 2000)
            history.append((now,0.,raw,state))
            return raw,state
        mode = "read-only-cadence" if read_only else "onsite"
        trace = TrialTrace(directory, directory/"model", mode)
        with mock.patch("trial.read_state",side_effect=read), mock.patch("trial.check_health"), \
             mock.patch("trial.select.select",return_value=([],[],[])):
            def run():
                control_loop(robot,None,None,history,lambda: None,pred,time.monotonic(),
                             state,.11,read_only,trace)
            if fail_second:
                with self.assertRaisesRegex(RuntimeError,"device gone"):
                    run()
            else:
                run()
        trace.save("testing")
        record = json.loads(trace.path.read_text())
        self.assertEqual(record["ticks"][0]["action_index"],0)
        self.assertEqual(record["plans"][0]["actions"],pred.tolist())
        self.assertEqual(record["ticks"][0]["goal_register_ticks"],[1999]*6)
        self.assertEqual(record["ticks"][0]["goal_register"],(state+0.8).tolist())
        self.assertEqual(record["ticks"][0]["raw_target"],(state+1).tolist())
        self.assertEqual(record["ticks"][0]["encoder_ticks"],[2000]*6)
        if read_only:
            robot.send_action.assert_not_called()
            self.assertEqual(record["commands_sent"],0)
            self.assertNotIn("returned_setpoint",record["ticks"][0])
        else:
            self.assertEqual(record["ticks"][0]["returned_setpoint"],(state+0.5).tolist())
            self.assertEqual(record["ticks"][0]["bounded_target"],(state+1).tolist())
            self.assertEqual(record["commands_sent"],1 if fail_second else 3)
            if fail_second:
                self.assertIn("send_started_s",record["ticks"][1])
                self.assertNotIn("send_returned_s",record["ticks"][1])
        return trace.path

    def test_saved_history_and_unique_paths(self):
        with tempfile.TemporaryDirectory() as path:
            a = self.exercise(Path(path))
            b = self.exercise(Path(path),read_only=True)
            self.assertNotEqual(a,b)
            self.assertTrue(a.is_file())

    def test_uncertain_second_dispatch_does_not_inflate_count(self):
        with tempfile.TemporaryDirectory() as path:
            self.exercise(Path(path), fail_second=True)


class ChunkScheduleTests(TestCase):
    def schedule(self, path, stop_camera=None, block=False, bad_index=None, late_worker=False,
                 mode="chunk", lift=7., tracks=False):
        # Physical time advances in deterministic 50 ms increments. No devices are opened.
        now = [100.]
        state = np.array([0.,0.,0.,0.,0.,35.])
        chunk = np.tile(state,(40,1))
        chunk[:,1] += np.linspace(0,lift,40)
        if bad_index is not None:
            chunk[bad_index,1] = 101.
        history = History(maxlen=24)
        def read(*_):
            raw = dict.fromkeys(JOINTS,2048)
            history.append((now[0],0,raw,state.copy()))
            return raw,state
        def frames():
            if stop_camera is not None and now[0]-100 > stop_camera:
                return None
            return {"frame_time":[now[0]-.06]*2,"state":state.copy()},None,None
        robot = mock.Mock()
        def send(action):
            if tracks:
                state[:] = [action[f"{j}.pos"] for j in JOINTS]
            return dict(action)
        robot.send_action.side_effect=send
        pool = mock.Mock()
        def submit(*_):
            submitted = now[0]
            job = mock.Mock()
            job.done.side_effect=lambda: not block and now[0]-submitted >= (.31 if late_worker else .09)
            proposal = chunk.copy()
            if mode == "launch":
                # A new plan aims near the measured posture at the time of the image.
                proposal = np.tile(state.copy(),(40,1))
                proposal[:,1] += np.linspace(0,5,40)
            job.result.return_value = proposal
            return job
        pool.submit.side_effect=submit
        trace = TrialTrace(path,path/"model","simulated-powered")
        self.trace = trace
        with mock.patch("trial.time.monotonic",side_effect=lambda:now[0]), \
             mock.patch("trial.time.sleep",side_effect=lambda duration:now.__setitem__(0,now[0]+duration)), \
             mock.patch("trial.select.select",return_value=([],[],[])), \
             mock.patch("trial.read_state",side_effect=read), mock.patch("trial.check_health"):
            return control_loop(robot, mock.Mock(), pool, history, frames, chunk,99.799,
                                state.copy(),3.,False,trace,action_mode=mode)

    def test_chunk_reaches_late_horizons_and_prefetches(self):
        with tempfile.TemporaryDirectory() as path:
            result = self.schedule(Path(path))
        self.assertEqual(result["commands_sent"],60)
        self.assertEqual(result["plan_switches"],1)
        rows = self.trace.rows
        self.assertGreaterEqual(max(r["action_index"] for r in rows),37)
        self.assertLess(max(r["action_index"] for r in rows),40)
        self.assertLess(max(r["camera_age_ms"] for r in rows),300)
        for a,b in zip(rows,rows[1:]):
            self.assertTrue((abs(np.array(b["bounded_target"])-a["bounded_target"])<=STEP).all())
            if a["plan"] == b["plan"]:
                self.assertIn(b["action_index"]-a["action_index"],(0,1,2))
            else:
                self.assertLessEqual(b["action_index"],7)
                self.assertGreaterEqual(a["action_index"],37)

    def test_launch_handover_requires_two_measured_advances(self):
        with tempfile.TemporaryDirectory() as path:
            result = self.schedule(Path(path),mode="launch",lift=42.,tracks=True)
        self.assertEqual(result["commands_sent"],60)
        i = self.trace.data["launch_handover"]["tick"]
        self.assertGreaterEqual(self.trace.rows[i-1]["shoulder_lift_from_start"],15.)
        self.assertGreaterEqual(self.trace.rows[i]["shoulder_lift_from_start"],15.)
        self.assertTrue(all(r["phase"]=="launch" for r in self.trace.rows[:i]))
        self.assertTrue(all(r["phase"]=="replan" for r in self.trace.rows[i:]))
        self.assertGreater(self.trace.rows[i-1]["action_index"],12)
        self.assertLessEqual(self.trace.rows[i]["action_index"],6)
        self.assertGreater(result["plan_switches"],3)

    def test_launch_stall_and_expiry_without_encoder_progress(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(RuntimeError,"未动"):
                self.schedule(Path(path),mode="launch",lift=22.)
            self.assertNotIn("launch_handover",self.trace.data)
            with self.assertRaisesRegex(TimeoutError,"首个动作块已到期"):
                self.schedule(Path(path),mode="launch",lift=5.)
            self.assertNotIn("launch_handover",self.trace.data)

    def test_launch_keeps_tracking_and_freshness_guards(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(ValueError,"shoulder_lift"):
                self.schedule(Path(path),mode="launch",bad_index=18)
            with self.assertRaisesRegex(TimeoutError,"双摄"):
                self.schedule(Path(path),mode="launch",stop_camera=.1)
            with self.assertRaisesRegex(TimeoutError,"超时|迟到"):
                self.schedule(Path(path),mode="launch",late_worker=True)


    def test_chunk_drops_stale_vision_before_end_of_valid_plan(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(TimeoutError,"双摄"):
                self.schedule(Path(path),stop_camera=.2)
            self.assertLess(self.trace.commands_sent,15)

    def test_chunk_rejects_late_and_unavailable_prefetch(self):
        for kwargs in ({"late_worker":True},{"block":True}):
            with tempfile.TemporaryDirectory() as path:
                with self.assertRaisesRegex(TimeoutError,"超时|用完|迟到"):
                    self.schedule(Path(path),**kwargs)
                self.assertLessEqual(max(r.get("action_index",0) for r in self.trace.rows),39)

    def test_chunk_rejects_illegal_late_target(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(ValueError,"shoulder_lift"):
                self.schedule(Path(path), bad_index=18)
            self.assertEqual(self.trace.rows[-1]["action_index"],18)
            self.assertNotIn("send_started_s",self.trace.rows[-1])


class SessionTests(TestCase):
    def run_session(self, bad_first=False, overload=False, check_only=False):
        state = np.array([0.,0.,0.,0.,0.,40.])
        raw = {j:2048 for j in JOINTS}
        powered = [False]
        robot = mock.Mock()
        dispatch = []
        robot.send_action.side_effect = lambda _action: dispatch.append(time.monotonic())
        bus = robot.bus
        bus.calibration = {j:SimpleNamespace(range_min=800,range_max=3300) for j in JOINTS}
        bus.motors = {j:SimpleNamespace(id=i+1) for i,j in enumerate(JOINTS)}
        bus._normalize.return_value = dict(zip(range(1,7),state))
        bus.enable_torque.side_effect = lambda: powered.__setitem__(0,True)
        def get(reg,*args,**kw):
            if reg=="Torque_Enable": return dict.fromkeys(JOINTS,int(powered[0]))
            return raw
        bus.sync_read.side_effect = get
        def read(reg,j,**kw):
            if reg=="Status": return int(overload and powered[0] and j=="gripper")*32
            if reg=="Torque_Enable": return 0
            return raw[j]
        bus.read.side_effect = read
        pred = np.tile(state,(40,1))
        if bad_first: pred[0,2]=106.8
        predictor=mock.Mock()
        predictor.infer.return_value = pred
        inputs = []
        def pair(views,hist,prev):
            stamp = hist[-1][0]
            inputs.append(stamp)
            return {"state":state,"frame_time":[stamp,stamp]}, None,None
        with tempfile.TemporaryDirectory() as temp:
            lock=Path(temp)/"lock"
            lock.touch()
            camera = mock.Mock()
            obs = SimpleNamespace(CameraFeed=mock.Mock(return_value=camera),LOCK=lock,PORT=lock,
                                  camera_devices=mock.Mock(return_value=["front","wrist"]),sample=pair)
            follower = SimpleNamespace(SO101Follower=mock.Mock(return_value=robot),SO101FollowerConfig=mock.Mock())
            servo = SimpleNamespace(INST_PING=1,INST_READ=2,INST_SYNC_READ=130)
            modules = {"observations":obs, "lerobot.robots.so101_follower.so101_follower":follower,
                       "scservo_sdk.scservo_def":servo}
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules,modules))
                stack.enter_context(mock.patch("trial.subprocess.run",return_value=SimpleNamespace(returncode=1)))
                stack.enter_context(mock.patch("trial.check_settings"))
                stack.enter_context(mock.patch("builtins.input",side_effect=["START",""]))
                stack.enter_context(mock.patch("trial.select.select",return_value=([],[],[])))
                with ExitStack() as io:
                    if bad_first:
                        with self.assertRaisesRegex(ValueError,"elbow_flex"):
                            onsite(None,predictor,.23,io)
                        bus.enable_torque.assert_not_called()
                        bus.sync_write.assert_not_called()
                        robot.send_action.assert_not_called()
                    elif check_only:
                        result=onsite(None,predictor,.23,io,check_only=True)
                        self.assertTrue(result["startup_passed"])
                        bus.sync_write.assert_not_called()
                        bus.enable_torque.assert_not_called()
                        with self.assertRaisesRegex(RuntimeError,"禁止"):
                            bus.packet_handler.txPacket(None,bytes([0,0,0,0,3]))
                    elif overload:
                        with self.assertRaisesRegex(RuntimeError,"gripper"):
                            onsite(None,predictor,.23,io)
                        robot.send_action.assert_not_called()
                        self.assertEqual(bus.disable_torque.call_count,7)
                    else:
                        result=onsite(None,predictor,.23,io)
                        self.assertEqual(result["commands_sent"],5)
                        self.assertEqual(robot.send_action.call_count,5)
                        np.testing.assert_allclose(np.diff(dispatch),.05,rtol=0,atol=.018)
                        bus.enable_torque.assert_called_once()
                        self.assertEqual(bus.disable_torque.call_count,6)

    def test_valid_session(self):
        self.run_session()

    def test_read_only_preview(self):
        self.run_session(check_only=True)

    def test_init_rejects_without_writes(self):
        self.run_session(bad_first=True)

    def test_overload_aborts(self):
        self.run_session(overload=True)


if __name__=="__main__":
    main()
