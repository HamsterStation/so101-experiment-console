"""Run without USB devices; exercise motion gates and per-episode controls."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jetson_recorder as jr
import record_jobs as r


def arm(raw=2050,alarm=0,torque=False,*,name='shoulder_lift',motor_id=2,limits=(800,3200)):
    low, high = limits
    def normalize(readings):
        position = max(low, min(high, readings[motor_id]))
        return {motor_id: 200 * (position - low) / (high - low) - 100}
    b=NS(is_calibrated=True, motors={name:NS(id=motor_id)},
         sync_read=lambda key,normalize: {name: int(torque) if key=='Torque_Enable' else raw},
         read=lambda key,n,normalize: alarm,_normalize=normalize)
    return NS(id='test',calibration={name:NS(range_min=low,range_max=high)},bus=b)


class RecordGates(unittest.TestCase):
    def test_preflight_uses_raw_position_before_saturating_encoder_mapping(self):
        self.assertEqual(jr.startup_check(arm(),arm()),{'shoulder_lift':2050})
        for a,b in [(arm(4909),arm()),(arm(),arm(794)),(arm(alarm=32),arm()),
                    (arm(),arm(torque=True)),(arm(),arm(2250))]:
            with self.assertRaises(RuntimeError): jr.startup_check(a,b)

    def test_elbow_hold_uses_calibrated_endpoint_for_small_readback_drift(self):
        elbow = dict(name='elbow_flex', motor_id=3)
        for leader_raw, follower_raw, hold in [(3003,3036,3033), (3070,3037,3033), (861,810,814)]:
            with self.subTest(follower_raw=follower_raw):
                leader = arm(leader_raw, limits=(861,3070), **elbow)
                follower = arm(follower_raw, limits=(814,3033), **elbow)
                self.assertEqual(jr.startup_check(leader,follower), {'elbow_flex': hold})
        for leader_raw, follower_raw in [(3070,3038), (861,809)]:
            with self.subTest(follower_raw=follower_raw):
                with self.assertRaisesRegex(RuntimeError, 'elbow_flex.*校准范围'):
                    jr.startup_check(arm(leader_raw, limits=(861,3070), **elbow),
                                     arm(follower_raw, limits=(814,3033), **elbow))

    def test_single_turn_boundary_is_strict_even_with_endpoint_tolerance(self):
        for edge, wrapped in [(4095,4096), (0,-1)]:
            with self.subTest(wrapped=wrapped):
                with self.assertRaisesRegex(RuntimeError, 'shoulder_lift.*单圈'):
                    jr.startup_check(arm(wrapped, limits=(0,4095)), arm(edge, limits=(0,4095)))

    def test_pose_mismatch_reports_joint_and_both_normalized_positions(self):
        with self.assertRaisesRegex(RuntimeError, r'shoulder_lift.*主臂 0\.0.*从臂 50\.0'):
            jr.startup_check(arm(2000), arm(2600))

    def test_episode_marker_does_not_apply_to_next_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            op=NS(home=Path(tmp),stopped=lambda:False)
            (op.home/'finish-3').touch()
            self.assertTrue(jr.Events(op,3)['exit_early'])
            self.assertFalse(jr.Events(op,0)['exit_early'])
            self.assertFalse(jr.Events(op,4)['exit_early'])

    def test_expired_heartbeat_stops_following_even_without_stop_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            op=object.__new__(jr.Operator);op.home=Path(tmp);op.last=time.monotonic()-7;op.lost=False
            self.assertTrue(jr.Events(op,1)['exit_early'])
            with self.assertRaises(InterruptedError): op.checkpoint()

    def test_refuse_motion_without_two_independent_cameras(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(jr,'BASE',Path(tmp)), patch.object(jr,'devices',return_value=[]), patch.object(jr,'Operator'):
            with self.assertRaisesRegex(RuntimeError,'没有两只'):
                jr.capture('test',{'front':'/dev/video0','wrist':'/dev/video2'})

    def test_routes_do_not_generate_arbitrary_commands(self):
        d=dict(onsite=True,front='/dev/video0',wrist='/dev/video2')
        for arg in [{'onsite':False},{'front':d['wrist']},{'task':'a\n'}, {'episodes':True},{'wrist':'/etc/passwd'}]:
            with self.assertRaises(ValueError):r.record_command({**d,**arg},'r-auto')
        for bad in ['../bad','a;echo','-option']:
            with self.assertRaises(ValueError):r.record_stop_command(bad)


if __name__ == '__main__': unittest.main()
