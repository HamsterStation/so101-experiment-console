"""Run the existing supervised trial with the November 23 cube49 checkpoint."""
from collections import deque
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np
import observations

import trial


BUNDLE = Path(__file__).resolve().parent / "cube49-20261123-step-004000"
TRAINED_WEIGHTS = "c5a362b1d601d5eccdc91735957ce99788cd8f64c5c60365228bdcc0506acc04"
EXPORTED_WEIGHTS = "f3f60196aae39437cf60a7c9d1bb918ce58f3efac0b3500929709889c6d77db7"
# Recorded before the five cube49 sessions. The ranges also reconstruct integer
# encoder ticks from all six state channels across all 7,504 training rows.
FOLLOWER_CALIBRATION = "533e7f23619e878daa61911fbb4782a43763ee159dd35b9ecab3cd18537e2b53"


if __name__ == "__main__":
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    if (manifest["checkpoint_files_sha256"]["model.safetensors"] != TRAINED_WEIGHTS or
            manifest["artifact_sha256"]["act-core.pt"] != EXPORTED_WEIGHTS):
        raise RuntimeError("模型文件与 cube49 的 4,000 步检查点不一致")
    trial.CALIBRATION_SHA = FOLLOWER_CALIBRATION
    sys.argv.insert(1, str(BUNDLE))
    if "--onsite" not in sys.argv:
        trial.main(stepwise=True)
    else:
        # Keep references to pairs already read by the controller. Encode only
        # after the trial has left its motor/serial cleanup block.
        pairs = deque(maxlen=160)
        sample_pair = observations.sample

        def observe(*args):
            pair = sample_pair(*args)
            if pair is not None:
                header, front, wrist = pair
                stamp = min(header["frame_time"])
                if not pairs or stamp - pairs[-1][0] >= .19:
                    pairs.append((stamp, front, wrist))
            return pair

        from cube_goal import GoalMonitor
        goal = GoalMonitor()
        observations.sample = observe
        try:
            trial.main(stepwise=True, goal=goal)
        finally:
            goal.close()
            observations.sample = sample_pair
            if pairs:
                directory = BUNDLE / ("camera-" + uuid.uuid4().hex[:10])
                directory.mkdir()
                times = []
                for index, (stamp, front, wrist) in enumerate(pairs):
                    rgb = np.concatenate((front, wrist), axis=1)
                    if not cv2.imwrite(str(directory / f"{index:03}.jpg"),
                                       cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                        raise OSError("双摄回放保存失败")
                    times.append(stamp)
                (directory / "monotonic.json").write_text(json.dumps(times))
                print("现场双摄保存到", directory, flush=True)
