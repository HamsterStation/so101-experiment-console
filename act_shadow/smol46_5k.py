"""Run the existing supervised trial with the full SmolVLA cube46 checkpoint."""
from collections import deque
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np
import observations

import smol46_trial as trial


BUNDLE = Path(__file__).resolve().parent / "smolvla46-v03-20261123-step-005000"
TRAINED_WEIGHTS = "ce5ccdafd87f6ee5087f7a6a1e1848af6c12e3634fcefd912f2c6993bf4485c5"
EXPORTED_WEIGHTS = "41f6fde9eb1eafcd7ce2afadfc7c991c5c6ad13780670052893595287d92a7c3"
# Follower calibration for the current dual-camera capture setup.
FOLLOWER_CALIBRATION = "533e7f23619e878daa61911fbb4782a43763ee159dd35b9ecab3cd18537e2b53"


class SmolPredictor(trial.Predictor):
    worker_script = "smolvla_policy.py"
    # The full model takes ~0.66 s on Orin. Overlap prediction with the current
    # 2 s plan. The live dual-camera watchdog remains independent at 0.30 s.
    inference_deadline = .95
    prefetch_age = .90
    switch_deadline = 1.20

if __name__ == "__main__":
    remote = "--gpu4080" in sys.argv
    if remote:
        from smol46_remote import RemotePredictor, transport_test
        sys.argv.remove("--gpu4080")
        BUNDLE = BUNDLE.with_name("smolvla46-4080-diagnostic-005000")
        if "--transport" in sys.argv:
            transport_test(BUNDLE)
            sys.exit(0)
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    if (manifest["checkpoint_sha256"] != TRAINED_WEIGHTS or
            manifest["artifact_sha256"]["smolvla-core.pt"] != EXPORTED_WEIGHTS):
        raise RuntimeError("模型文件与 cube46 的 5,000 步检查点不一致")
    if not {"--onsite", "--check-start", "--measure-loop"}.intersection(sys.argv):
        import os
        os.execv("/usr/bin/python3", ["/usr/bin/python3", str(Path(__file__).with_name("smolvla_policy.py")), str(BUNDLE), "--benchmark"])
    observations.sample = trial.aligned_sample
    trial.Predictor = RemotePredictor if remote else SmolPredictor
    trial.CALIBRATION_SHA = FOLLOWER_CALIBRATION
    if "--action-mode" not in sys.argv:
        sys.argv.extend(["--action-mode", "chunk"])
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

        observations.sample = observe
        try:
            trial.main(stepwise=True)
        finally:
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
