"""Run the existing supervised trial with the cube46 v03 checkpoint."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np
import observations

import trial


BUNDLE = Path(__file__).resolve().parent / "cube46-v03-20261123-step-005000"
TRAINED_WEIGHTS = "2ded4e80934293e90ab6d1e40191280bda84e19c22a0bf7e0258f179d5051dd9"
EXPORTED_WEIGHTS = "f11956be9ff3539cdb3c31fb6211354e6d4569bb47d03612677e931265a855c4"
# Follower calibration for the current dual-camera capture setup.
FOLLOWER_CALIBRATION = "533e7f23619e878daa61911fbb4782a43763ee159dd35b9ecab3cd18537e2b53"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--compression", choices=("fp32", "int8"))
    selection, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    if selection.compression:
        from cube46_engines import ENGINES
        directory, artifact, checksum = ENGINES[selection.compression]
        BUNDLE = BUNDLE.parent / directory
        manifest = json.loads((BUNDLE / "manifest.json").read_text())
        if manifest["artifact_sha256"][artifact] != checksum:
            raise RuntimeError("ACT 压缩引擎与所选型号不一致")
        print(f"ACT {selection.compression.upper()} · {artifact} · SHA256 {checksum}", flush=True)
    else:
        manifest = json.loads((BUNDLE / "manifest.json").read_text())
        if (manifest["checkpoint_files_sha256"]["model.safetensors"] != TRAINED_WEIGHTS or
                manifest["artifact_sha256"]["act-core.pt"] != EXPORTED_WEIGHTS):
            raise RuntimeError("模型文件与 cube46 的 5,000 步检查点不一致")
    trial.CALIBRATION_SHA = FOLLOWER_CALIBRATION
    sys.argv.insert(1, str(BUNDLE))
    if "--onsite" not in sys.argv:
        trial.main(stepwise=True, align_inputs=bool(selection.compression))
    else:
        # Keep references to pairs already read by the controller. Encode only
        # after the trial has left its motor/serial cleanup block.
        pairs = deque(maxlen=160)
        sample_pair = observations.sample

        def observe(*args, **kwargs):
            pair = sample_pair(*args, **kwargs)
            if pair is not None:
                header, front, wrist = pair
                stamp = min(header["frame_time"])
                if not pairs or stamp - pairs[-1][0] >= .19:
                    pairs.append((stamp, front, wrist))
            return pair

        observations.sample = observe
        try:
            trial.main(stepwise=True, align_inputs=bool(selection.compression))
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
