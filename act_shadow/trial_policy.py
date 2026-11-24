"""CUDA inference subprocess. Input is a six-joint state and two RGB camera frames; output is one ACT chunk."""
import argparse
import json
import sys
import struct
import numpy as np
import torch
from pathlib import Path
from run_shadow import file_hash, take_image

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    assert manifest["output"]["joint_names"] == [f"{j}.pos" for j in JOINTS]
    assert manifest["output"]["fps"] == 20 and manifest["output"]["shape"] == [1, 40, 6]
    model_file = "act-core.pt"
    if "tensorrt_engine" in manifest:
        from cube46_engines import ENGINES, ENGINE_ROOT, EnginePolicy
        directory, model_file, expected_sha = ENGINES[manifest["tensorrt_engine"]]
        if args.bundle.name != directory or manifest["artifact_sha256"][model_file] != expected_sha:
            raise ValueError("TensorRT 型号与模型目录不一致")
        model_path = ENGINE_ROOT / model_file
    else:
        model_path = args.bundle / model_file
    if file_hash(model_path) != manifest["artifact_sha256"][model_file]:
        raise ValueError(f"Model checksum mismatch for {model_file}")
    for name in ("act-input-0.npz", "act-input-1.npz"):
        if file_hash(args.bundle / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Bundle checksum mismatch for {name}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    if "tensorrt_engine" in manifest:
        model = EnginePolicy(model_path)
    else:
        model = torch.jit.load(str(model_path), map_location="cuda").eval()
    stream = sys.stdin.buffer
    output = sys.stdout.buffer
    with torch.inference_mode():
        for name in ("act-input-0.npz", "act-input-1.npz"):
            with np.load(args.bundle / name) as case:
                inputs = [torch.from_numpy(case[k]).cuda() for k in ("state", "front", "wrist")]
                prediction = model(*inputs).cpu().numpy()
                if prediction.shape != (1, 40, 6) or not np.allclose(
                    prediction, case["ref"], rtol=0, atol=manifest["validation"]["absolute_tolerance"]
                ):
                    raise ValueError(f"Reference mismatch for {name}")
        for _ in range(5):
            model(*inputs)
        torch.cuda.synchronize()
        output.write(b"R")
        output.flush()
        while state_bytes := stream.read(24):
            if len(state_bytes) != 24:
                raise EOFError("Incomplete joint state")
            state = torch.tensor([struct.unpack("<6f", state_bytes)], device="cuda")
            prediction = model(state, take_image(stream), take_image(stream)).cpu().numpy()
            if prediction.shape != (1, 40, 6) or not np.isfinite(prediction).all():
                raise ValueError("Invalid ACT output")
            output.write(prediction.astype("<f4", copy=False).tobytes())
            output.flush()


if __name__ == "__main__":
    main()
