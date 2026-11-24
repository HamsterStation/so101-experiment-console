"""Run a trained ACT chunk predictor on Jetson. Predictions go only to a JSON report."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
import torch

CAPTURE_PYTHON = Path.home() / "miniconda3/envs/lerobot/bin/python"
SENSORS = Path(__file__).with_name("observations.py")
FRAME_BYTES = 480 * 640 * 3


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def summary(data):
    return {k: float(np.percentile(data, q)) for k, q in
            (("min", 0), ("p50", 50), ("p95", 95), ("max", 100))}


def take_image(pipe):
    data = pipe.read(FRAME_BYTES)
    if len(data) != FRAME_BYTES:
        raise RuntimeError("Observation stream ended during a frame")
    chw = np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3).transpose(2, 0, 1).copy()
    return torch.from_numpy(chw).to(device="cuda", dtype=torch.float32).div_(255).unsqueeze(0)


def load_pair(pipe, first_size):
    size = struct.unpack("<I", first_size)[0]
    obs = json.loads(pipe.read(size))
    state = torch.tensor([obs["state"]], device="cuda", dtype=torch.float32)
    return obs, (state, take_image(pipe), take_image(pipe))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--precision", choices=("fp32", "amp-fp16"), default="fp32")
    args = parser.parse_args()
    if args.count < 2:
        raise ValueError("Need at least two observations")
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    expected = manifest["artifact_sha256"]
    for name in ("act-core.pt", "act-input-0.npz", "act-input-1.npz"):
        if file_hash(args.bundle / name) != expected[name]:
            raise ValueError(f"Export checksum mismatch: {name}")
    report = {"checkpoint": manifest["checkpoint"], "mode": "live read-only shadow", "motor_writes": 0,
              "precision_requested": args.precision, "output_units": "LeRobot normalized joint position (gripper 0..100, other joints -100..100)",
              "cuda": torch.cuda.get_device_name(), "torch": torch.__version__,
              "model_sha256": expected["act-core.pt"]}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    model = torch.jit.load(str(args.bundle/"act-core.pt"), map_location="cuda").eval()
    parity = []
    # Preserve the trained float32 artifact. Mixed precision is a separate parity test.
    with torch.inference_mode(), torch.autocast(
        "cuda", torch.float16, enabled=args.precision == "amp-fp16"
    ):
        cases = sorted(args.bundle.glob("act-input-*.npz"))
        if len(cases) < 2:
            raise RuntimeError("Need at least two 4080 reference cases")
        for case in cases:
            with np.load(case, allow_pickle=False) as data:
                inputs = [torch.from_numpy(data[k]).cuda() for k in ("state", "front", "wrist")]
                reference = data["ref"]
            prediction = model(*inputs).cpu().numpy()
            if prediction.shape != reference.shape or prediction.shape != (1, 40, 6):
                raise RuntimeError(f"Unexpected shape for {case.name}: {prediction.shape} vs {reference.shape}")
            error = np.abs(prediction - reference)
            if not np.isfinite(error).all():
                raise ValueError(f"Invalid model output in {case.name}")
            parity.append({"case": case.name, "max_abs": float(error.max()), "mean_abs": float(error.mean())})
            tolerance = manifest["validation"]["absolute_tolerance"] if args.precision == "fp32" else 0.5
            if error.max() > tolerance:
                raise ValueError(f"Jetson prediction diverged from the 4080 reference: {parity[-1]}")
        report["reference_max_abs_tolerance"] = tolerance
        report["validation_reference"] = parity
        for _ in range(5):
            model(*inputs)
        torch.cuda.synchronize()

        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        helper = subprocess.Popen([str(CAPTURE_PYTHON), str(SENSORS), "--serve", "--count", str(args.count)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment)
        samples = []
        try:
            for index in range(args.count):
                request_time = time.monotonic()
                helper.stdin.write(b"N")
                helper.stdin.flush()
                size = helper.stdout.read(4)
                if len(size) != 4:
                    raise RuntimeError("Read-only sensor helper exited before a complete observation")
                obs, inputs = load_pair(helper.stdout, size)
                torch.cuda.synchronize()
                before = time.monotonic()
                chunk = model(*inputs)
                torch.cuda.synchronize()
                finished = time.monotonic()
                chunk = chunk.cpu().numpy()
                if chunk.shape != (1, 40, 6) or not np.isfinite(chunk).all():
                    raise ValueError("Unexpected or non-finite model output")
                chunk = chunk[0]
                first_delta = (chunk[0] - np.array(obs["state"])).tolist()
                low = np.array([-100.] * 5 + [0.])
                high = np.full(6, 100.)
                inside = ((chunk >= low) & (chunk <= high)).all(axis=1)
                start_delta = np.array([8., 12., 12., 8., 8., 8.])
                near_start = (np.abs(chunk - np.array(obs["state"])) <= start_delta).all(axis=1)
                obs.update(first_step_inside=bool(inside[0]), first_eight_inside=bool(inside[:8].all()),
                           whole_chunk_inside=bool(inside.all()),
                           first_step_passes_start_guard=bool((inside & near_start)[0]),
                           first_eight_pass_start_guard=bool((inside & near_start)[:8].all()),
                           first_action=chunk[0].tolist(), first_delta=first_delta,
                           chunk_min=chunk.min(axis=0).tolist(), chunk_max=chunk.max(axis=0).tolist(),
                           first_outside_joint_limits=((chunk[0] < low) | (chunk[0] > high)).tolist(),
                           outside_joint_limits_per_joint=((chunk < low) | (chunk > high)).sum(axis=0).tolist(),
                           core_ms=(finished-before)*1000, request_to_output_ms=(finished-request_time)*1000,
                           image_to_output_ms=(finished-min(obs["frame_time"]))*1000, finished_time=finished)
                samples.append(obs)
            helper.stdin.close()
            if helper.wait(timeout=20) != 0:
                raise RuntimeError("Sensor helper reported a failure")
        finally:
            if not helper.stdin.closed:
                helper.stdin.close()
            helper.stdout.close()
            try:
                helper.wait(timeout=10)
            except subprocess.TimeoutExpired:
                helper.terminate()
                helper.wait(timeout=5)
    report["observations"] = len(samples)
    report["output_shape"] = [1, 40, 6]
    report["sampled_start_unix_approx"] = time.time() - (time.monotonic() - samples[0]["state_time"])
    report["sampled_pose"] = samples[0]["state"]
    report["joint_names"] = samples[0]["joints"]
    report["device_nodes"] = samples[0]["devices"]
    report["device_roles"] = ["front", "wrist"]
    report["calibration_sha256"] = samples[0]["calibration_sha256"]
    report["timing_scope"] = "1 prediction per fresh pair; V4L2 driver timestamps, software-paired; live pose remains unpowered"
    report["timing_ms"] = {k: summary([sample[k] for sample in samples])
                           for k in ("pair_skew_ms", "state_skew_ms", "state_read_ms", "core_ms",
                                     "image_to_output_ms")}
    report["effective_observations_per_s"] = ((len(samples)-1) /
                                               (samples[-1]["finished_time"]-samples[0]["finished_time"]))
    report["largest_abs_first_action_minus_state"] = np.abs([s["first_delta"] for s in samples]).max(axis=0).tolist()
    report["legal_counts"] = {
        key: sum(s[key] for s in samples)
        for key in ("first_step_inside", "first_eight_inside", "whole_chunk_inside",
                    "first_step_passes_start_guard", "first_eight_pass_start_guard")
    }
    report["first_outside_joint_limits_counts"] = np.sum([s["first_outside_joint_limits"] for s in samples],axis=0).tolist()
    report["chunk_outside_joint_limits_counts"] = np.sum([s["outside_joint_limits_per_joint"] for s in samples],axis=0).tolist()
    report["last_read_opcodes"] = samples[-1]["read_opcodes"]
    report["samples"] = samples
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.report)
    print(json.dumps({k:v for k,v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
