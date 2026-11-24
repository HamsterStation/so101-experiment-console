"""Export the two-camera SO101 ACT checkpoint and verify the complete inference path.

Usage: python export_so101_act.py PRETRAINED_MODEL VALIDATION_DATASET NEW_OUTPUT_DIR
The artifact runs on CUDA, with batch size one and a fixed 480x640 RGB image per camera.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


CAMERAS = ("observation.images.camera1", "observation.images.camera2")
JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos",
          "wrist_roll.pos", "gripper.pos"]
TOLERANCE = 0.002  # absolute difference in calibrated LeRobot position units


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


class LiveACT(torch.nn.Module):
    """A trained policy with its input standardization and output unstandardization."""

    def __init__(self, policy, pre_step, post_step):
        super().__init__()
        self.core = policy.model
        input_stats = pre_step.state_dict()
        output_stats = post_step.state_dict()
        self.eps = pre_step.eps
        for label, feature in zip(("state", "front", "wrist"), (OBS_STATE, *CAMERAS), strict=True):
            self.register_buffer(f"{label}_mean", input_stats[f"{feature}.mean"])
            self.register_buffer(f"{label}_std", input_stats[f"{feature}.std"])
        self.register_buffer("action_mean", output_stats["action.mean"])
        self.register_buffer("action_std", output_stats["action.std"])

    def forward(self, state, front, wrist):
        state = (state - self.state_mean) / (self.state_std + self.eps)
        front = (front - self.front_mean) / (self.front_std + self.eps)
        wrist = (wrist - self.wrist_mean) / (self.wrist_std + self.eps)
        chunk = self.core({OBS_STATE: state, OBS_IMAGES: [front, wrist]})[0]
        return chunk * self.action_std + self.action_mean


def check_contract(checkpoint, policy, dataset):
    config = policy.config
    if tuple(config.image_features) != CAMERAS or set(config.input_features) != {OBS_STATE, *CAMERAS}:
        raise ValueError("Unexpected checkpoint input keys or camera order")
    if any(tuple(config.input_features[key].shape) != (3, 480, 640) for key in CAMERAS):
        raise ValueError("Expected 480x640 RGB images")
    if config.chunk_size != 40 or config.n_obs_steps != 1 or dataset.meta.fps != 20:
        raise ValueError("Expected a single observation and 40 actions at 20 Hz")
    for key in (OBS_STATE, "action"):
        if dataset.meta.features[key]["names"] != JOINTS:
            raise ValueError(f"Unexpected axis order for {key}")
    pre = json.loads((checkpoint / "policy_preprocessor.json").read_text())["steps"]
    post = json.loads((checkpoint / "policy_postprocessor.json").read_text())["steps"]
    if [s["registry_name"] for s in pre] != [
        "rename_observations_processor", "to_batch_processor", "device_processor", "normalizer_processor"
    ] or pre[0]["config"]["rename_map"] or pre[2]["config"]["float_dtype"] is not None:
        raise ValueError("Unrecognized preprocessing pipeline")
    if [s["registry_name"] for s in post] != ["unnormalizer_processor", "device_processor"]:
        raise ValueError("Unrecognized postprocessing pipeline")
    for kind in ("STATE", "VISUAL"):
        if pre[-1]["config"]["norm_map"][kind] != "MEAN_STD":
            raise ValueError(f"Unsupported normalization for {kind}")
    if post[0]["config"]["norm_map"]["ACTION"] != "MEAN_STD":
        raise ValueError("Unsupported action unnormalization")


def examples(dataset):
    # Sample distinct parts of each held-out trajectory, including both early and late camera poses.
    for episode in range(dataset.meta.total_episodes):
        info = dataset.meta.episodes[episode]
        start, stop = int(info["dataset_from_index"]), int(info["dataset_to_index"])
        for fraction in (0.2, 0.5, 0.8):
            index = start + round((stop - start - 1) * fraction)
            sample = dataset[index]
            if int(sample["episode_index"]) != episode:
                raise ValueError("Validation sample points at a different episode")
            batch = {k: sample[k].unsqueeze(0) for k in (OBS_STATE, *CAMERAS)}
            if any(batch[k].dtype != torch.uint8 or batch[k].shape != (1, 3, 480, 640) for k in CAMERAS):
                raise ValueError("Expected uint8 RGB from the video decoder")
            inputs = (
                batch[OBS_STATE].float().cuda(),
                *(batch[k].cuda().float().div(255) for k in CAMERAS),
            )
            yield episode, int(sample["frame_index"]), batch, inputs


@torch.inference_mode()
def build(checkpoint, validation, stage):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    dataset = LeRobotDataset("local/so101_recording", root=validation, video_backend="pyav", return_uint8=True)
    policy = ACTPolicy.from_pretrained(str(checkpoint)).cuda().eval()
    check_contract(checkpoint, policy, dataset)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    model = LiveACT(policy, pre.steps[-1], post.steps[0]).cuda().eval()

    path = stage / "act-core.pt"
    first_inputs = next(examples(dataset))[-1]
    traced = torch.jit.trace(model, first_inputs, strict=True, check_trace=False)
    torch.jit.freeze(traced.eval(), optimize_numerics=False).save(str(path))
    del traced
    loaded = torch.jit.load(str(path), map_location="cuda").eval()
    cases = []
    for episode, frame, batch, inputs in examples(dataset):
        reference_batch = _preprocess_dataset_batch(batch, list(CAMERAS), {}, pre)
        reference = post(policy.predict_action_chunk(reference_batch)).cpu()
        output = loaded(*inputs).cpu()
        if reference.shape != (1, 40, 6) or output.shape != reference.shape:
            raise RuntimeError("Expected one 40-step, six-axis action chunk")
        error = (output - reference).abs()
        if not torch.isfinite(error).all() or error.max() > TOLERANCE:
            raise ValueError(f"Export mismatch at episode {episode}, frame {frame}: {error.max().item()}")
        case = dict(episode=episode, frame=frame, max_abs=float(error.max()), mean_abs=float(error.mean()))
        cases.append(case)
        # Two distinct eps/poses for a portable cross-device check.
        if len(cases) in (1, dataset.meta.total_episodes * 3):
            number = int(len(cases) != 1)
            file = stage / f"act-input-{number}.npz"
            np.savez_compressed(file, state=inputs[0].cpu().numpy(), front=inputs[1].cpu().numpy(),
                                wrist=inputs[2].cpu().numpy(), ref=reference.numpy())
            case["fixture"] = file.name

    artifacts = {file.name: sha256(file) for file in sorted(stage.iterdir())}
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_files_sha256": {f.name: sha256(f) for f in sorted(checkpoint.iterdir()) if f.is_file()},
        "artifact_sha256": artifacts,
        "torch": torch.__version__,
        "lerobot": version("lerobot"),
        "device": torch.cuda.get_device_name(),
        "runtime": "TorchScript CUDA float32, fixed batch size 1",
        "inputs_in_order": [
            {"argument": "state", "key": OBS_STATE, "shape": [1, 6], "units": "LeRobot calibrated follower positions"},
            {"argument": "front", "key": CAMERAS[0], "shape": [1, 3, 480, 640], "format": "float32 RGB in [0,1]"},
            {"argument": "wrist", "key": CAMERAS[1], "shape": [1, 3, 480, 640], "format": "float32 RGB in [0,1]"},
        ],
        "output": {"shape": [1, 40, 6], "units": "LeRobot calibrated absolute joint targets",
                   "fps": 20, "joint_names": JOINTS},
        "validation": {"root": str(validation), "episodes": dataset.meta.total_episodes,
                       "cases": cases, "max_abs_error": max(item["max_abs"] for item in cases),
                       "mean_abs_error": sum(item["mean_abs"] for item in cases)/len(cases),
                       "absolute_tolerance": TOLERANCE},
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("EXPORTED", path.name, artifacts[path.name], "cases", len(cases),
          "max_abs_error", manifest["validation"]["max_abs_error"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("output", type=Path, help="New folder, created after parity checks pass")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as partial:
        stage = Path(partial) / "artifact"
        stage.mkdir()
        build(args.checkpoint.resolve(), args.validation.resolve(), stage)
        stage.rename(output)
    print(output, flush=True)


if __name__ == "__main__":
    main()
