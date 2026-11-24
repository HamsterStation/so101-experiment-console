"""Fixed ACT real-robot job and its post-run replay, hosted on this Mac."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import sys

REMOTE = "/home/nvidia/cie6007-edge/act-shadow"
DIRECT = os.environ.get("SO101_DIRECT_ROOT", "/home/trainer/robotlab/so101-direct")
if not re.fullmatch(r"/[A-Za-z0-9_/-]+", DIRECT) or ".." in DIRECT.split("/"):
    raise ValueError("SO101_DIRECT_ROOT must be an absolute Linux path")
DIRECT_MODEL = "smolvla46-4080-direct"
COMPRESSED_ACT = {"cube46-kd": "fp32", "cube46-kd-int8": "int8"}
NEW_ZONES = ("A1", "A2", "B1", "B2", "四点中心")
MODELS = {
    "cube46-kd": ("ACT 剪枝＋蒸馏 · FP32", "cube46_5k.py", "cube46-kd-step500-fp32", NEW_ZONES),
    "cube46-kd-int8": ("ACT 剪枝＋蒸馏＋量化 · INT8", "cube46_5k.py", "cube46-kd-step500-w8a8", NEW_ZONES),
    DIRECT_MODEL: ("SmolVLA 5,000 步 · 4080 USB 直连", "smol46_direct.py", "smolvla46-4080-direct-005000", ("A1", "A2", "B1", "B2", "四点中心")),
    "smolvla46-4080": ("SmolVLA 5,000 步 · 4080 对照", "smol46_5k.py", "smolvla46-4080-diagnostic-005000", ("A1", "A2", "B1", "B2", "四点中心")),
    "smolvla46": ("SmolVLA 5,000 步 · Jetson 本地", "smol46_5k.py", "smolvla46-v03-20261123-step-005000", ("A1", "A2", "B1", "B2", "四点中心")),
    "cube46": ("ACT 5,000 步 · 新四点位", "cube46_5k.py", "cube46-v03-20261123-step-005000", ("A1", "A2", "B1", "B2")),
    "cube49": ("ACT 4,000 步", "cube49_4k.py", "cube49-20261123-step-004000", ("A1", "A2", "A3", "B1", "B2", "B3")),
}
SSH = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
       "-o", "ServerAliveInterval=3", "-o", "ServerAliveCountMax=2", "jetson-lan"]
def command(model):
    if model == DIRECT_MODEL:
        return SSH[:-1] + ["gpu4080", f"cd {DIRECT} && exec ../.venv/bin/python -u smol46_direct.py "
                            "--onsite --console --action-mode chunk --seconds 15"]
    if model == "smolvla46-4080":
        return [sys.executable, "-u", str(Path(__file__).with_name("smol4080_bridge.py")), "--onsite", "--seconds", "15"]
    script = MODELS[model][1]
    if model in COMPRESSED_ACT:
        script += " --compression " + COMPRESSED_ACT[model]
    return SSH + [f"cd {REMOTE} && exec env -u PYTHONPATH PYTHONNOUSERSITE=1 "
                  f"/home/nvidia/miniconda3/envs/lerobot/bin/python -u {script} "
                  "--onsite --console --action-mode chunk --seconds 15"]


PHASES = {
    "checking": "正在加载模型，检查双摄、从臂和首个动作",
    "ready": "预检通过，等待现场确认接管",
    "arming": "正在复核当前位置与首个动作",
    "running": "模型正在控制从臂",
    "stopping": "已请求停止，等待控制端确认",
    "holding": "跟随已停止，请托稳从臂、放稳物体，再卸力",
    "releasing": "正在逐关节确认卸力",
    "released": "六个关节已确认卸力",
}


def save_evidence(folder, trace, camera, model="cube49"):
    host, root = ("gpu4080", DIRECT) if model == DIRECT_MODEL else ("jetson-lan", REMOTE)
    if not re.fullmatch(re.escape(root) + r"/traces/[0-9]{8}T[0-9]{6}Z-onsite-[a-f0-9]{12}\.json", trace):
        raise ValueError("控制端没有返回有效的试跑记录")
    folder.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(SSH[:-1] + [host, "cat " + trace], capture_output=True, check=True, timeout=12)
    data = json.loads(result.stdout)
    (folder / "trace.json").write_bytes(result.stdout)
    if not data.get("commands_sent") or not camera:
        return False
    if not re.fullmatch(re.escape(root + "/" + MODELS[model][2]) + r"/camera-[a-f0-9]{10}", camera):
        raise ValueError("双摄回放目录不匹配")
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        subprocess.run(["rsync", "-rt", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=6",
                        host + ":" + camera + "/", tmp + "/"], check=True, capture_output=True, timeout=60)
        path = Path(tmp)
        times = json.loads((path / "monotonic.json").read_text())
        start = data["loop_start_s"]
        indices = [i for i,t in enumerate(times) if t >= start - .45]
        if not indices:
            raise ValueError("没有本轮动作对应的摄像头画面")
        lines = []
        for i in indices:
            lines.append(f"file '{i:03d}.jpg'")
            if i + 1 < len(times):
                lines.append(f"duration {max(.001, times[i+1]-times[i]):.6f}")
        (path / "frames.txt").write_text("\n".join(lines) + "\n")
        ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-f", "concat", "-i", "frames.txt",
                        "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags",
                        "+faststart", "replay.mp4"], cwd=path, capture_output=True, check=True, timeout=60)
        (path / "replay.mp4").replace(folder / "replay.mp4")
    return True
