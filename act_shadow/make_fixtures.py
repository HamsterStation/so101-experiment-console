"""Create independent TensorRT CLI reference outputs for the two console bundles."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from cube46_engines import ENGINES, ENGINE_ROOT

source = Path.home() / "cie6007-edge/act-shadow/cube46-v03-20261123-step-005000"
base = json.loads((source / "manifest.json").read_text())
stage = Path(sys.argv[1])
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

for variant, (directory, artifact, sha) in ENGINES.items():
    engine = ENGINE_ROOT / artifact
    assert digest(engine) == sha
    bundle = stage / directory
    bundle.mkdir(exist_ok=True)
    checksums = {artifact: sha}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i in range(2):
            filename = f"act-input-{i}.npz"
            assert digest(source / filename) == base["artifact_sha256"][filename]
            with np.load(source / filename) as fixture:
                inputs = {k: fixture[k].astype("<f4") for k in ("state", "front", "wrist")}
            for key, value in inputs.items():
                value.tofile(tmp / key)
            names = ",".join(f"{k}:{tmp/k}" for k in inputs)
            output = tmp / "output.json"
            cmd = ["/usr/src/tensorrt/bin/trtexec", f"--loadEngine={engine}",
                   f"--loadInputs={names}", f"--exportOutput={output}",
                   "--duration=0", "--warmUp=0", "--iterations=1"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError(proc.stdout.decode()[-5000:])
            results = json.loads(output.read_text())
            assert len(results) == 1 and results[0]["name"] == "actions"
            ref = np.array(results[0]["values"], dtype="<f4").reshape(1, 40, 6)
            assert np.isfinite(ref).all()
            np.savez_compressed(bundle / filename, **inputs, ref=ref)
            checksums[filename] = digest(bundle / filename)
    manifest = dict(tensorrt_engine=variant, artifact_sha256=checksums,
                    teacher_sha256=base["checkpoint_files_sha256"]["model.safetensors"],
                    distilled_weights_sha256="deb258974a62b82dca686f418bdc9d4d8f6040fd66b63bfde5bb4af4f125f3dc",
                    inputs_in_order=base["inputs_in_order"], output=base["output"],
                    validation=dict(reference="Jetson trtexec, fixed validation frames", absolute_tolerance=.002))
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(directory, "two reference cases", flush=True)
