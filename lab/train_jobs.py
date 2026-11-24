"""Local training/reproduction launcher; all GPU work goes to the configured 4080."""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

BASE = Path.home() / "robotlab"
RUNS = BASE / "cie6007-lab/runs"
PY = BASE / ".venv/bin/python"
CACHE = Path.home() / ".cache/huggingface/hub"
SMOL = CACHE / "models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
VLM = CACHE / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
PATH = re.compile(re.escape(str(BASE)) + r"/cie6007-recordings/run-[A-Za-z0-9_-]+/dataset\Z")
def check(test, msg):
    if not test:
        raise ValueError(msg)
def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
def read(path):
    return json.loads(path.read_text())
def dump(path, data, exclusive=False):
    body = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)+"\n"
    if exclusive:
        with path.open("x") as file:
            file.write(body)
        path.chmod(0o444)
    else:
        temp=path.with_name(path.name+".tmp");temp.write_text(body);temp.replace(path)
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()
def tree(files):
    """Hash names and complete contents; filenames in manifest are root-relative."""
    entries=[(key,sha(path)) for key,path in sorted(files)]
    return hashlib.sha256(json.dumps(entries,separators=(",",":")).encode()).hexdigest()
def dataset_at(name):
    check(PATH.fullmatch(name),"请选择已转换到 4080 的数据集。")
    root=Path(name)
    check(str(root.resolve())==name and root.is_dir(),"数据集路径无效。")
    check((root.parent/"recording-check.json").is_file(),"数据集尚未完成转换校验。")
    meta=read(root/"meta/info.json"); checked=read(root.parent/"recording-check.json")
    check(meta["codebase_version"]=="v3.0" and meta["total_frames"]==checked["frames"]>0 and meta["total_episodes"]==checked["episodes"]>0,
          "数据集版本或数量与已校验记录不一致。")
    cameras=sorted(k for k,v in meta["features"].items() if v["dtype"]=="video")
    check(len(cameras)==2 and sorted(checked["cameras"])==cameras and all(meta["features"][k]["shape"]==[6] for k in ("action","observation.state")),"需要双路图像和六维 SO101 数据。")
    files=[]
    for component in ("data","meta","videos"):
        parent=root/component
        check(parent.resolve()==parent and parent.is_dir(),"数据集缺少 "+component)
        subset=[]
        for p in parent.rglob("*"):
            check(not p.is_symlink(),"数据集目录中不能包含符号链接。")
            if p.is_file():subset.append((str(p.relative_to(root)),p))
        check(subset,"空数据目录："+component)
        files.extend(subset)
    return dict(path=name,sha256=tree(files),files=len(files),episodes=checked["episodes"],frames=checked["frames"],cameras=cameras)
def split_plan(data, mode="all", manual=""):
    """IDs are original, zero-based demonstration IDs, never frame indexes."""
    import random
    check(mode in ("all","episode80","manual"),"不支持的划分方式。")
    check(mode=="manual" or not manual,"仅手动划分可指定验证编号。")
    if mode=="all": return None
    n=data["episodes"]
    check(n>=2,"至少需要两条完整演示，才能分出训练和验证。")
    if mode=="manual":
        check(isinstance(manual,str) and re.fullmatch(r"[0-9]+(?:,[0-9]+)*",manual),"验证集编号应为逗号分隔的 0-based 整数。")
        val=[int(v) for v in manual.split(",")]
        check(len(set(val))==len(val) and all(0<=v<n for v in val) and len(val)<n,"验证 ID 越界、重复或没有剩余训练 episode。")
        val.sort()
    else:
        k=max(1,min(n-1,(n+2)//5))
        val=sorted(random.Random(42).sample(range(n),k))
    ids=set(val)
    spec=dict(mode=mode,split_seed=42 if mode=="episode80" else None,
              train_episodes=[i for i in range(n) if i not in ids],validation_episodes=val,
              stats_scope="train episodes only",algorithm="episode-holdout-v1")
    key=hashlib.sha256(json.dumps([data["sha256"],spec],sort_keys=True,separators=(",",":")).encode()).hexdigest()
    spec["root"]=str(RUNS.parent/"splits"/key)
    return spec

def materialize_split(data,spec):
    """Create separate v3.0 datasets using LeRobot's per-episode stats aggregation."""
    import tempfile,fcntl
    from lerobot.datasets.dataset_tools import split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dst=Path(spec["root"]);dst.parent.mkdir(exist_ok=True,parents=True)
    def content(root):
        files=[(str(p.relative_to(root)),p) for p in root.rglob("*") if p.is_file() and p.name!="partition.json" and not any(part.startswith(".") for part in p.relative_to(root).parts)]
        return tree(files)
    with (dst.parent/(dst.name+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if dst.exists():
            c=read(dst/"partition.json")
            check(c["data_sha256"]==data["sha256"] and c["split"]==spec and c["files_sha256"]==content(dst),"已保存的拆分发生变化，请检查后重试。")
            return c
        with tempfile.TemporaryDirectory(dir=dst.parent,prefix=".build-") as temp:
            stage=Path(temp)/"view"
            src=LeRobotDataset("local/so101_recording",root=Path(data["path"]),video_backend="pyav")
            ids={"train":spec["train_episodes"],"validation":spec["validation_episodes"]}
            views=split_dataset(src,ids,output_dir=stage)
            import numpy as np
            import pyarrow.parquet as pq
            # Check that reindexed sequences and per-split action/state statistics
            # actually match the ORIGINAL rows selected, not a global stats cache.
            totals={}; source={}
            for part in sorted((src.root/"data").rglob("*.parquet")):
                tab=pq.read_table(part,columns=["episode_index","frame_index","observation.state","action"]).to_pandas()
                for eid,group in tab.groupby("episode_index",sort=False):
                    source.setdefault(int(eid),[]).append(group)
            import pandas as pd
            for name,view in views.items():
                selected=ids[name]
                groups=[pd.concat(source[i]).sort_values("frame_index") for i in selected]
                expected=pd.concat(groups,ignore_index=True)
                actual=pd.concat([pq.read_table(p,columns=["episode_index","frame_index","observation.state","action"]).to_pandas() for p in sorted((view.root/"data").rglob("*.parquet"))]).sort_values(["episode_index","frame_index"])
                check(view.meta.total_episodes==len(selected) and view.meta.total_frames==len(expected)==len(actual),"划分后帧数或演示数不匹配。")
                check(np.array_equal(actual["episode_index"].to_numpy(),np.repeat(np.arange(len(selected)),[len(g) for g in groups])) and np.array_equal(actual["frame_index"].to_numpy(),expected["frame_index"].to_numpy()),"划分后的 episode 映射不正确。")
                stats=read(view.root/"meta/stats.json")
                for k in ("observation.state","action"):
                    x=np.stack(expected[k]).astype("float64")
                    check(np.array_equal(x,np.stack(actual[k])) and np.allclose(x.mean(0),stats[k]["mean"],rtol=1e-4,atol=1e-5),"帧数据或归一化统计未匹配所选演示。")
                check(all(k in stats for k in data["cameras"]),"缺少分区内的相机统计。")
                totals[name]={"episodes":len(selected),"frames":len(actual)}
            check(sum(v["frames"] for v in totals.values())==data["frames"],"分区遗漏数据。")
            c=dict(split=spec,data_sha256=data["sha256"],files_sha256=content(stage),counts=totals)
            dump(stage/"partition.json",c,True)
            stage.rename(dst)
            return c

def recipe(model,steps,seed,data,split=None):
    check(model in ("act","smolvla") and type(steps) is int and 1<=steps<=200000 and type(seed) is int and 0<=seed<2**31,"无效模型、训练步数或随机种子。")
    args=["--dataset.repo_id=local/so101_recording","--dataset.root="+data["path"],"--dataset.video_backend=pyav","--policy.device=cuda","--policy.push_to_hub=false","--save_checkpoint_to_hub=false","--trackio.enable=false","--env_eval_freq=0","--num_workers=0","--cudnn_deterministic=true","--tf32=false","--log_freq="+str(min(steps,20)),"--save_freq="+str(steps),"--steps="+str(steps),"--seed="+str(seed)]
    if model=="act":
        args+=["--policy.type=act","--batch_size=4"]
        weights=sha(Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    else:
        cams=data["cameras"]
        common=["observation.images.camera1","observation.images.camera2"]
        if cams!=common:
            check(cams==["observation.images.front","observation.images.wrist"],"SmolVLA 需要 camera1/camera2 或 front/wrist 字段。")
            args+=["--rename_map="+json.dumps(dict(zip(cams,common)),separators=(",",":"))]
        args+=["--policy.path="+str(SMOL),"--policy.vlm_model_name="+str(VLM),"--policy.freeze_vision_encoder=true","--policy.train_expert_only=true","--accelerator.mixed_precision=bf16","--batch_size=1",f"--policy.scheduler_warmup_steps={min(1000,max(0,steps//10))}",f"--policy.scheduler_decay_steps={steps}"]
        files=[]
        for key,d in (("policy",SMOL),("vlm",VLM)):
            check(d.is_dir(),"缺少缓存的基础权重。")
            files += [(key+"/"+str(p.relative_to(d)),p) for p in d.rglob("*") if p.is_file()]
        weights=tree(files)
    r=dict(version=1,model=model,steps=steps,seed=seed,training_args=args,initial_weights_sha256=weights)
    if split:
        r["split"]=split
        args[args.index("--dataset.root="+data["path"])]="--dataset.root="+str(Path(split["root"])/"train")
    return r
def software():
    versions=sorted((d.metadata["Name"],d.version) for d in importlib.metadata.distributions() if d.metadata["Name"])
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").is_file() else "nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],text=True).strip()
    return dict(python=sys.version.split()[0],gpu=gpu,packages=versions)
def cancel_stale(path, status):
    if status.get("state")=="running" and status.get("pid"):
        try: os.kill(status["pid"],0)
        except ProcessLookupError:
            status=dict(state="interrupted",error="4080 任务进程已退出，未确认检查点完整。",completed=stamp())
            dump(path/"status.json",status)
    return status
def watch(path):
    pos=0
    while True:
        log=path/"train.log"
        if log.exists():
            with log.open("rb") as f:
                f.seek(pos); data=f.read(64*1024);pos=f.tell()
            if data: sys.stdout.buffer.write(data);sys.stdout.buffer.flush();continue
        status=cancel_stale(path, read(path/"status.json"))
        if status["state"]!="running":
            print("LAB_RESULT "+json.dumps(status,ensure_ascii=False),flush=True)
            return 0 if status["state"]=="completed" else 1
        time.sleep(.5)
def execute(path,lock_fd):
    # The detached worker inherits the flock until its last descriptor closes.
    model_dir=None
    try:
        m=read(path/"manifest.json")
        work=path/"output"
        command=[str(PY.parent/"lerobot-train"),*m["recipe"]["training_args"],"--output_dir="+str(work)]
        env=dict(os.environ,HF_HUB_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",WANDB_MODE="disabled",PYTHONNOUSERSITE="1",PYTHONUNBUFFERED="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",TOKENIZERS_PARALLELISM="false")
        env.pop("PYTHONPATH",None)
        if m["recipe"].get("split"):
            os.environ.update(HF_HUB_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1")
            part=materialize_split(m["dataset"],m["recipe"]["split"])
            dump(path/"partition.json",part,True)
            if m.get("source_id"):
                original=read(RUNS/m["source_id"]/"partition.json")
                check(original==part,"用于复现的训练子集或划分统计发生变化。")
            print("LAB_SPLIT "+json.dumps(part["counts"]),flush=True)
        subprocess.run(command,cwd=BASE,env=env,check=True,pass_fds=(lock_fd,))
        model_dir=(work/"checkpoints/last/pretrained_model").resolve()
        check(model_dir.is_relative_to(work) and (model_dir/"model.safetensors").is_file() and (model_dir/"train_config.json").is_file(),"训练入口退出但没有找到完整检查点。")
        result=dict(state="completed",completed=stamp(),run_id=m["run_id"],checkpoint=str(model_dir),model_sha256=sha(model_dir/"model.safetensors"))
        if m.get("source_id"):
            source=read(RUNS/m["source_id"]/"status.json")
            result.update(source_id=m["source_id"],source_model_sha256=source["model_sha256"],same_checkpoint_bytes=result["model_sha256"]==source["model_sha256"])
        dump(path/"status.json",result)
    except BaseException as exc:
        dump(path/"status.json",dict(state="failed",completed=stamp(),error=str(exc)));raise
def plan_or_launch(request,source_bytes):
    import fcntl
    run_id=request["run_id"];check(ID.fullmatch(run_id),"无效运行编号。")
    path=RUNS/run_id
    if request["mode"]=="watch":
        check(path.is_dir(),"运行不存在。");return watch(path)
    src=None
    if request["mode"]=="reproduce":
        sid=request["source_id"];check(ID.fullmatch(sid) and sid!=run_id,"无效来源运行编号。")
        home=RUNS/sid
        check((home/"status.json").is_file() and read(home/"status.json").get("state")=="completed","仅能复现已成功完成的运行。")
        src=read(home/"manifest.json");data_name=src["dataset"]["path"]
        model,steps,seed=(src["recipe"][k] for k in ("model","steps","seed"))
    else:
        data_name,model,steps,seed=(request[k] for k in ("dataset","model","steps","seed"))
    check(not path.exists(),"运行编号已存在；不会覆盖上次实验。")
    data=dataset_at(data_name)
    if src:
        old=src["recipe"].get("split");mode=old["mode"] if old else "all"
        manual=",".join(str(i) for i in old["validation_episodes"]) if mode=="manual" else ""
    else:
        mode=request.get("split","all");manual=request.get("validation_episodes","")
    r=recipe(model,steps,seed,data,split_plan(data,mode,manual));env=software()
    if src:
        # Execute the archived implementation, even if the Mac UI has been updated.
        source_bytes=(RUNS/src["run_id"]/"runner.py").read_bytes()
        check(hashlib.sha256(source_bytes).hexdigest()==src["runner_sha256"],"原始运行的程序快照校验失败。")
        check(src["dataset"]==data,"数据集内容已改变；已拒绝复现。")
        check(src["recipe"]==r and src["software"]==json.loads(json.dumps(env)),"训练配方、基础权重或软件环境已改变；已拒绝复现。")
    m=dict(run_id=run_id,schema=1,created=stamp(),recipe=r,dataset=data,software=env,source_id=request.get("source_id"),runner_sha256=hashlib.sha256(source_bytes).hexdigest())
    print("LAB_PLAN "+json.dumps(dict(run_id=run_id,source_id=m["source_id"],dataset=data,recipe=r,run_dir=str(path)),ensure_ascii=False),flush=True)
    if request["dry_run"]:return 0
    RUNS.mkdir(parents=True,exist_ok=True)
    with (RUNS.parent/"gpu-training.lock").open("a") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError("4080 正在运行另一个平台训练，请等待完成。") from None
        path.mkdir(mode=0o700)
        try:
            dump(path/"manifest.json",m,True);runner=path/"runner.py";runner.write_bytes(source_bytes);runner.chmod(0o444)
            with (path/"train.log").open("xb") as log:
                worker=subprocess.Popen([str(PY),str(runner),"_execute",str(path),str(lock.fileno())],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),start_new_session=True)
            dump(path/"status.json",dict(state="running",pid=worker.pid,started=stamp(),run_id=run_id))
        except BaseException as exc:
            dump(path/"status.json",dict(state="failed",error=str(exc)));raise
    return watch(path)
def cli():
    parser=argparse.ArgumentParser(description=__doc__);s=parser.add_subparsers(dest="mode",required=True)
    for flag in ("train","reproduce","watch"):
        p=s.add_parser(flag);p.add_argument("--run-id",required=True)
        if flag!="watch":p.add_argument("--dry-run",action="store_true")
        if flag=="train":
            p.add_argument("--split",choices=["all","episode80","manual"],default="all")
            p.add_argument("--validation-episodes",default="")
            p.add_argument("--dataset",required=True);p.add_argument("--model",choices=["act","smolvla"],required=True)
            p.add_argument("--steps",type=int,default=1000);p.add_argument("--seed",type=int,default=42)
        if flag=="reproduce":p.add_argument("--source-id",required=True)
    args=vars(parser.parse_args());check(ID.fullmatch(args["run_id"]),"无效运行编号。")
    payload=base64.urlsafe_b64encode(json.dumps(args).encode()).decode()
    source=Path(__file__).read_bytes()
    # Transmit the fixed worker as source on stdin, never interpolate user fields into shell commands.
    bootstrap="import sys; source=sys.stdin.buffer.read(); g={'__name__':'lab_worker'}; exec(compile(source,'lab_train_jobs','exec'),g); import base64,json; sys.exit(g['plan_or_launch'](json.loads(base64.urlsafe_b64decode(sys.argv[1])),source))"
    cmd='exec "$HOME/robotlab/.venv/bin/python" ' + shlex.join(["-Iu","-c",bootstrap,payload])
    return subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=8","-o","ServerAliveInterval=10","-o","ServerAliveCountMax=3","gpu4080",cmd],input=source).returncode
if __name__=="__main__":
    try:
        if len(sys.argv)==4 and sys.argv[1]=="_execute":execute(Path(sys.argv[2]),int(sys.argv[3]))
        else:sys.exit(cli())
    except (ValueError,OSError,KeyError) as err:
        print(str(err),file=sys.stderr);sys.exit(2)
