# SO-101 Experiment Console

这套本地实验台把 SO-101 主从臂、双摄像头、Jetson 和 GPU 训练机连接到工作电脑上的一个网页。操作员可以检查设备、遥操录制、传输和校验演示数据、启动训练、选择已部署的模型进行实机取放，并回看试跑录像。

The workstation serves a local web interface. A Jetson handles cameras and motors. A separate GPU host converts demonstrations and trains policies. The repository includes the ACT baseline, pruned and distilled FP32 student, and INT8 student trial adapters. It also includes experimental SmolVLA runners.

## 打开界面

工作电脑需要 Python 3.10 或以上版本。控制台后端使用 Python 标准库，前端使用原生 JavaScript。从仓库根目录运行：

```sh
python3 lab/server.py
```

在浏览器打开 <http://127.0.0.1:8797>。macOS 也可以双击 `lab/打开实验台.command`。默认端口可通过 `SO101_CONSOLE_PORT` 设置。作业状态和回放保存在 `local-data/`，可用 `SO101_CONSOLE_DATA` 指定其他目录。离线打开时，设备页会报告远端连接状态。

## 三台机器怎样配合

| 位置 | 依赖和目录 | 负责的操作 |
| --- | --- | --- |
| 工作电脑 | Python、OpenSSH、rsync、FFmpeg | 网页、作业状态、操作者确认、回放合成 |
| Jetson Orin Nano | Linux、V4L2、OpenCV、FFmpeg、兼容 LeRobot v2.1 录制接口的环境、Feetech 串口驱动。ACT 推理使用 CUDA PyTorch，学生模型使用 TensorRT | 双摄和关节检查、遥操录制、端侧试跑 |
| GPU 训练机 | Linux 或 WSL、CUDA PyTorch、支持 v2.1 → v3.0 转换的 LeRobot、PyAV。导出和量化还使用 ONNX、ONNX Runtime 和 NumPy | 数据转换、训练、导出和静态量化 |

1. 在工作电脑的 SSH 配置中设置别名 `jetson`、`jetson-lan` 和 `gpu4080`。前两个指向同一块 Jetson，后者指向训练机。控制台使用 SSH 密钥连接。实机路径预设 Jetson 账户的主目录为 `/home/nvidia`，录制环境为 `~/miniconda3/envs/lerobot`，训练机环境为 `~/robotlab/.venv`。Jetson 的系统 Python 还需能够导入推理和相机脚本使用的包。
2. 将 `check_cameras.py` 安装在 Jetson 的 `~/cie6007-edge/checks/`，将 `act_shadow/` 中的脚本安装在 `~/cie6007-edge/act-shadow/`，在 `~/cie6007-edge/lab/` 下创建 `sessions/` 和 `recordings/`。训练任务与录制任务的工作程序会由控制台经 SSH 发送。
3. 在 `lab/jetson_recorder.py` 中分别填入主臂、从臂的 USB 序列号。在 Jetson 上核对 `act_shadow/observations.py` 的从臂串口、顶摄和腕摄身份。可以用 `ls -l /dev/serial/by-id/` 查看串口，用 `python3 check_cameras.py` 读取摄像头型号和序列号。录制界面支持从已识别的设备中指定两个视角。
4. 使用自己设备的 LeRobot 标定文件。控制器会校验从臂标定和已部署模型的 SHA-256。安装自有模型时，需在对应的运行脚本中登记其校验值，并准备该模型的清单和参考输入输出。基线程序为 `act_shadow/cube46_5k.py`，学生模型在 `act_shadow/cube46_engines.py` 中登记。两个 TensorRT 引擎从 Jetson 的 `~/cie6007-edge/act-comparison/cube46-v1/` 读取。旧版 cube49 的自动入盒检测还需要该相机机位对应的空盒参考图 `empty_tray.png`。
5. 在网页完成设备检查，选择双摄视角后开始遥操录制；关闭该次录制后，可将数据传至训练机、转换并启动训练。将验证过的模型安装到 Jetson 后，在试跑页选择模型和起点。启动、停止和卸力由现场操作者确认。

## 源码入口

| 路径 | 内容 |
| --- | --- |
| `lab/server.py`、`lab/static/` | 网页服务、数据集目录、训练作业和试跑控制 |
| `lab/jetson_recorder.py`、`lab/record_jobs.py` | 遥操、双摄采集、分段录制、会话管理 |
| `stage_recording.sh`、`verify_recording.py`、`lab/train_jobs.py` | 数据传输、转换校验、训练与复现 |
| `training/export_so101_act.py`、`training/train_student.py`、`training/quantize_student.py` | ACT 导出、结构化剪枝和蒸馏、INT8 校准 |
| `training/copy-model.sh`、`act_shadow/make_fixtures.py`、`training/edge_compare.py` | 将学生 ONNX 传至 Jetson、为已构建引擎生成参考输出、测量端侧推理延迟 |
| `act_shadow/`、`lab/trial_jobs.py` | 双摄同步、动作块执行、试跑阶段控制和录像回放 |
| `so101-direct/`、`training/worker.py`、`lab/smol4080_bridge.py` | 可选的 SmolVLA GPU 直连及远程推理适配器 |

训练与部署脚本保留实验中的输入形状、目录布局和模型校验约定。复用压缩实验时，在 `training/train_student.py` 指定自己的训练集、教师检查点和输出目录；学生模型导出后在目标 Jetson 构建对应 TensorRT 引擎。仓库发布源码，数据集、模型包及现场录像由使用者自行准备。

可选的 WSL 直连模式使用 `SO101_DIRECT_ROOT` 指定训练机上的绝对路径（默认 `/home/trainer/robotlab/so101-direct`）。该模式还需在 `so101-direct/observations.py` 中填写设备序列号，把支持 DirectShow 的 FFmpeg（命名为 `capture-ffmpeg.exe`）和由 `HostClock.cs` 编译的 `HostClock.exe` 放到 Windows 的 `C:\so101-direct-tools`，并按所用运行脚本准备本地标定、模型包和推理环境。

## 无硬件检查

```sh
python3 -m unittest discover -s lab/tests -p 'test_*.py'
```

这组测试使用模拟关节和操作者输入，覆盖串口与相机前置检查、会话中断、录制分段以及启动和卸力确认。

本仓库的代码以 [MIT License](LICENSE) 发布。上游依赖包括 [LeRobot](https://github.com/huggingface/lerobot)、[ONNX Runtime](https://github.com/microsoft/onnxruntime) 和 [TensorRT](https://github.com/NVIDIA/TensorRT)。
