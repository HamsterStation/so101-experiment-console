"""Registered TensorRT students for the four-position ACT experiment."""

from pathlib import Path

# Each entry identifies one deployed policy and its own trace/replay directory.
ENGINES = {
    "fp32": ("cube46-kd-step500-fp32", "student.engine",
             "45b7e04c48645e72bf3f1580415d5c5b319fe736960769fa7606271f8a84e0e2"),
    "int8": ("cube46-kd-step500-w8a8", "student-w8a8.engine",
             "d8a3111f4adcfe1ef2a57c4049eab96336fa75f0a5f340205fb0fd3ed3d0d430"),
}
ENGINE_ROOT = Path.home() / "cie6007-edge/act-comparison/cube46-v1"


class EnginePolicy:
    def __init__(self, path):
        import tensorrt as trt
        import torch

        self.log = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.log)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT 无法加载 {path.name}")
        layout = {"state": (1, 6), "front": (1, 3, 480, 640),
                  "wrist": (1, 3, 480, 640), "actions": (1, 40, 6)}
        if self.engine.num_io_tensors != len(layout):
            raise ValueError("ACT 引擎输入输出数量不匹配")
        self.context = self.engine.create_execution_context()
        self.buffers = {}
        for name, shape in layout.items():
            mode = trt.TensorIOMode.OUTPUT if name == "actions" else trt.TensorIOMode.INPUT
            if (tuple(self.engine.get_tensor_shape(name)) != shape
                    or self.engine.get_tensor_dtype(name) != trt.DataType.FLOAT
                    or self.engine.get_tensor_mode(name) != mode):
                raise ValueError(f"ACT 引擎输入输出不匹配 {name}")
            buffer = torch.empty(shape, device="cuda", dtype=torch.float32)
            self.buffers[name] = buffer
            if not self.context.set_tensor_address(name, buffer.data_ptr()):
                raise RuntimeError(f"TensorRT 无法绑定 {name}")

    def __call__(self, state, front, wrist):
        import torch

        for key, value in zip(("state", "front", "wrist"), (state, front, wrist)):
            self.buffers[key].copy_(value)
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT 推理失败")
        return self.buffers["actions"]
