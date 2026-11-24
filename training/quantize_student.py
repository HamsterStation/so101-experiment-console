from collections import Counter
import json
import os

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from train_student import DATA, OUT, samples, sha

OPS = ('Conv','MatMul','Gemm')


class Calibration(CalibrationDataReader):
    def __init__(self):
        self.used=[]
        self.selected=set()
        for line in (DATA/'dataset_old/meta/episodes.jsonl').read_text().splitlines():
            row=json.loads(line)
            for frac in (.05,.35,.65,.95):
                self.selected.add((row['episode_index'],4*round((row['length']-1)*frac/4)))
        assert len(self.selected)==152
        self.iterator=self.read()

    def read(self):
        for ep,t,state,top,wrist,_,_ in samples('dataset',4):
            if (ep,t) in self.selected:
                self.used.append([ep,t])
                convert=lambda x:np.ascontiguousarray(x.transpose(2,0,1)[None],dtype=np.float32)/255.
                yield dict(state=state[None].copy(),front=convert(top),wrist=convert(wrist))

    def get_next(self):
        return next(self.iterator,None)


def main():
    src=OUT/'student.onnx';dst=OUT/'student-w8a8.onnx'
    cal=Calibration()
    # Fixed shapes let the calibrator attach range observers directly to each tensor.
    shaped=OUT/'student-inferred.onnx'
    onnx.shape_inference.infer_shapes_path(str(src),str(shaped))
    quantize_static(shaped,dst,cal,quant_format=QuantFormat.QDQ,op_types_to_quantize=list(OPS),
        per_channel=True,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,calibration_providers=['CPUExecutionProvider'],
        extra_options={'ActivationSymmetric':True,'WeightSymmetric':True,'MatMulConstBOnly':False})
    assert set(map(tuple,cal.used))==cal.selected
    model=onnx.load(dst)
    onnx.checker.check_model(model)
    producer={o:n.op_type for n in model.graph.node for o in n.output}
    eligible=[n for n in model.graph.node if n.op_type in OPS]
    uncovered=[n.name for n in eligible if any(producer.get(arg)!='DequantizeLinear' for arg in n.input[:2])]
    report=dict(parent_sha256=sha(src),student_checkpoint_sha256=sha(OUT/'best_student/model.safetensors'),
        onnx_sha256=sha(dst),calibration_split='train',calibration_frames=cal.used,eligible=Counter(n.op_type for n in eligible),
        both_inputs_qdq=len(eligible)-len(uncovered),uncovered=uncovered,quantized_ops=list(OPS),
        nodes=Counter(n.op_type for n in model.graph.node),weight_dtype='signed INT8 per-channel, symmetric',
        activation_dtype='signed INT8 per-tensor, symmetric minmax')
    (OUT/'quantization.json').write_text(json.dumps(report,indent=2))
    print('Coverage',report['eligible'],report['both_inputs_qdq'],'uncovered',uncovered,'size',dst.stat().st_size,flush=True)

if __name__=='__main__':main()
