"""TensorRT 10.3 accepts statically materialized bias, and INT8 data/weight Q/DQ."""
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
root=Path(__file__).resolve().parent
path=root/'student-w8a8.onnx'
model=onnx.load(path)
graph=model.graph
initial={x.name:x for x in graph.initializer}
removed=[]
for n in list(graph.node):
    if n.op_type!='DequantizeLinear' or n.input[0] not in initial or initial[n.input[0]].data_type!=onnx.TensorProto.INT32:continue
    q,s,z=[numpy_helper.to_array(initial[k]) for k in n.input]
    assert q.ndim==1 and np.all(z==0) and s.size in (1,q.size)
    graph.initializer.append(numpy_helper.from_array((q.astype(np.float32)-z.astype(np.float32))*s,n.output[0]))
    removed.append(n.name)
    graph.node.remove(n)
used={i for n in graph.node for i in n.input}
for x in list(graph.initializer):
    if x.name not in used:graph.initializer.remove(x)
assert len(removed)==9
onnx.checker.check_model(model)
tmp=root/'bias-folded.onnx'
onnx.save(model,tmp)
tmp.replace(path)
report=json.loads((root/'quantization.json').read_text())
report['pre_fold_sha256']=report['onnx_sha256']
with path.open('rb') as f:report['onnx_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
report['constant_bias_dq_folded']=removed
report['nodes']['DequantizeLinear']-=len(removed)
(root/'quantization.json').write_text(json.dumps(report,indent=2))
