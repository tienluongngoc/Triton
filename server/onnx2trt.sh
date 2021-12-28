/usr/src/tensorrt/bin/trtexec  --onnx=model.onnx --saveEngine=model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:4x3x224x224 --fp16
