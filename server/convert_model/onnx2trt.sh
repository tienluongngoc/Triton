/usr/src/tensorrt/bin/trtexec  --onnx=model.onnx --saveEngine=model.plan --explicitBatch --minShapes=input:1x3x122x122 --optShapes=input:4x3x122x122 --maxShapes=input:4x3x122x122 --fp16
