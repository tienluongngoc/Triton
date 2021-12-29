/usr/src/tensorrt/bin/trtexec  --onnx=iresnet18.onnx --saveEngine=model.plan --explicitBatch --minShapes=input:1x3x112x112 --optShapes=input:4x3x112x112 --maxShapes=input:4x3x112x112 --fp16
