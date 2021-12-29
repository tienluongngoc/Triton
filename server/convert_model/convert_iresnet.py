import os
#import cv2
import time
import sys
import logging
import argparse
import torch
import onnx
import backbones
import numpy as np

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)

# Load model
logging.info('Load TORCH model...')
image_size = (112, 112)
network = 'iresnet124'
model_path = 'weights/iresnet124.pth'
ONNX_FILE_PATH = 'weights/iresnet124-dynamic.onnx'

weight = torch.load(model_path)
device = torch.device("cpu")
model = eval("backbones.{}".format(network))(False)
model.load_state_dict(weight)
model.to(device)
model.eval()
# Convert to ONNX
logging.info('Convert to ONNX-runtime with dynamic shape')
dummy_input = torch.randn(1, 3, 112, 112).to(device)
torch.onnx.export(model, 
                  dummy_input, 
                  ONNX_FILE_PATH, 
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                  input_names=['input'],
                  output_names=['output'], 
                  export_params=True, 
                  verbose=True)
# Check for any error
logging.info('Print GRAPH ONNX-runtime')
onnx_model = onnx.load(ONNX_FILE_PATH)
onnx.checker.check_model(onnx_model)


