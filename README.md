# Triton
## Run
  
    docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --gpus all --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:21.07-py3

## install opencv
  
    pip3 install opencv-python

    apt-get update

    apt install -y libgl1-mesa-glx
## Dali

    https://github.com/triton-inference-server/dali_backend/tree/main/docs/examples/resnet50_trt

## install nvidia
  
    pip3 install nvidia-pyindex
    
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
  
## install client triton

    pip3 install tritonclient[all]
