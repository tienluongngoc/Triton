# FaceRecognition_DS

## 1. Install deepstream 6.0
    
    download : https://developer.nvidia.com/deepstream_sdk_v6.0.0_x86_64tbz2
    
    sudo tar -xvf deepstream_sdk_v6.0.0_x86_64.tbz2 -C /
    cd /opt/nvidia/deepstream/deepstream-6.0/
    sudo ./install.sh
    sudo ldconfig
    
 ## 2. Download model Onnx
 
    cd apps/FaceRecognition_DS_Final/models/ 
    gdown https://drive.google.com/uc?id=1hECx9RpLoMp1Rs50vqC6XlWOWxalwAiZ
    gdown https://drive.google.com/uc?id=1DcAS5Lza_-v4dsUOEbYVVwhLqxgH3-pG
    
 ## 3. Convert to tensorrt
 
    Edit file onnx_to_trt.py
    
    Extract feature:
        convert_onnx(input_onnx="iresnet124_4.onnx",engine_file_path="iresnet124_4.engine",force_fp16=True,max_batch_size=4,im_size= (112,112))
    Detection :
        convert_onnx(input_onnx="centerface_4.onnx",engine_file_path="centerface_4.engine",force_fp16=True,max_batch_size=4,im_size= (112,112))
    
    And run python3 onnx_to_trt.py

## 4. Build
    
    mkdir build
    cd build 
    cmake ..
    make 
    
## 5. Run


    Download video test :  gdown https://drive.google.com/uc?id=16IkPvGitbSPy-ygNv0RnBKpJ41P_4DRD    
    Edit source in apps/FaceRecognition_DS_Final/facerecognition_ds_final.cpp
    cd build
    ./apps/FaceRecognition_DS_Final/facerecognition_ds_final
  
## 6. Install kafka
    
    Install node : https://www.youtube.com/watch?v=AlQfpG10vAc&list=PLxoOrmZMsAWxXBF8h_TPqYJNsh3x4GyO4&index=5&ab_channel=SelfTuts
    Install tool : 
        sudo –s 
        cd opt 

        git clone https://github.com/edenhill/librdkafka.git 
        cd librdkafka 
        git reset --hard 7101c2310341ab3f4675fc565f64f0967e135a6a 
        ./configure --enable-ssl 
        make 
        sudo make install 
        sudo cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream/lib/ 
        sudo ldconfig 

        apt-get install libglib2.0 libglib2.0-dev 
        apt-get install  libjansson4  libjansson-dev 
        apt-get install libssl-dev 
        
        pip3 install kafka-python

 
    
    
