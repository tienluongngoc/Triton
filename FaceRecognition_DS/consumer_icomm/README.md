# face_tx2



convert extract feature:
    
    cd convert/convert_onnx_tensorrt
    
    download 
    
        link :
        
            ires18_onnx : gdown https://drive.google.com/uc?id=1lwhwBPT4nrT4FiPNwLq2sLR2n6z1F1Sb
            ires34_onnx : gdown https://drive.google.com/uc?id=1Vflq7pLqh6cdXrr--NjuxPyIDm4vaCMA
            ires50_onnx : gdown https://drive.google.com/uc?id=1s6IEJg1uQkHFV5N9yTylPxNI8wJHFOQa
            ires100_onnx : gdown https://drive.google.com/uc?id=1Ii7lhTpF2jA19ut4_ZqwE_aOFG6JVzGv
            ires124 onnx : gdown https://drive.google.com/uc?id=1Ifn-znzYnfdSmiGmS5STnr0f4Q_9yTWQ

    
    modify path model and max_batch_size in onnx_to_trt.py
    
    move model to folder extract feature : 
        
        mkdir  *.engine ../../face_features/weights
            
        mv *.engine   ../../face_features/weights
     
     modify file face_features/arcface.py at model_file in ArcFace(model_file...)

convert face detection
    
    cd convert/Pytorch_Retinaface-master
    
    python3 detect.py --save_model
    
    python3 genwts.py
    
    mv retinaface.wts ../../face_detection/
    
    cd ../../face_detection/
    
    mkdir build
    
    cd build && cmake ..
    
    make
    
    sudo ./retina_mnet -s
    
    mkdir ../weights
    
    mv retina_mnet.engine ../weights/
    
Pycuda:
   
    export PATH=$PATH:/usr/local/cuda-10.2/bin
    export CPATH=$CPATH:/usr/local/cuda-10.2/include
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
    
    pip3 install pycuda --user
    
 NMS:
 
       cd face_detection/nms/
       sh setup.sh
    
 
    
    
Test :

       python3 test_face.py
       
