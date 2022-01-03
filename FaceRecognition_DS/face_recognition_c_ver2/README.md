# Face_Recognition_C


# install tensorrt server :

        go to https://developer.nvidia.com/tensorrt-getting-started 
        download tensorrt suitable for cuda version (file .deb)
        check: lsb_release -a and nvidia-smi
        eg. nv-tensorrt-repo-ubuntu1804-cuda10.2-trt8.0.0.3-ea-20210423_1-1_amd64.deb
        
        install :
        
                os="ubuntu1804"
                tag="cuda10.2-trt8.0.0.3-ea-20210423"
                sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
                sudo apt-key add /var/nv-tensorrt-repo-${os}-${tag}/7fa2af80.pub
                sudo apt-get update
                sudo apt-get install tensorrt
                sudo apt-get install python3-libnvinfer-dev
        test:
        
                python
                import tensorrt

# convert :
      mkdir weights
      
       Detection_with_mask:
        
            cd include/detection_with_mask
            gdown https://drive.google.com/uc?id=1540V4Jwzuq59iFw0rn6Y2XYJ2fecvDJM
            mkdir build
            cd build
            cmake ..
            make
            
            ./retinafaceAntiCov -s
            mv retinafaceAntiCov.engine../../../weights/retina_mnet_mask.engine
            cd ../../../
    
      
      Detection:(old , dont have facemask)
          
          cd include/detection/
          gdown https://drive.google.com/uc?id=1liYQ7-9Xlm7SOSSPEGpdw1OUlywDoCym
          mkdir build 
          cd build
          cmake ..
          make

          ./convert_mnet -s
          mv retina_mnet.engine ../../../weights/
          cd ../../../
          
     Extraction:
         cd weights
         gdown https://drive.google.com/uc?id=15a2lyH6DmfL8WIP8v5tse3XCG73NhKbA
         python3 onnx_to_trt.py
         
     Antispoofing:
        cd include/antispoofing
        gdown https://drive.google.com/uc?id=1MmNyYy3VXNdRchtQ_flkKLeC1qolaiAC
        mkdir build
        cd build
        cmake ..
        make
        ./yolov5 -s ../best_s.wts ats.engine s
        mv ats.engine ../../../weights
        
        
        
     
# Test :

        mkdir build
        cd build
        cmake ..
        make


        #test face recognition

                ./facerecognition

        #add face :
                ./add_face path_image mnv    # mnv mã nhân viên , ảnh lấy face to nhất

                example ./add_face ../test_img/hao1.jpg hao

                #note

                    it can run even when facerecognition is running

                    face data will update when restart facerecognition

    
    
          
      
          
          
