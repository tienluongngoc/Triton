
// #include "retina_mnet.h"

// #include "alignment.h"
#include "retina_mnet_mask.h"
#include "ExtractFeature.h"
#include "search.h"
#include<bits/stdc++.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#define CLOCKS_PER_SEC  1000000
#define DEVICE 0 


FaceDection obj;
FaceExtraction face_extractor;
SearchFace face_searchor;
clock_t start_d, end_d,start_e,end_e,start_s,end_s;
struct res_final{
    std::string mnv;
    res_det det;

};


std::vector<res_final> identify(cv::Mat img ){
    std::vector<res_final> res;
    std::vector<cv::Mat> img_aligneds;
    std::vector<cv::Mat> features;
    std::vector<bool> facemasks;
    std::string mnv;
    res_final res_cur;


    start_d = clock();
    std::vector<res_det> dets = obj.detect(img);
    


    // std::cout<<"number face "<<dets.size()<<"\n";
    for (int i = 0 ;i<dets.size();i++){
        img_aligneds.push_back(dets[i].img_aligned);
        facemasks.push_back(dets[i].face_mask);
    }
    end_d = clock();

    start_e = clock();
    features=face_extractor.extract(img_aligneds,facemasks);
    end_e = clock(); 

    start_s = clock();
    for (int i = 0 ;i<img_aligneds.size();i++){

        mnv= face_searchor.search(features[i],facemasks[i]);
        res_cur.det=dets[i];
        res_cur.mnv=mnv;
        res.push_back(res_cur);
    }
    end_s = clock();

    float time_detect = ((double) (end_d - start_d)) / CLOCKS_PER_SEC;
    float time_extract = ((double) (end_e - start_e)) / CLOCKS_PER_SEC;\
    float time_search = ((double) (end_s - start_s)) / CLOCKS_PER_SEC;
    std::cout<<"time detect : "<<time_detect<<" with "<<res.size()<<" face\n";
    std::cout<<"time extract : "<<time_extract<<" with "<<res.size()<<" face\n";
    std::cout<<"time search : "<<time_search<<" with "<<res.size()<<" face\n";

    return res;

}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    face_extractor.setup("../weights/feature_4.engine");
    obj.set_up("../weights/retina_mnet_mask.engine");
    face_searchor.setup(); 

    std::vector<res_final> res_ids;
    cv::Mat frame;
    cv::VideoCapture cap;
    std::string mnv,res_text ;
            
   
    // cap.open("rtsp://admin:CUSTRJ@192.168.0.102:554");
    // cap.open("../test.mp4");
    // if (!cap.isOpened()) {
    //     std::cerr << "ERROR! Unable to open camera\n";
    //     return -1;
    // }

    // for (;;)
    // {
    //     cap.read(frame);
    //     if (frame.empty()) {
    //         cerr << "ERROR! blank frame grabbed\n";
    //         break;
    //     }
        
    //     res_ids = identify(frame);
        
    //     for(int i = 0 ; i<res_ids.size();i++){
    //         mnv=res_ids[i].mnv;
    //         if(res_ids[i].det.face_mask)
    //             res_text=mnv+"_facemask";
    //         else
    //             res_text=mnv+"_nonfacemask";
    //         cv::rectangle(frame, res_ids[i].det.boxes, cv::Scalar(0, 255, 0));
    //         cv::putText(frame,res_text,res_ids[i].det.boxes.tl(),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
    //     }
      

        
    //     cv::imshow("Live", frame);
    //     if (cv::waitKey(5) >= 0)
    //         break;
    // }


//test IMG:
    frame = cv::imread("../test_img/test_st.jpg");
    res_ids = identify(frame);
        
    for(int i = 0 ; i<res_ids.size();i++){
        mnv=res_ids[i].mnv;
        if(res_ids[i].det.face_mask)
            res_text=mnv+"_facemask";
        else
            res_text=mnv+"_nonfacemask";
        cv::rectangle(frame, res_ids[i].det.boxes, cv::Scalar(0, 255, 0));
        cv::putText(frame,res_text,res_ids[i].det.boxes.tl(),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
    }
        cv::imshow("Live", frame);
        cv::waitKey(0);
         

    return 0;





}
