#include<bits/stdc++.h>
#include "retina_mnet_mask.h"

FaceDection detector;


int main(){
    std::string facemask;
    detector.set_up("../weights/retinafaceAntiCov.engine");
    cv::Mat img;
    img = cv::imread("../test_images/b.png");
    std::vector<res_det> dets=detector.detect(img);

    std::cout<<"number face : "<<dets.size()<<"\n";
    for(int i=0 ;i<dets.size();i++){
        cv::imwrite("aligned_"+std::to_string(i)+".jpg",dets[i].img_aligned);
        facemask= dets[i].face_mask? "face_mask":"non_face_mask";
        std::cout<<i<<" "<<facemask<<"\n";

    }


}