#include <opencv2/opencv.hpp>
#include<bits/stdc++.h>
using namespace std;
struct FaceInfor{
    std::string mnv;
    cv::Mat feature;

};

class SearchFace{
public:

    std::vector<FaceInfor> list_nv;
    float threshold_distance = 1.1;
    float threshold_distance_facemask = 0.95;
    std::string data_dir = "../data/";
    std::string list_nv_txt = data_dir+"nv.txt";


    void setup(){
        std:cout<<"load \n";
        fstream my_file;
        my_file.open(list_nv_txt, ios::in);
        string ch;
        while (1) {
            my_file >> ch;
            if (my_file.eof() || ch=="")
                break;
            add_face_from_path_feature(ch);

           
        }
        std::cout<<"Load successfully "<<list_nv.size()<<" Employees \n";

    }


    float distance(cv::Mat f1,cv::Mat f2){
        return cv::norm( f1, f2);
    };

    bool add_face_from_path_feature(std::string mnv){
        try{
            float fnum[512];
            float* prob=fnum;
            // cv::Mat feature(512, 1, CV_8UC4);
            // feature = cv::imread(data_dir+mnv+".jpg");

            // std::cout<<mnv<<" "<<feature.at<float>(0,0)<<"\n";
            // // cv::Mat out(512, 1, CV_32FC1, feature.data);
            // cv::Mat out_norm;
            // cv::normalize(feature, out_norm);
            std::cout<<mnv<<"\n";

            ifstream in(data_dir+mnv, ios::in | ios::binary);
            in.read((char *) &fnum,  512 * sizeof(float));
            in.close();
            for(int i = 0 ; i<10;i++)
                std::cout<<prob[i]<<" ";
            std::cout<<"\n";
            cv::Mat out(512, 1, CV_32FC1, prob);
            cv::Mat out_norm;
            cv::normalize(out, out_norm);
           
        
            FaceInfor new_nv ;
            new_nv.mnv=mnv;
            new_nv.feature = out_norm;
            list_nv.push_back(new_nv);
            return true;
        }
        catch(...){
            return false;
        }
    }

    static bool check_nv_facemask(std::string str){
        size_t found = str.find("facemask");
        if (found != string::npos)
            return true;
        return false;
    }

    std::string search(cv::Mat feature,bool face_mask){
      
        std::string res="";
        float min_dist=2.0;
        float dis;
        cv::Mat feature_cur;

        for (int i=0;i<list_nv.size();i++){
            
            if((!face_mask &  !check_nv_facemask(list_nv[i].mnv)) || (face_mask & check_nv_facemask(list_nv[i].mnv))){
                feature_cur = list_nv[i].feature;
               
                dis=distance(feature_cur,feature);
                std::cout<<"dis"<<dis<<"\n";
                if(dis<min_dist){
                    min_dist=dis;
                    
                    if(!face_mask & min_dist<=threshold_distance){
                        res=list_nv[i].mnv;
                    }
                    if(face_mask & min_dist<=threshold_distance_facemask){
                        res=list_nv[i].mnv;
                        
                    }
                    
                }
            }
        }
        
    return res;
        
    }
    vector<std::string>  search(std::vector<cv::Mat> features,std::vector<bool> facemasks){
        vector<std::string> res;
        std::string res_cur;

        for(int i = 0 ;i<features.size();i++){
            res_cur = search(features[i],facemasks[i]);
            res.push_back(res_cur);
        }
        return res;

    }

};