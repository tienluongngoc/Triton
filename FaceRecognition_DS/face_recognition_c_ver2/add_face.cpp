#include "retina_mnet_mask.h"
// #include "alignment.h"
#include "ExtractFeature.h"
#include <bits/stdc++.h>
#include <fstream>

#define DEVICE 0

using namespace std;
FaceDection obj;
FaceExtraction face_extractor;

std::string data_dir = "../data/";
std::string list_nv_txt = data_dir + "nv.txt";

void add_employee(cv::Mat img, std::string mnv)
{
    // try{
    std::vector<cv::Mat> res;
    float *features;

    std::vector<res_det> dets = obj.detect(img);

    if (dets.size() == 0)
    {
        std::cout << "warning !!!!!!!!!!!!!!!";
        std::cout << "have " << dets.size() << " "
                  << "face\n";
        std::cout << "add unsuccessfull\n";
    }

    else
    {
        if (dets.size() > 1)
        {
            std::cout << "warning !!!!!!!!!!!!!!!";
            std::cout << "have " << dets.size() << " "
                      << "face\n";
        }
        std::ofstream outfile;
        cv::Mat img_aligned;
        float s, smax = -1;

        for (int i = 0; i < dets.size(); i++)
        {
            s = (dets[i].boxes.br().x - dets[i].boxes.tl().x) * (dets[i].boxes.br().y - dets[i].boxes.tl().y);
            if (s > smax)
            {
                smax = s;
                img_aligned = dets[i].img_aligned;
            }
        }

        features = face_extractor.extract_file(img_aligned, mnv);
        outfile.open(list_nv_txt, std::ios_base::app); // append instead of overwrite
        float *feature_non_facemask = &features[0];
        float *feature_facemask = &features[512];

        ofstream out(data_dir + mnv, ios::out | ios::binary);
        out.write((char *)feature_non_facemask, 512 * sizeof(float));
        out.close();

        ofstream out1(data_dir + mnv + "_facemask", ios::out | ios::binary);
        out1.write((char *)feature_facemask, 512 * sizeof(float));
        out1.close();

        outfile << mnv << "\n";
        outfile << mnv + "_facemask"
                << "\n";
        std::cout << "add successfull";
    }
}

int main(int argc, char **argv)
{
    cudaSetDevice(DEVICE);

    std::string path_img = argv[1];
    std::string mnv = argv[2];
    std::cout << path_img << " " << mnv;
    face_extractor.setup("../weights/iresnet18_4.engine");
    obj.setup("../weights/retina_mnet_mask.engine");
    cv::Mat img = cv::imread(path_img);
    add_employee(img, mnv);

    return 0;
}
