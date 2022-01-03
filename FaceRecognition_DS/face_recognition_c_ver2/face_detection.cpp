#include "retina_mnet_mask.h"
#include "include/headpose_estimation/pose_estimate.h"
#include <bits/stdc++.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define max(a, b)                   \
    (                               \
        {                           \
            __typeof__(a) _a = (a); \
            __typeof__(b) _b = (b); \
            _a > _b ? _a : _b;      \
        })

#define min(a, b)                   \
    (                               \
        {                           \
            __typeof__(a) _a = (a); \
            __typeof__(b) _b = (b); \
            _a < _b ? _a : _b;      \
        })

FaceDection face_detector;
Estimator* face_estimator = new Estimator();

int main(int argc, char **argv)
{

    cudaSetDevice(DEVICE);
    face_detector.setup("../weights/retina_mnet_mask.engine");

    cv::Mat frame;
    cv::VideoCapture cap;
    std::string mnv, res_text;
    // cap.open("rtsp://admin:CUSTRJ@192.168.0.102:554");
    cap.open("/home/haobk/oneface.avi");
    cap.read(frame);
    face_estimator->init(frame.cols,frame.rows);
    
    if (!cap.isOpened())
    {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    for (;;)
    {
        cap.read(frame);
        if (frame.empty())
        {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        std::vector<res_det> dets = face_detector.detect(frame);

        for (int i = 0; i < dets.size(); i++)
        {
            mnv = "";
            if (dets[i].face_mask)
                res_text = mnv + "_facemask";
            else
                res_text = mnv + "_nonfacemask";
            cv::rectangle(frame, dets[i].boxes, cv::Scalar(0, 255, 0));
            cv::putText(frame, res_text, dets[i].boxes.tl(),
                        cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0), 2, false);
        }

        cv::imshow("Live", frame);
        if (cv::waitKey(5) >= 0)
            break;
    }
}
