
// #include "retina_mnet.h"

// #include "alignment.h"
#include "retina_mnet_mask.h"
#include "ExtractFeature.h"
#include "face_antispoofing.h"
#include "search.h"
#include <bits/stdc++.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#define CLOCKS_PER_SEC 1000000
#define DEVICE 0

#define max(a, b)                                                              \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

#define min(a, b)                                                              \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a < _b ? _a : _b;                                                         \
  })

FaceDection obj;
FaceExtraction face_extractor;
SearchFace face_searchor;
FaceAntispoofing ats_detection;

cv::Mat img_aligned_st = cv::imread("../cropped.jpg");
clock_t start_d, end_d, start_e, end_e, start_s, end_s;
struct res_final {
  std::string mnv;
  res_det det;
};

float distance1(cv::Mat f1, cv::Mat f2) { return cv::norm(f1, f2); };

// padding percent
cv::Mat crop(cv::Mat image, cv::Rect r,
             float padding) { // img(Range(start_row, end_row), Range(start_col,
                              // end_col))
  cv::imwrite("origin.jpg", image);
  cv::Mat img = image.clone();
  int x1, x2, y1, y2;
  float padding_x = padding * (r.br().x - r.tl().x);
  float padding_y = padding * (r.br().y - r.tl().y);
  int cols = img.cols;
  int rows = img.rows;

  x1 = max((int)r.tl().x - padding_x, 0);
  x2 = min((int)r.br().x + padding_x, cols);
  y1 = max((int)r.tl().y - padding_y, 0);
  y2 = min((int)r.br().y + padding_y, rows);
  return img(cv::Range(y1, y2), cv::Range(x1, x2)); // Slicing to crop the image
}


std::vector<cv::Mat> extractf(cv::Mat img) {
  std::vector<cv::Mat> img_aligneds;
  std::vector<bool> facemasks;

  img_aligneds.push_back(img);
  // cv::imwrite(std::to_string(i)+"aligned.jpg",dets[i].img_aligned);
  facemasks.push_back(false);
  return face_extractor.extract(img_aligneds, facemasks);
}


std::vector<cv::Mat> identify(cv::Mat img) {
  std::vector<res_final> res;
  std::vector<cv::Mat> img_aligneds;
  std::vector<cv::Mat> features;
  std::vector<bool> facemasks;
  std::vector<bool> spoofings;
  std::string mnv;
  cv::Mat img_cropped;
  res_final res_cur;

  start_d = clock();
  // std::vector<res_det> dets = obj.detect(img);

  // std::cout<<"number face "<<dets.size()<<"\n";
  for (int i = 0; i < 1; i++)

  {
    //     img_cropped = crop(img, dets[i].boxes, 2);
    //     cv::imwrite(std::to_string(i) + "cropped.jpg", img_cropped);

    // spoofings.push_back(ats_detection.detect(img_cropped));
    spoofings.push_back(false);

    img_aligneds.push_back(img_aligned_st);
    // cv::imwrite(std::to_string(i)+"aligned.jpg",dets[i].img_aligned);
    facemasks.push_back(false);
  }
  end_d = clock();

  start_e = clock();

  features = face_extractor.extract(img_aligneds, facemasks);
  end_e = clock();

  start_s = clock();
  for (int i = 0; i < img_aligneds.size(); i++) {
    // cv::imwrite(std::to_string(i)+".jpg",img_aligneds[i]);

    mnv = face_searchor.search(features[i], facemasks[i]);
  }
  end_s = clock();

  return features;
}

int main(int argc, char **argv) {

  cudaSetDevice(DEVICE);
  face_extractor.setup("../weights/iresnet124_4.engine");
  // obj.setup("../weights/retina_mnet_mask.engine");
  // ats_detection.setup("../weights/ats.engine");
  face_searchor.setup();

  std::vector<res_final> res_ids;
  cv::Mat frame;
  cv::VideoCapture cap;
  std::string mnv, res_text;
  std::vector<cv::Mat> f1, f2, f3, f4;
  cv::Mat img;
  img = cv::imread("../0_0f5f05d8c8_img_cropped.jpg");
  f1 = extractf(img);
  img = cv::imread("../0_0f581dc398_img_cropped.jpg");
  f2 = extractf(img);
  std::cout << "distance : " << distance1(f1[0], f2[0]);

  // cap.open("rtsp://admin:CUSTRJ@192.168.0.102:554");
  // cap.open("/home/haobk/oneface.mp4");
  // if (!cap.isOpened())
  // {
  //     std::cerr << "ERROR! Unable to open camera\n";
  //     return -1;
  // }

  // for (;;)
  // {
  //     cap.read(frame);
  //     if (frame.empty())
  //     {
  //         cerr << "ERROR! blank frame grabbed\n";
  //         break;
  //     }

  //     res_ids = identify(frame);
  // }

  // for (int i = 0; i < res_ids.size(); i++)
  // {
  //     mnv = res_ids[i].mnv;
  //     if (res_ids[i].det.face_mask)
  //         res_text = mnv + "_facemask";
  //     else
  //         res_text = mnv + "_nonfacemask";
  //     cv::rectangle(frame, res_ids[i].det.boxes, cv::Scalar(0, 255, 0));
  //     cv::putText(frame, res_text, res_ids[i].det.boxes.tl(),
  //     cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0), 2, false);
  // }

  // cv::imshow("Live", frame);
  // if (cv::waitKey(5) >= 0)
  //     break;
  // }

  // test IMG:
  // frame = cv::imread("../test_img/test_st.jpg");
  // res_ids = identify(frame);

  // for(int i = 0 ; i<res_ids.size();i++){
  //     mnv=res_ids[i].mnv;
  //     if(res_ids[i].det.face_mask)
  //         res_text=mnv+"_facemask";
  //     else
  //         res_text=mnv+"_nonfacemask";
  //     std::cout<<"\n"<<res_text<<"\n";
  //     cv::rectangle(frame, res_ids[i].det.boxes, cv::Scalar(0, 255, 0));
  //     cv::putText(frame,res_text,res_ids[i].det.boxes.tl(),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
  // }
  //     cv::imshow("Live", frame);
  //     cv::waitKey(0);

  // return 0;

  // test facemask
  // cv::Mat frame1;
  // std::vector<cv::Mat> img_aligneds,img_aligneds1;
  // std::vector<cv::Mat> features;
  // std::vector<bool> facemasks;

  // frame = cv::imread("../test_img/st_facemask_aligned.jpg");
  // frame1 = cv::imread("../test_img/st_non_facemask_aligned.jpg");

  // img_aligneds.push_back(frame);
  // facemasks.push_back(true);
  // img_aligneds.push_back(frame1);-0.489247 -0.31651 0.49116 -0.169335
  // 0.512596 -0.561965 0.367865 1.10791 1.80836 -1.98765

  // features=face_extractor.extract(img_aligneds,facemasks);

  // std::cout<<distance(features[0],features[1]);

  return 0;
}
//-0.764025 -0.406269 0.132476 -0.137195 0.241286 -0.521208 0.522199
// 0.849268 1.73079 -2.06767