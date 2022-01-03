
#include "types/facelandmark.h"
#include <opencv2/opencv.hpp>
#include <vector>

struct FaceDet {
  cv::Rect box;
  FaceLandmark landmark;
  cv::Mat face_aligned;
  bool facemask;
};
