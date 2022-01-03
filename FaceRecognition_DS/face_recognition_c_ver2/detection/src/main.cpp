#include "header/RetinaFace.h"

int main(int argc, char **argv) {
  cv::Mat frame;

  cv::VideoCapture cap;
  int deviceID = 0;        // 0 = open default camera
  int apiID = cv::CAP_ANY; // 0 = autodetect default API
  cap.open(deviceID, apiID);
  if (!cap.isOpened()) {
    cerr << "ERROR! Unable to open camera\n";
    return -1;
  }

  std::string config_file = "../config/config_anti.yaml";
  // std::string folder_name = argv[2];
  RetinaFace RetinaFace(config_file);
  RetinaFace.LoadEngine();
  // RetinaFace.InferenceFolder(folder_name);

  for (;;) {
    cap.read(frame);
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> frames_process;

    if (frame.empty()) {
      cerr << "ERROR! blank frame grabbed\n";
      break;
    }
    frames.push_back(frame);
    frames_process = RetinaFace.Detect_mutil(frames);

    cv::imshow("Live", frames_process[0]);
    if (cv::waitKey(30) >= 0)
      break;
  }

  RetinaFace.Destroy();
  return 0;
}