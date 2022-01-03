#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <bits/stdc++.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <unordered_map>
#define CLOCKS_PER_SEC 1000000
clock_t start_d, end_d;
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
int mask_instance_width = 2;
int mask_instance_height = 5;
int outSize[] = {252000};
int cols = 640; // size origin
int rows = 640;
bool gen_anchor = false;

struct FaceBox {
  float x;
  float y;
  float w;
  float h;
};

struct FaceRes {
  float confidence;
  FaceBox face_box;
  std::vector<cv::Point2f> keypoints;
  bool has_mask = false;
};

int BATCH_SIZE = 1;
int INPUT_CHANNEL = 3;
int IMAGE_WIDTH = 640;
int IMAGE_HEIGHT = 640;
float obj_threshold = 0.45;
float nms_threshold = 0.45;
bool detect_mask = false;
float mask_thresh = 0.1;
float landmark_std = 1;
int anchor_num = 2;
int bbox_head = 3;
int landmark_head = 10;

int sum_of_feature;
std::vector<std::vector<int>> feature_maps;
std::vector<std::vector<float>> anchors;
std::vector<std::vector<int>> anchor_sizes = {{512, 256}, {128, 64}, {32, 16}};
std::vector<int> feature_sizes;
std::vector<int> feature_steps = {32, 16, 8};

void GenerateAnchors() {
  float base_cx = 7.5;
  float base_cy = 7.5;

  int line = 0;
  for (size_t feature_map = 0; feature_map < feature_maps.size();
       feature_map++) {
    for (int height = 0; height < feature_maps[feature_map][0]; ++height) {
      for (int width = 0; width < feature_maps[feature_map][1]; ++width) {
        for (int anchor = 0; anchor < anchor_sizes[feature_map].size();
             ++anchor) {

          std::vector<float> anchor_;

          anchor_.push_back(base_cx +
                            (float)width * feature_steps[feature_map]);
          anchor_.push_back(base_cy +
                            (float)height * feature_steps[feature_map]);
          anchor_.push_back(anchor_sizes[feature_map][anchor]);
          anchors.push_back(anchor_);
        }
      }
    }
  }
}

float IOUCalculate(const FaceBox &det_a, const FaceBox &det_b) {
  cv::Point2f center_a(det_a.x + det_a.w / 2, det_a.y + det_a.h / 2);
  cv::Point2f center_b(det_b.x + det_b.w / 2, det_b.y + det_b.h / 2);
  cv::Point2f left_up(std::min(det_a.x, det_b.x), std::min(det_a.y, det_b.y));
  cv::Point2f right_down(std::max(det_a.x + det_a.w, det_b.x + det_b.w),
                         std::max(det_a.y + det_a.h, det_b.y + det_b.h));
  float distance_d = (center_a - center_b).x * (center_a - center_b).x +
                     (center_a - center_b).y * (center_a - center_b).y;
  float distance_c = (left_up - right_down).x * (left_up - right_down).x +
                     (left_up - right_down).y * (left_up - right_down).y;
  float inter_l = det_a.x > det_b.x ? det_a.x : det_b.x;
  float inter_t = det_a.y > det_b.y ? det_a.y : det_b.y;
  float inter_r = det_a.x + det_a.w < det_b.x + det_b.w ? det_a.x + det_a.w
                                                        : det_b.x + det_b.w;
  float inter_b = det_a.y + det_a.h < det_b.y + det_b.h ? det_a.y + det_a.h
                                                        : det_b.y + det_b.h;
  if (inter_b < inter_t || inter_r < inter_l)
    return 0;
  float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
  float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
  if (union_area == 0)
    return 0;
  else
    return inter_area / union_area - distance_d / distance_c;
}

void NmsDetect(std::vector<FaceRes> &detections) {
  sort(detections.begin(), detections.end(),
       [=](const FaceRes &left, const FaceRes &right) {
         return left.confidence > right.confidence;
       });

  for (int i = 0; i < (int)detections.size(); i++)
    for (int j = i + 1; j < (int)detections.size(); j++) {
      float iou = IOUCalculate(detections[i].face_box, detections[j].face_box);
      if (iou > nms_threshold)
        detections[j].confidence = 0;
    }

  detections.erase(
      std::remove_if(detections.begin(), detections.end(),
                     [](const FaceRes &det) { return det.confidence == 0; }),
      detections.end());
}

std::vector<std::vector<FaceRes>> postProcess(float *out, const int &outSize,
                                              int cols, int rows) {
  std::vector<std::vector<FaceRes>> vec_result;
  std::vector<FaceRes> result;
  float ratio =
      float(cols) / float(IMAGE_WIDTH) > float(rows) / float(IMAGE_HEIGHT)
          ? float(cols) / float(IMAGE_WIDTH)
          : float(rows) / float(IMAGE_HEIGHT);
  int result_cols = (detect_mask ? 3 : 2) + bbox_head + landmark_head;
  // cv::Mat result_matrix = cv::Mat(sum_of_feature, result_cols, CV_32FC1,
  // out);

  for (int item = 0; item < sum_of_feature * result_cols; item += result_cols) {

    auto *current_row = out + item;
    if (current_row[0] > obj_threshold) {
      FaceRes headbox;
      headbox.confidence = current_row[0];
      std::vector<float> anchor = anchors[item / result_cols];
      auto *bbox = current_row + 1;
      auto *keyp = current_row + 2 + bbox_head;
      auto *mask = current_row + 2 + bbox_head + landmark_head;

      headbox.face_box.x = (anchor[0] + bbox[0] * anchor[2]) * ratio;
      headbox.face_box.y = (anchor[1] + bbox[1] * anchor[2]) * ratio;
      headbox.face_box.w = anchor[2] * exp(bbox[2]) * ratio;
      headbox.face_box.h = anchor[2] * exp(bbox[3]) * ratio;

      headbox.keypoints = {
          cv::Point2f((anchor[0] + keyp[0] * anchor[2] * landmark_std) * ratio,
                      (anchor[1] + keyp[1] * anchor[2] * landmark_std) * ratio),
          cv::Point2f((anchor[0] + keyp[2] * anchor[2] * landmark_std) * ratio,
                      (anchor[1] + keyp[3] * anchor[2] * landmark_std) * ratio),
          cv::Point2f((anchor[0] + keyp[4] * anchor[2] * landmark_std) * ratio,
                      (anchor[1] + keyp[5] * anchor[2] * landmark_std) * ratio),
          cv::Point2f((anchor[0] + keyp[6] * anchor[2] * landmark_std) * ratio,
                      (anchor[1] + keyp[7] * anchor[2] * landmark_std) * ratio),
          cv::Point2f((anchor[0] + keyp[8] * anchor[2] * landmark_std) * ratio,
                      (anchor[1] + keyp[9] * anchor[2] * landmark_std) *
                          ratio)};

      if (detect_mask and mask[0] > mask_thresh)
        headbox.has_mask = true;
      result.push_back(headbox);
    }
  }

  NmsDetect(result);
  vec_result.push_back(result);

  return vec_result;
}

extern "C" bool NvDsInferCustomRetinaFace(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList);

bool NvDsInferCustomRetinaFace(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList) {

  if (!gen_anchor) {

    for (const int step : feature_steps) {

      int feature_height = IMAGE_HEIGHT / step;
      int feature_width = IMAGE_WIDTH / step;
      std::vector<int> feature_map = {feature_height, feature_width};
      feature_maps.push_back(feature_map);
      int feature_size = feature_height * feature_width;
      feature_sizes.push_back(feature_size);
    }
    if (detect_mask)
      landmark_std = 0.2;

    sum_of_feature =
        std::accumulate(feature_sizes.begin(), feature_sizes.end(), 0) *
        anchor_num;
    GenerateAnchors();
    gen_anchor = true;
  }
  start_d = clock();
  std::vector<std::vector<FaceRes>> results = postProcess(
      (float *)(outputLayersInfo[0].buffer), outSize[0], cols, rows);

  std::vector<FaceRes> result = results[0];
  for (int j = 0; j < result.size(); j++) {
    NvDsInferInstanceMaskInfo object;
    if (!result[j].has_mask)
      object.classId = static_cast<int>(0);
    else
      object.classId = static_cast<int>(1);

    object.detectionConfidence = result[j].confidence;
    object.left = MAX(result[j].face_box.x - result[j].face_box.w / 2, 0);
    object.top = MAX(result[j].face_box.y - result[j].face_box.h / 2, 0);
    object.width = result[j].face_box.w;
    object.height = result[j].face_box.h;
    // std::cout<<object.left<<" "<<object.top<<" "<<object.width<<" "<<
    // object.height<<"\n";

    object.mask_size =
        sizeof(float) * mask_instance_width * mask_instance_height;
    object.mask = new float[mask_instance_width * mask_instance_height];
    object.mask_width = mask_instance_width;
    object.mask_height = mask_instance_height;

    float landmark[10];
    for (int k = 0; k < 5; k++) {

      landmark[2 * k + 1] = result[j].keypoints[k].x;
      landmark[2 * k] = result[j].keypoints[k].y;
    }

    float *lm = landmark;
    memcpy(object.mask, lm,
           sizeof(float) * mask_instance_width * mask_instance_height);

    // object.mask = result[j].landmark;
    // for (int i = 0; i < 10; i++)
    //   object.mask[i] = 0;

    objectList.push_back(object);
  }
  end_d = clock();
  float time_detect = ((double)(end_d - start_d)) / CLOCKS_PER_SEC;
  std::cout << "time postprocess : " << time_detect << "\n";

  return true;
}

extern "C" bool NvDsInferParseCustomExtraction(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList, std::string &attrString);

bool NvDsInferParseCustomExtraction(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {

  // std::cout << "feature norm : "
  //           << "-0.764025 -0.406269 0.132476 -0.137195 0.241286 -0.521208 "
  //              "0.522199 0.849268 1.73079 -2.06767\n";
  float *feature = (float *)outputLayersInfo[0].buffer;

  // std::ofstream
  // out("/home/haobk/Mydata/face_recognition_c_ver2/data/feature",
  // std::ios::out | std::ios::binary); out.write((char *)feature, 512 *
  // sizeof(float)); out.close();

  std::cout << "feature extract : ";
  for (int i = 0; i < 10; i++)
    std::cout << feature[i] << " ";
  std::cout << "\n\n\n";
  attrString = "";
  for (int i = 0; i < 512; i++) {
    attrString += std::to_string(feature[i]) + " ";
  }
  NvDsInferAttribute faceatt;
  faceatt.attributeIndex = 0;
  faceatt.attributeValue = 1;
  faceatt.attributeLabel = strdup(attrString.c_str());
  attrList.push_back(faceatt);
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferCustomRetinaFace);