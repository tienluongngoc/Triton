

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <numeric>

using namespace std;
#define CONF_THRESH 0.75
#define IOU_THRESH 0.4

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


int outSize[] = {25600, 51200, 51200, 256000};
int bufferSize[] = {0, 102400, 204800, 204800, 1024000};
int BATCH_SIZE = 1;
int IMAGE_WIDTH = 640;
int IMAGE_HEIGHT = 640;
int cols = 640; //size origin
int rows = 640;
float obj_threshold = 0.5;
float nms_threshold = 0.45;

struct FaceBox
{
  float x;
  float y;
  float w;
  float h;
};
int mask_instance_width = 2;
int mask_instance_height = 5;
struct FaceRes
{
  float confidence;
  FaceBox face_box;
  std::vector<cv::Point2f> keypoints;
  float landmark[10];
};

float IOUCalculate(const FaceBox &det_a, const FaceBox &det_b)
{
  cv::Point2f center_a(det_a.x, det_a.y);
  cv::Point2f center_b(det_b.x, det_b.y);
  cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                      std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
  cv::Point2f right_down(
      std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
      std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
  float distance_d = (center_a - center_b).x * (center_a - center_b).x +
                     (center_a - center_b).y * (center_a - center_b).y;
  float distance_c = (left_up - right_down).x * (left_up - right_down).x +
                     (left_up - right_down).y * (left_up - right_down).y;
  float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2
                      ? det_a.x - det_a.w / 2
                      : det_b.x - det_b.w / 2;
  float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2
                      ? det_a.y - det_a.h / 2
                      : det_b.y - det_b.h / 2;
  float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2
                      ? det_a.x + det_a.w / 2
                      : det_b.x + det_b.w / 2;
  float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2
                      ? det_a.y + det_a.h / 2
                      : det_b.y + det_b.h / 2;
  if (inter_b < inter_t || inter_r < inter_l)
    return 0;
  float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
  float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
  if (union_area == 0)
    return 0;
  else
    return inter_area / union_area - distance_d / distance_c;
}
void NmsDetect(std::vector<FaceRes> &detections)
{
  sort(detections.begin(), detections.end(),
       [=](const FaceRes &left, const FaceRes &right)
       {
         return left.confidence > right.confidence;
       });

  for (int i = 0; i < (int)detections.size(); i++)
    for (int j = i + 1; j < (int)detections.size(); j++)
    {
      float iou = IOUCalculate(detections[i].face_box, detections[j].face_box);
      if (iou > nms_threshold)
        detections[j].confidence = 0;
    }

  detections.erase(
      std::remove_if(detections.begin(), detections.end(),
                     [](const FaceRes &det)
                     { return det.confidence == 0; }),
      detections.end());
}

std::vector<std::vector<FaceRes>>
postProcess(float *output_1, float *output_2, float *output_3, float *output_4,
            const int &outSize_1, const int &outSize_2, const int &outSize_3,
            const int &outSize_4)
{

  float x, y, w, h;
  int image_size = IMAGE_WIDTH / 4 * IMAGE_HEIGHT / 4;

  float ratio =
      float(cols) / float(IMAGE_WIDTH) > float(rows) / float(IMAGE_HEIGHT)
          ? float(cols) / float(IMAGE_WIDTH)
          : float(rows) / float(IMAGE_HEIGHT);

  std::vector<std::vector<FaceRes>> vec_result;
  int index = 0;
  std::vector<FaceRes> result;

  float *score = output_1 + index * outSize_1;
  float *scale0 = output_2 + index * outSize_2;
  float *scale1 = scale0 + image_size;
  float *offset0 = output_3 + index * outSize_3;
  float *offset1 = offset0 + image_size;
  float *landmark = output_4 + index * outSize_4;

  for (int i = 0; i < IMAGE_HEIGHT / 4; i++)
  {
    for (int j = 0; j < IMAGE_WIDTH / 4; j++)
    {
      int current = i * IMAGE_WIDTH / 4 + j;
      if (score[current] > obj_threshold)
      {
        FaceRes headbox;
        headbox.confidence = score[current];
        headbox.face_box.h = std::exp(scale0[current]) * 4 * ratio;
        headbox.face_box.w = std::exp(scale1[current]) * 4 * ratio;
        headbox.face_box.x = ((float)j + offset1[current] + 0.5f) * 4 * ratio;
        headbox.face_box.y = ((float)i + offset0[current] + 0.5f) * 4 * ratio;
        for (int k = 0; k < 5; k++)
        {

          headbox.landmark[2 * k + 1] =
              (headbox.face_box.x - headbox.face_box.w / 2 + landmark[(2 * k + 1) * image_size + current] * headbox.face_box.w);
          headbox.landmark[2 * k] =
              headbox.face_box.y - headbox.face_box.h / 2 + landmark[(2 * k) * image_size + current] * headbox.face_box.h;
        }

        result.push_back(headbox);
      }
    }
  }
  NmsDetect(result);
  vec_result.push_back(result);
  index++;
  return vec_result;
}

extern "C" bool NvDsInferCustomCenterFace(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList);

bool NvDsInferCustomCenterFace(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList)
{
  // std::cout<<"height "<< networkInfo.height << " width "<<networkInfo.width<<"\n";

  std::vector<std::vector<FaceRes>> results =
      postProcess((float *)(outputLayersInfo[0].buffer),
                  (float *)(outputLayersInfo[1].buffer),
                  (float *)(outputLayersInfo[2].buffer),
                  (float *)(outputLayersInfo[3].buffer), outSize[0], outSize[1],
                  outSize[2], outSize[3]);
  std::vector<FaceRes> result = results[0];

  for (int j = 0; j < result.size(); j++)
  {
    NvDsInferInstanceMaskInfo object;

    object.classId = static_cast<int>(0);
    object.detectionConfidence = result[j].confidence;
    object.left = MAX(result[j].face_box.x - result[j].face_box.w / 2, 0);
    object.top = MAX(result[j].face_box.y - result[j].face_box.h / 2, 0);
    object.width = result[j].face_box.w;
    object.height = result[j].face_box.h;
    // std::cout<<object.left<<" "<<object.top<<" "<<object.width<<" "<< object.height<<"\n";

    object.mask_size =
        sizeof(float) * mask_instance_width * mask_instance_height;
    object.mask = new float[mask_instance_width * mask_instance_height];
    object.mask_width = mask_instance_width;
    object.mask_height = mask_instance_height;

    float *lm = result[j].landmark;
    memcpy(object.mask, lm,
           sizeof(float) * mask_instance_width * mask_instance_height);

    // object.mask = result[j].landmark;
    // for (int i = 0; i < 10; i++)
    //   object.mask[i] = 0;

    objectList.push_back(object);
  }

  return true;
}

extern "C" bool NvDsInferParseCustomExtraction(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList, std::string &attrString);

bool NvDsInferParseCustomExtraction(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList, std::string &attrString)
{
  
  float *feature = (float *)outputLayersInfo[0].buffer;
  // ofstream out("feature", ios::out | ios::binary);
  // out.write((char *)feature, 512 * sizeof(float));
  // out.close();

  // cv::Mat out_m(512, 1, CV_32FC1, feature);
  // cv::Mat out_norm;
  // cv::normalize(out_m, feature_out.f);

  // for(int i = 0 ; i<10;i++)
  //   std::cout<<feature[i]<<" ";
  // std::cout<<"\n";
  attrString = "Hao" + std::to_string(rand() % 10 + 1);
  NvDsInferAttribute faceatt;
  faceatt.attributeIndex = 0;
  faceatt.attributeValue = 1;
  faceatt.attributeLabel = strdup(attrString.c_str());
  attrList.push_back(faceatt);
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferCustomCenterFace);