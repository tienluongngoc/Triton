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

int d=0;
int nbBindings = 10;
int mask_instance_width = 2;
int mask_instance_height = 5 + 2; // 5 : landmark 2 top_left bottom_right

int outSize[] = {12800, 51200, 128000, 3200, 12800, 32000, 800, 3200, 8000};
bool gen_anchor = false;
std::vector<int> listStride{8, 16, 32};
float dConfThreshold = 0.7;
int modelSize = 640;

std::vector<std::vector<float>> anchors;
std::vector<std::vector<std::vector<float>>> allAnchorCenters;

struct Box
{
  float x1;
  float x2;
  float y1;
  float y2;
  std::vector<cv::Point> landmarks;
  float confident;
};

static bool cmp(Box a, Box b)
{
  if (a.confident > b.confident)
    return true;
  return false;
}

static void nms(std::vector<Box> &input_boxes, float NMS_THRESH)
{
  std::vector<float> vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i)
  {
    vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) *
               (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i)
  {
    for (int j = i + 1; j < int(input_boxes.size());)
    {
      float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
      float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
      float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
      float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
      float w = std::max(float(0), xx2 - xx1 + 1);
      float h = std::max(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= NMS_THRESH)
      {
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      }
      else
      {
        j++;
      }
    }
  }
}

static void FilterBoxes(std::vector<Box> &predictedBoxes)
{
#pragma omp parallel for num_threads(2)

  std::sort(predictedBoxes.begin(), predictedBoxes.end(), cmp);
  nms(predictedBoxes, 0.4);
}

std::vector<std::vector<float>> GenerateAnchorCenters(int stride,
                                                      int modelSize)
{
  assert(modelSize % 32 == 0);
  std::vector<std::vector<float>> anchors;
  int size_height = modelSize / stride;
  int size_width = modelSize / stride;
  float cy = 0;
  for (int i = 0; i < size_height; i++)
  {
    float cx = 0;
    for (int k = 0; k < size_width; k++)
    {
      std::vector<float> anchor{cx, cy};
      anchors.emplace_back(anchor);
      anchors.emplace_back(anchor);
      cx = cx + (float)stride;
    }
    cy = cy + (float)stride;
  }
  return anchors;
}

void GenerateProposals(const std::vector<std::vector<float>> &anchors,
                       int feat_stride, const std::vector<float> &score_blob,
                       const std::vector<float> &bbox_blob,
                       const std::vector<float> &kps_blob,
                       std::vector<Box> &faceobjects, float dConfThreshold)
{
  for (int i = 0; i < score_blob.size(); i++)
  {
    float prob = score_blob[i];
    if (prob > dConfThreshold)
    {
      float box0 = bbox_blob[i * 4 + 0];
      float box1 = bbox_blob[i * 4 + 1];
      float box2 = bbox_blob[i * 4 + 2];
      float box3 = bbox_blob[i * 4 + 3];

      float dx = box0 * feat_stride;
      float dy = box1 * feat_stride;
      float dw = box2 * feat_stride;
      float dh = box3 * feat_stride;

      float cx = anchors[i][0];
      float cy = anchors[i][1];

      float x1 = cx - dx;
      float y1 = cy - dy;
      float x2 = cx + dw;
      float y2 = cy + dh;

      Box box;
      box.x1 = x1;
      box.y1 = y1;
      box.x2 = x2;
      box.y2 = y2;
      box.confident = prob;

      for (int k = 0; k < 5; k++)
      {
        float kpsx = cx + kps_blob[i * 10 + 2 * k] * feat_stride;
        float kpsy = cy + kps_blob[i * 10 + 2 * k + 1] * feat_stride;
        box.landmarks.emplace_back(cv::Point{kpsx, kpsy});
      }
      faceobjects.push_back(box);
    }
  }
}

extern "C" bool
NvDsInferCustomSCRFD(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                     NvDsInferNetworkInfo const &networkInfo,
                     NvDsInferParseDetectionParams const &detectionParams,
                     std::vector<NvDsInferInstanceMaskInfo> &objectList);

bool NvDsInferCustomSCRFD(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList)
{

  if (!gen_anchor)
  {
    for (auto &stride : listStride)
    {
      std::vector<std::vector<float>> anchorCenters =
          GenerateAnchorCenters(stride, modelSize);
      allAnchorCenters.emplace_back(anchorCenters);
    }
    gen_anchor = true;
  }
  start_d = clock();
  std::vector<std::vector<float>> outputs;
  std::vector<Box> predictedBoxes;

  float scale = 1;
  outputs.clear();
  predictedBoxes.clear();

  for (int i = 1; i < nbBindings; i++)
  {
    std::vector<float> v{(float *)(outputLayersInfo[i - 1].buffer), (float *)(outputLayersInfo[i - 1].buffer) + outSize[i - 1]};
    outputs.push_back(v);
  }

  for (int i = 0; i < listStride.size(); i++)
  {
    std::vector<Box> faceObjects;
    GenerateProposals(allAnchorCenters[i], listStride[i], outputs[i * 3], outputs[i * 3 + 1], outputs[i * 3 + 2], faceObjects, dConfThreshold);
    predictedBoxes.insert(predictedBoxes.end(), faceObjects.begin(), faceObjects.end());
  }

  FilterBoxes(predictedBoxes);

  for (auto &box : predictedBoxes)
  {
    // Rescale rectangles and landmarks
    box.x1 = box.x1 / scale;
    box.y1 = box.y1 / scale;
    box.x2 = box.x2 / scale;
    box.y2 = box.y2 / scale;
    for (auto &point : box.landmarks)
    {
      point.x = point.x / scale;
      point.y = point.y / scale;
    }
  }
  // std::cout<<"number face "<< predictedBoxes.size()<<"\n";

  for (int j = 0; j < predictedBoxes.size(); j++)
  {
    NvDsInferInstanceMaskInfo object;
    // if (!result[j].has_mask)
    object.classId = static_cast<int>(0);
    // else
    // object.classId = static_cast<int>(1);

    object.detectionConfidence = predictedBoxes[j].confident;
    object.left = MAX(predictedBoxes[j].x1,0);
    object.top = MAX(predictedBoxes[j].y1,0);
    object.width = MAX(predictedBoxes[j].x2 - predictedBoxes[j].x1,0);
    object.height = MAX(predictedBoxes[j].y2 - predictedBoxes[j].y1,0);
    // std::cout<<object.left<<" "<<object.top<<" "<<object.width<<" "<<
    // object.height<<"\n";

    object.mask_size =
        sizeof(float) * mask_instance_width * mask_instance_height;
    object.mask = new float[mask_instance_width * mask_instance_height];
    object.mask_width = mask_instance_width;
    object.mask_height = mask_instance_height;

    float landmark_box[mask_instance_width * mask_instance_height];
    for (int k = 0; k < 5; k++)
    {

      landmark_box[2 * k + 1] = predictedBoxes[j].landmarks[k].x;
      landmark_box[2 * k] = predictedBoxes[j].landmarks[k].y;
    }
    landmark_box[11] = MAX(predictedBoxes[j].x1,0);
    landmark_box[10] = MAX(predictedBoxes[j].y1,0);
    landmark_box[13] = MAX(predictedBoxes[j].x2,0);
    landmark_box[12] = MAX(predictedBoxes[j].y2,0);

    float *lm = landmark_box;
    memcpy(object.mask, lm,
           sizeof(float) * mask_instance_width * mask_instance_height);

    // object.mask = result[j].landmark;
    // for (int i = 0; i < 10; i++)
    //   object.mask[i] = 0;

    objectList.push_back(object);
  }
  end_d = clock();
  float time_detect = ((double)(end_d - start_d)) / CLOCKS_PER_SEC;
  // std::cout << "time postprocess : " << time_detect << "\n";
  // std::cout<<"number face :"<<objectList.size()<<"\n"; 
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

  // std::cout << "feature norm : "
  //           << "-0.764025 -0.406269 0.132476 -0.137195 0.241286 -0.521208 "
  //              "0.522199 0.849268 1.73079 -2.06767\n";
  float *feature = (float *)outputLayersInfo[0].buffer;

  // std::ofstream
  // out("/home/haobk/Mydata/face_recognition_c_ver2/data/feature",
  // std::ios::out | std::ios::binary); out.write((char *)feature, 512 *
  // sizeof(float)); out.close();

   //std::cout << "feature extract1 : ";
   //for (int i = 0; i < 10; i++)
     //std::cout << feature[i] << " ";
   //std::cout << d<<" ";
  // d+=1;
  attrString = "";
  for (int i = 0; i < 512; i++)
  {
    attrString += std::to_string(feature[i]) + " ";
  }
  NvDsInferAttribute faceatt;
  faceatt.attributeIndex = 0;
  faceatt.attributeValue = 1;
  faceatt.attributeLabel = strdup(attrString.c_str());
  attrList.push_back(faceatt);

  return true;
}

extern "C" bool NvDsInferParseCustomExtractionTriton(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList, std::string &attrString);

bool NvDsInferParseCustomExtractionTriton(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {

  auto layerFinder =
      [&outputLayersInfo](
          const std::string &name) -> const NvDsInferLayerInfo * {
    for (auto &layer : outputLayersInfo) {

      if (layer.dataType == FLOAT &&
          (layer.layerName && name == layer.layerName)) {
        return &layer;
      }
    }
    return nullptr;
  };
  //objectList.clear();
  const NvDsInferLayerInfo *feature_infer = layerFinder("output");
  float *feature = (float *)(feature_infer->buffer);
  

  // std::ofstream
  // out("/home/haobk/Mydata/face_recognition_c_ver2/data/feature",
  // std::ios::out | std::ios::binary); out.write((char *)feature, 512 *
  // sizeof(float)); out.close();

   //std::cout << "feature extract2 : ";
   //for (int i = 0; i < 10; i++)
     //std::cout << feature[i] << " ";
   //std::cout << d<<" ";
   //d+=1;
  attrString = "";
  //for (int i = 0; i < 512; i++) {
   // attrString += std::to_string(feature[i]) + " ";
  //}
  NvDsInferAttribute faceatt;
  faceatt.attributeIndex = 0;
  faceatt.attributeValue = 1;
  faceatt.attributeLabel = strdup(attrString.c_str());
  attrList.push_back(faceatt);

  return true;
}
/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferCustomSCRFD);
