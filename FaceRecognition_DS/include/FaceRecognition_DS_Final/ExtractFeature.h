#ifndef Face_Extraction
#define Face_Extraction
#include "cuda_utils.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "FaceRecognition_DS_Final/logging.h"

#define BATCH_SIZE 4

using namespace nvinfer1;

static const int INPUT_H_F = 112;
static const int INPUT_W_F = 112;
static const int OUTPUT_SIZE_F = 512;
static const char *INPUT_BLOB_NAME_F = "input";
static const char *OUTPUT_BLOB_NAME_F = "output";

class FaceExtraction
{
    private:
        char *trtModelStream{nullptr};
        size_t size;
        ICudaEngine *engine;
        IExecutionContext *context;
        IRuntime *runtime;
        Logger *gLogger;
        cudaStream_t *stream;
        float data[BATCH_SIZE * 3 * INPUT_H_F * INPUT_W_F];
        float prob[BATCH_SIZE * OUTPUT_SIZE_F];
        void *buffers[2];
        int inputIndex;
        int outputIndex;

    public:
        void setup(cudaStream_t *stream, IRuntime *runtime,Logger *gLogger,std::string path);
        cv::Mat drop_face(cv::Mat img);
        void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output, int batchSize);
        void prepare(std::vector<cv::Mat> imgs);
        float distance(cv::Mat f1, cv::Mat f2);
        std::vector<cv::Mat> extract_non_facemask(std::vector<cv::Mat> imgs);
        std::vector<cv::Mat> extract(std::vector<cv::Mat> imgs, std::vector<bool> face_mask);
        float* extract_file(cv::Mat img);
        void destroy();
        
};

float distance(cv::Mat f1, cv::Mat f2);

#endif
