#ifndef FaceAntispoofing
#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"

#define USE_FP16 // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0 // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
// #define max(a,b) \
//    ({ __typeof__ (a) _a = (a); \
//        __typeof__ (b) _b = (b); \
//      _a > _b ? _a : _b; })

// #define min(a,b) \
//    ({ __typeof__ (a) _a = (a); \
//        __typeof__ (b) _b = (b); \
//      _a < _b ? _a : _b; })

const int INPUT_H_A = Yolo::INPUT_H;
const int INPUT_W_A = Yolo::INPUT_W;
const int OUTPUT_SIZE_A = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1; // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1

// stuff we know about the network and the input/output blobs

class FaceAntispoofing
{
public:
    const int CLASS_NUM = Yolo::CLASS_NUM;
    char *INPUT_BLOB_NAME = "data";
    char *OUTPUT_BLOB_NAME = "prob";
    float data[BATCH_SIZE * 3 * INPUT_H_A * INPUT_W_A];
    float prob[BATCH_SIZE * OUTPUT_SIZE_A];
    int inputIndex;
    int outputIndex;
    IExecutionContext *context;
    IRuntime *runtime;
    ICudaEngine *engine;
    cudaStream_t stream;
    void *buffers[2];

    void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output, int batchSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H_A * INPUT_W_A * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE_A * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    bool setup(std::string engine_name)
    {
        std::ifstream file(engine_name, std::ios::binary);
        if (!file.good())
        {
            std::cerr << "read " << engine_name << " error!" << std::endl;
            return false;
        }
        char *trtModelStream = nullptr;
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();

        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        context = engine->createExecutionContext();
        delete[] trtModelStream;
        assert(engine->getNbBindings() == 2);

        inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H_A * INPUT_W_A * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE_A * sizeof(float)));

        CUDA_CHECK(cudaStreamCreate(&stream));
        std::cout << "load ast detection done \n";
        return true;
    }

    void preprocess(cv::Mat img)
    {
        cv::Mat pr_img = preprocess_img(img, INPUT_W_A, INPUT_H_A);
        int i = 0, b = 0;
        for (int row = 0; row < INPUT_H_A; ++row)
        {
            uchar *uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < INPUT_W_A; ++col)
            {
                data[b * 3 * INPUT_H_A * INPUT_W_A + i] = (float)uc_pixel[2] / 255.0;
                data[b * 3 * INPUT_H_A * INPUT_W_A + i + INPUT_H_A * INPUT_W_A] = (float)uc_pixel[1] / 255.0;
                data[b * 3 * INPUT_H_A * INPUT_W_A + i + 2 * INPUT_H_A * INPUT_W_A] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }
    }

    bool detect(cv::Mat img)
    {
        preprocess(img);
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        std::vector<Yolo::Detection> res;
        nms(res, &prob[0], CONF_THRESH, NMS_THRESH);
        if (res.size() > 0)
            return true;
        else
            return false;
    }

    void destroy()
    {
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(buffers[inputIndex]));
        CUDA_CHECK(cudaFree(buffers[outputIndex]));
        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
    }
};

#endif
