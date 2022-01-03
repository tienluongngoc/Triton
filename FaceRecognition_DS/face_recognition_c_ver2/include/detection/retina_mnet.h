#ifndef retina_mnet
#include <fstream>
#include <bits/stdc++.h>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "calibrator.h"
#include "alignment.h"
#define BATCH_SIZE 1

static const int INPUT_H = decodeplugin::INPUT_H; // H, W must be able to  be divided by 32.
static const int INPUT_W = decodeplugin::INPUT_W;
;
static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2 * 15 + 1;
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";
static cudaStream_t stream;
static Logger gLogger;
static IRuntime *runtime;
#define retina_mnet

#define USE_FP16 // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0 // GPU id

#define CONF_THRESH 0.75
#define IOU_THRESH 0.4
#define CLOCKS_PER_SEC 1000000


struct face_landmark
{
	float x[5];
	float y[5];
};

struct res_det
{
    cv::Rect boxes;
    float conf;
    face_landmark landmark;
    cv::Mat img_aligned;
};


class FaceDection
{
private:
    std::string path_model; //"retina_mnet.engine" ;
    char *trtModelStream{nullptr};
    size_t size{0};
    
    ICudaEngine *engine;
    IExecutionContext *context;
    float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    float prob[BATCH_SIZE * OUTPUT_SIZE];


public:
    void set_up(std::string path)
    {
        path_model = path;
        std::ifstream file(path_model, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        context = engine->createExecutionContext();
    }


    void doInference(IExecutionContext &context, float *input, float *output, int batchSize){
        const ICudaEngine &engine = context.getEngine();
        assert(engine.getNbBindings() == 2);
        void *buffers[2];
        const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
        const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
        CHECK(cudaStreamCreate(&stream));
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
    }

    void prepare(cv::Mat img)
    {
        cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);
        for (int b = 0; b < BATCH_SIZE; b++)
        {
            float *p_data = &data[b * 3 * INPUT_H * INPUT_W];
            for (int i = 0; i < INPUT_H * INPUT_W; i++)
            {
                p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
                p_data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
                p_data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
            }
        }
    }

    std::vector<res_det> postprocess(cv::Mat img)
    {   
        std::vector<res_det> ress;
        std::vector<decodeplugin::Detection> res;
        nms(res, &prob[0], IOU_THRESH);
        cv::Mat tmp = img.clone();
        for (size_t j = 0; j < res.size(); j++)
        {
            
            if (res[j].class_confidence < CONF_THRESH)
                continue;
            
            res_det res_c;
            res_c.conf = res[j].class_confidence;
            cv::Rect r = get_rect_adapt_landmark(tmp, INPUT_W, INPUT_H, res[j].bbox, res[j].landmark);
            res_c.boxes = r;
            
            
            // cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            for (int k = 0; k < 10; k += 2) {
                    int index=k/2;
                    res_c.landmark.x[index]=res[j].landmark[k];
                    res_c.landmark.y[index]=res[j].landmark[k+1];
            //     cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
            }
            int ret_ali = get_aligned_face(img,(float *)&res_c.landmark,5,112,res_c.img_aligned);
            ress.push_back(res_c);
            
        }
        
    
        return ress;
    }

    std::vector<res_det> detect(cv::Mat img)
    {
        prepare(img);
        doInference(*context, data, prob, BATCH_SIZE);
        return postprocess(img);
    }

    void destroy()
    {
        context->destroy();
        engine->destroy();
        runtime->destroy();
    }
};

#endif