#ifndef ExtractFeature
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
#include "logging.h"

#define BATCH_SIZE 4
const int INPUT_H_F = 112;
const int INPUT_W_F = 112;
using namespace nvinfer1;
static const int OUTPUT_SIZE_F = 512;
const char *INPUT_BLOB_NAME_F = "input";
const char *OUTPUT_BLOB_NAME_F = "output";

class FaceExtraction
{
private:
    std::string path_model;
    ;
    char *trtModelStream{nullptr};
    size_t size;
    ICudaEngine *engine;
    IExecutionContext *context;
    float data[BATCH_SIZE * 3 * INPUT_H_F * INPUT_W_F];
    float prob[BATCH_SIZE * OUTPUT_SIZE_F];
    void *buffers[2];
    int inputIndex;
    int outputIndex;

public:
    void setup(std::string path)
    {
        path_model = path;
        std::vector<char> trtModelStream_;
        size_t size{0};

        std::ifstream file(path_model, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream_.resize(size);

            file.read(trtModelStream_.data(), size);
            file.close();
        }
        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
        context = engine->createExecutionContext();
        inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME_F);
        outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME_F);
        delete[] trtModelStream;
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&stream));
        std::cout << "batch size " << engine->getMaxBatchSize() << "\n";
        std::cout << "load extract_feature detection done \n";
    }

    void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output, int batchSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H_F * INPUT_W_F * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE_F * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    void prepare(std::vector<cv::Mat> imgs)
    {

        for (int b = 0; b < imgs.size(); b++)
        {
            float *p_data = &data[b * 3 * INPUT_H_F * INPUT_W_F];
            for (int i = 0; i < INPUT_H_F * INPUT_W_F; i++)
            {
                p_data[i] = ((float)imgs[b].at<cv::Vec3b>(i)[2] / 255.0 - 0.5) / 0.5;
                p_data[i + INPUT_H_F * INPUT_W_F] = ((float)imgs[b].at<cv::Vec3b>(i)[1] / 255.0 - 0.5) / 0.5;
                p_data[i + 2 * INPUT_H_F * INPUT_W_F] = ((float)imgs[b].at<cv::Vec3b>(i)[0] / 255.0 - 0.5) / 0.5;
            }
        }
    }

    cv::Mat drop_face(cv::Mat img)
    {
        cv::Mat image = img.clone();
        int cols = 112;
        int rows = 112;
        for (int row = 74; row < rows; ++row)
            for (int col = 0; col < cols; ++col)
            {
                cv::Vec3b &color = image.at<cv::Vec3b>(row, col);
                color[0] = 0;
                color[1] = 0;
                color[2] = 0;
            }
        return image;
    }

    std::vector<cv::Mat> extract_non_facemask(std::vector<cv::Mat> imgs)
    {
        int size = imgs.size();
        std::vector<cv::Mat> res;
        prepare(imgs);
        doInference(*context, stream, buffers, data, prob, size);
        for (int i = 0; i < size; i++)
        {
            float *pro = &prob[i * 512];

            cv::Mat out(512, 1, CV_32FC1, pro);
            cv::Mat out_norm;
            cv::normalize(out, out_norm);
            res.push_back(out_norm);
        }
        return res;
    }

    std::vector<cv::Mat> extract(std::vector<cv::Mat> imgs, std::vector<bool> face_mask)
    {
        int size = imgs.size();
        std::vector<cv::Mat> res;
        prepare(imgs);
        for (int i = 0; i < size; i++)
        {
            if (face_mask[i])
                imgs[i] = drop_face(imgs[i]);
            // cv::imwrite(std::to_string(i)+"extract.jpg",imgs[i]);
        }

        doInference(*context, stream, buffers, data, prob, size);
        for (int i = 0; i < size; i++)
        {
            float *pro = &prob[i * 512];
            for(int i = 0 ; i<10;i++)
                std::cout<<prob[i]<<" ";
            std::cout<<"\n";

            cv::Mat out(512, 1, CV_32FC1, pro);
            cv::Mat out_norm;
            cv::normalize(out, out_norm);
            res.push_back(out_norm);
        }
        return res;
    }

    float *extract_file(cv::Mat img, std::string name)
    {

        std::vector<cv::Mat> imgs;
        imgs.push_back(img);
        cv::Mat img_dropped = drop_face(img);
        // cv::imwrite(name+"_add_face.jpg",img_dropped);
        imgs.push_back(img_dropped);
        prepare(imgs);
        doInference(*context, stream, buffers, data, prob, 2);
        return prob;
    }

    void destroy()
    {
        context->destroy();
        engine->destroy();
        runtime->destroy();
    }
};

float distance(cv::Mat f1, cv::Mat f2)
{
    return cv::norm(f1, f2);
};

#endif
