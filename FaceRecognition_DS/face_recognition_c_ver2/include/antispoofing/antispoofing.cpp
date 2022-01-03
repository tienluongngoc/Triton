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
#define max(a, b)                   \
    (                               \
        {                           \
            __typeof__(a) _a = (a); \
            __typeof__(b) _b = (b); \
            _a > _b ? _a : _b;      \
        })

#define min(a, b)                   \
    (                               \
        {                           \
            __typeof__(a) _a = (a); \
            __typeof__(b) _b = (b); \
            _a < _b ? _a : _b;      \
        })

const int INPUT_H = Yolo::INPUT_H;
const int INPUT_W = Yolo::INPUT_W;
const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1; // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1

// stuff we know about the network and the input/output blobs

static Logger gLogger;

class FaceExtraction
{
public:
    const int CLASS_NUM = Yolo::CLASS_NUM;
    char *INPUT_BLOB_NAME = "data";
    char *OUTPUT_BLOB_NAME = "prob";
    float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    float prob[BATCH_SIZE * OUTPUT_SIZE];
    int inputIndex;
    int outputIndex;
    IExecutionContext *context;
    IRuntime *runtime;
    ICudaEngine *engine;
    cudaStream_t stream;
    void *buffers[2];

    void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output, int batchSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
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
        CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

        CUDA_CHECK(cudaStreamCreate(&stream));
        std::cout << "load ast detection done \n";
        return true;
    }

    void preprocess(cv::Mat img)
    {
        cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);
        int i = 0, b = 0;
        for (int row = 0; row < INPUT_H; ++row)
        {
            uchar *uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < INPUT_W; ++col)
            {
                data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }
    }

    std::vector<Yolo::Detection> detect(cv::Mat img)
    {
        preprocess(img);
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        std::vector<Yolo::Detection> res;
        nms(res, &prob[0], CONF_THRESH, NMS_THRESH);
        if (res.size > 0)
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

// cv::Mat crop(cv::Mat img,cv::Rect r,int padding){//img(Range(start_row, end_row), Range(start_col, end_col))
//     int x1,x2,y1,y2;
//     int cols = img.cols;
//     int rows = img.rows;
//     x1=max((int) r.tl().x-padding , 0);
//     x2=min((int) r.br().x+padding,cols);
//     y1=max((int) r.tl().y-padding,0);
//     y2=min((int) r.br().y+padding,rows);
//     return img(cv::Range(y1,y2),cv::Range(x1,x2)); // Slicing to crop the image
// }

// cv::Mat drop_face(cv::Mat img){
//     cv::Mat image = img.clone();
//     int cols = 112;
//     int rows = 112;
//     for (int row = 56; row < rows; ++row)
//         for (int col = 0; col < cols; ++col){
//             cv::Vec3b & color = image.at<cv::Vec3b>(row,col);
//             color[0] = 255;
//             color[1] = 255;
//             color[2] = 255;

//         }
//     return image;
// }

// int main(int argc, char** argv) {
//     cudaSetDevice(DEVICE);
//     std::string engine_name = "ats.engine";
// cv::Mat img = cv::imread("/home/haobk/Mydata/Engine/server/res2tensort_server/Face_Recognition_C/test_img/test.jpg");
// cv::Mat img_dropped = drop_face(img);
// cv::imwrite("origin.jpg",img);
// cv::imwrite("dropped.jpg",img_dropped);

// FaceExtraction ats_detection;
// ats_detection.setup(engine_name);

// cv::Mat img;
// cv::VideoCapture cap;
// cap.open("../a.mp4");
// if (!cap.isOpened()) {
//     std::cerr << "ERROR! Unable to open camera\n";
//     return -1;
// }
// for (;;)
// {
//     cap.read(img);
//     if (img.empty()) {
//         std::cerr << "ERROR! blank frame grabbed\n";
//         break;
//     }

//     std::vector<Yolo::Detection> res = ats_detection.detect(img);
//     for (size_t j = 0; j < res.size(); j++) {
//         cv::Rect r = get_rect(img, res[j].bbox);
//         cv::imwrite("image_cropped.jpg",crop(img,r,0));
//         cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
//         cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);

//     }
//     cv::imshow("Live", img);
//     if (cv::waitKey(5) >= 0)
//         break;

// }

//     return 0;
// }
