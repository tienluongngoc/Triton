#ifndef retina_mnet_mask

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
#include "decode.h"
#include "alignment.h"
#include "cuda_utils.h"
#define retina_mnet_mask


//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1  // currently, only support BATCH=1

using namespace nvinfer1;

static const int INPUT_H = decodeplugin::INPUT_H;
static const int INPUT_W = decodeplugin::INPUT_W;
static const int DETECTION_SIZE = sizeof(decodeplugin::Detection) / sizeof(float);
static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * DETECTION_SIZE + 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;
REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);
static cudaStream_t stream;
static IRuntime *runtime;



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
    bool face_mask;

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
    float mask_thresh = 0.5;
    float CONF_THRESH = 0.8;
    int inputIndex;
    int outputIndex;
    void* buffers[2];


public:
    
    void setup(std::string path)
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
        delete[] trtModelStream;
        assert(engine->  getNbBindings() == 2);
        
        inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
        
        CUDA_CHECK(cudaStreamCreate(&stream));
        std::cout<<"load face detection done \n";
        
    }

    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
            CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
            context.enqueue(batchSize, buffers, stream, nullptr);
            CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);

    }

    void prepare(cv::Mat img) {
        cv::Mat pr_img = preprocess_img(img);
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            data[i] = ((float)pr_img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
            data[i + INPUT_H * INPUT_W] = ((float)pr_img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
            data[i + 2 * INPUT_H * INPUT_W] = ((float)pr_img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
        }
    }
    cv::Mat preprocess_img(cv::Mat& img) {
        int w, h, x, y;
        float r_w = INPUT_W / (img.cols*1.0);
        float r_h = INPUT_H / (img.rows*1.0);
        if (r_h > r_w) {
            w = INPUT_W;
            h = r_w * img.rows;
            x = 0;
            y = (INPUT_H - h) / 2;
        } else {
            w = r_h* img.cols;
            h = INPUT_H;
            x = (INPUT_W - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
        cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
        return out;
    }


    static bool cmp(decodeplugin::Detection& a, decodeplugin::Detection& b) {
        return a.class_confidence > b.class_confidence;
    }
    
    cv::Rect get_rect_adapt_landmark(cv::Mat& img, float bbox[4], float lmk[10]) {
            int l, r, t, b;
            float r_w = INPUT_W / (img.cols * 1.0);
            float r_h = INPUT_H / (img.rows * 1.0);
            if (r_h > r_w) {
                l = bbox[0] / r_w;
                r = bbox[2] / r_w;
                t = (bbox[1] - (INPUT_H - r_w * img.rows) / 2) / r_w;
                b = (bbox[3] - (INPUT_H - r_w * img.rows) / 2) / r_w;
                for (int i = 0; i < 10; i += 2) {
                    lmk[i] /= r_w;
                    lmk[i + 1] = (lmk[i + 1] - (INPUT_H - r_w * img.rows) / 2) / r_w;
                }
            } else {
                l = (bbox[0] - (INPUT_W - r_h * img.cols) / 2) / r_h;
                r = (bbox[2] - (INPUT_W - r_h * img.cols) / 2) / r_h;
                t = bbox[1] / r_h;
                b = bbox[3] / r_h;
                for (int i = 0; i < 10; i += 2) {
                    lmk[i] = (lmk[i] - (INPUT_W - r_h * img.cols) / 2) / r_h;
                    lmk[i + 1] /= r_h;
                }
            }
            return cv::Rect(l, t, r-l, b-t);
    }

    float iou(float lbox[4], float rbox[4]) {
        float interBox[] = {
            std::max(lbox[0], rbox[0]), //left
            std::min(lbox[2], rbox[2]), //right
            std::max(lbox[1], rbox[1]), //top
            std::min(lbox[3], rbox[3]), //bottom
        };

        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
        return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) -interBoxS + 0.000001f);
    }
    
    void nms(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh = 0.4) {
        std::vector<decodeplugin::Detection> dets;
        for (int i = 0; i < output[0]; i++) {
            if (output[DETECTION_SIZE * i + 1 + 4] <= 0.1) continue;
            decodeplugin::Detection det;
            memcpy(&det, &output[DETECTION_SIZE * i + 1], sizeof(decodeplugin::Detection));
            dets.push_back(det);
        }
        std::sort(dets.begin(), dets.end(), cmp);
        if (dets.size() > 5000) dets.erase(dets.begin() + 5000, dets.end());
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            //std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

    std::vector<res_det> postprocess(cv::Mat img){   
            
            std::vector<res_det> ress;
            std::vector<decodeplugin::Detection> res;
            nms(res, prob);
            cv::Mat tmp = img.clone();
            for (size_t j = 0; j < res.size(); j++){
                
                if (res[j].class_confidence < CONF_THRESH)
                    continue;
                
                res_det res_c;
                res_c.conf = res[j].class_confidence;
                cv::Rect r = get_rect_adapt_landmark(img, res[j].bbox, res[j].landmark);
                res_c.boxes = r;
                
                // cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                // cv::putText(img, "face: " + std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y + 20), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
               
                // cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                for (int k = 0; k < 10; k += 2) {
                        int index=k/2;
                       res_c.landmark.x[index]=res[j].landmark[k];
                       res_c.landmark.y[index]=res[j].landmark[k+1];
                       if(res[j].mask_confidence >mask_thresh)
                            res_c.face_mask=true;
                       else
                            res_c.face_mask=false;}
                //draw landmark
                // for (int k = 0; k < 10; k += 2) {
                        // cv::circle(img, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
                // }
                // cv::putText(img, "mask: " + std::to_string((int)(res[j].mask_confidence * 100)) + "%", cv::Point(r.x, r.y + 40), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x00, 0x00, 0xFF), 1);
                
                int a=get_aligned_face(img,(float *)&res_c.landmark,5,112,res_c.img_aligned);
                ress.push_back(res_c);
                }
               
                // cv::imwrite("out.jpg", img);

                //     cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
                
             
                
            
            
        
            return ress;
    }

    std::vector<res_det> detect(cv::Mat img)
    {
        prepare(img);
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        return postprocess(img);
    }

    void destroy()
    {
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(buffers[inputIndex]));
        CUDA_CHECK(cudaFree(buffers[outputIndex]));
        context->destroy();
        engine->destroy();
        runtime->destroy();
    }
};


#endif