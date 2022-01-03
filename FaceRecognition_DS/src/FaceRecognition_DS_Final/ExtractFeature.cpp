#include"FaceRecognition_DS_Final/ExtractFeature.h"


void FaceExtraction::setup(cudaStream_t *stream, IRuntime *runtime,Logger *gLogger,std::string path)
{
    
    std::vector<char> trtModelStream_;
    this->stream = stream;
    this->runtime = runtime;
    this->gLogger = gLogger;

    size_t size{0};

    std::ifstream file(path, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream_.resize(size);

        file.read(trtModelStream_.data(), size);
        file.close();
    }
    
    engine = this->runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
    context = engine->createExecutionContext();
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME_F);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME_F);
    delete[] trtModelStream;
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H_F * INPUT_W_F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE_F * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(this->stream));
    std::cout << "batch size " << engine->getMaxBatchSize() << "\n";
    std::cout << "load extract_feature detection done \n";
}

void FaceExtraction::doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output, int batchSize)
{
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H_F * INPUT_W_F * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE_F * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void FaceExtraction::prepare(std::vector<cv::Mat> imgs)
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

cv::Mat FaceExtraction::drop_face(cv::Mat img)
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

std::vector<cv::Mat> FaceExtraction::extract_non_facemask(std::vector<cv::Mat> imgs)
{
    int size = imgs.size();
    std::vector<cv::Mat> res;
    prepare(imgs);
    doInference(*context, *this->stream, this->buffers, this->data, this->prob, size);
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

std::vector<cv::Mat> FaceExtraction::extract(std::vector<cv::Mat> imgs, std::vector<bool> face_mask)
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

    doInference(*this->context, *this->stream, this->buffers, this->data, this->prob, size);
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

float* FaceExtraction::extract_file(cv::Mat img)
{

    std::vector<cv::Mat> imgs;
    imgs.push_back(img);
    cv::Mat img_dropped = drop_face(img);
    // cv::imwrite(name+"_add_face.jpg",img_dropped);
    imgs.push_back(img_dropped);
    prepare(imgs);
    doInference(*this->context, *this->stream, this->buffers, this->data, this->prob, 2);
    return prob;
}

void FaceExtraction:: destroy()
{
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

float FaceExtraction::distance(cv::Mat f1, cv::Mat f2)
{
    return cv::norm(f1, f2);
};


