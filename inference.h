#pragma once
#define RET_OK nullptr
#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif
#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include <filesystem>

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

enum MODEL_TYPE
{
    YOLO_DET_SEG_V8 = 1,
    YOLO_CLS_V8 = 2,
};

typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    float rectConfidenceThreshold = 0.6f;
    float iouThreshold = 0.5f;
    bool cudaEnable = false;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
} DL_INIT_PARAM;

typedef struct _DL_RESULT
{
    int classId;
    std::string className;
    float confidence;
    cv::Rect box;
    std::vector<std::vector<cv::Point>> contours; // 分割的轮廓点
    cv::Scalar color;
} DL_RESULT;



class YOLO_V8
{
public:
    YOLO_V8();
    ~YOLO_V8();
public:
    char* CreateSession(DL_INIT_PARAM& iParams);
    char* RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);
    char* WarmUpSession();
    template<typename N>
    char* TensorProcess(std::chrono::_V2::system_clock::time_point& starttime_1, cv::Vec4d& params, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        std::vector<DL_RESULT>& oResult);
    char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);
    int ReadClassNames(const std::string& txtPath, std::vector<std::string>& classNames);
    std::vector<std::string> classes{};
    int classNums = 80;
    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    std::vector<DL_RESULT> Inference(const std::string& imagePath,const std::string& txtPath);
private:
    bool cudaEnable;
    Ort::Env env;
    Ort::Session* session;
    Ort::RunOptions options;
    bool runSegmentation = false;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> outputShapes;
    float rectConfidenceThreshold;
    float iouThreshold;
    // bool isDynamicInputShape{};
};
