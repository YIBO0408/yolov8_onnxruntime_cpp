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

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif


enum MODEL_TYPE
{
    //FLOAT32 MODEL
    YOLO_DETECT_V8 = 1,
    YOLO_POSE = 2,
    YOLO_CLS = 3,

    //FLOAT16 MODEL
    YOLO_DETECT_V8_HALF = 4,
    YOLO_POSE_V8_HALF = 5,
};


typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    MODEL_TYPE modelType = YOLO_DETECT_V8;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    // int	keyPointsNum = 2;
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
    // std::vector<cv::Point2f> keyPoints;
    cv::Mat boxMask;       //矩形框内mask
    cv::Scalar color;
} DL_RESULT;


class YOLO_V8
{
public:
    YOLO_V8();
    ~YOLO_V8();

public:
    char* CreateSession(DL_INIT_PARAM& iParams);
    void DrawPred(cv::Mat& img, std::vector<DL_RESULT>& result);
    char* RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);
    char* WarmUpSession();
    template<typename N>
    // char* TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
    //     std::vector<DL_RESULT>& oResult);
    char* TensorProcess(clock_t& starttime_1, cv::Vec4d& params, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        std::vector<DL_RESULT>& oResult);

    char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);

    int ReadClassNames(const std::string& yamlPath, std::vector<std::string>& classNames);
    
    std::vector<std::string> classes{};

    bool cudaEnable;

    std::vector<int> imgSize;


private:
    Ort::Env env;
    Ort::Session* session;
    Ort::RunOptions options;
    bool RunSegmentation = false;
    // bool cudaEnable;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;
    // std::vector<int> imgSize;

    MODEL_TYPE modelType;
    float rectConfidenceThreshold;
    float iouThreshold;
    // float resizeScales;
};
