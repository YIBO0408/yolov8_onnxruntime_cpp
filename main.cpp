#include <iostream>
#include <getopt.h>
#include <filesystem>
#include <opencv2/core.hpp>
#include <fstream>
#include "inference.h"

std::filesystem::path projectRoot = std::filesystem::current_path().parent_path();

std::vector<DL_RESULT> DetectImage(const std::string& imagePath, const std::string& modelPath, const std::string& yamlPath, 
const cv::Size& imgSize, float rectConfidenceThreshold = 0.45, float iouThreshold = 0.5, bool useGPU = false) 
{
    std::vector<DL_RESULT> results;

    DL_INIT_PARAM params{modelPath, YOLO_DET_SEG_V8, {imgSize.width, imgSize.height}, rectConfidenceThreshold, iouThreshold, useGPU};
    std::unique_ptr<YOLO_V8> yoloDetector(new YOLO_V8);
    if (yoloDetector->CreateSession(params) != 0) {
        std::cerr << "[YOLO_V8]: Failed to create session" << std::endl;
        return results;
    }
    std::vector<std::string> classNames;
    if (yoloDetector->ReadClassNames(yamlPath, classNames) != 0) {
        std::cerr << "[YOLO_V8]: Failed to read class names" << std::endl;
        return results;
    }
    yoloDetector->classes = std::move(classNames);
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "[YOLO_V8]: Failed to load image" << std::endl;
        return results;
    }
    std::vector<DL_RESULT> res;
    yoloDetector->RunSession(image, res);
    for (const auto& result : res) {
        results.push_back(result);
    }
    return results;
}


std::vector<DL_RESULT> ClassifyImage(const std::string& imagePath, const std::string& modelPath, const std::string& yamlPath, const cv::Size& imgSize, bool useGPU) 
{
    std::vector<DL_RESULT> results;

    DL_INIT_PARAM params{modelPath, YOLO_CLS_V8, {imgSize.width, imgSize.height}};

    std::unique_ptr<YOLO_V8> yoloClassifier(new YOLO_V8);


    params.cudaEnable = useGPU;

    if (yoloClassifier->CreateSession(params) != 0) {
        std::cerr << "[YOLO_V8]: Failed to create session" << std::endl;
        return results;
    }

    std::vector<std::string> classNames;
    if (yoloClassifier->ReadClassNames(yamlPath, classNames) != 0) {
        std::cerr << "[YOLO_V8]: Failed to read class names" << std::endl;
        return results;
    }

    yoloClassifier->classes = std::move(classNames);

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "[YOLO_V8]: Failed to load image" << std::endl;
        return results;
    }


    std::vector<DL_RESULT> res;

    if (yoloClassifier->RunSession(image, res) != 0) {
        std::cerr << "[YOLO_V8]: Failed to run session" << std::endl;
        return results;
    }

    float maxConfidence = 0;
    int maxIndex = -1;

    for (int i = 0; i < res.size(); i++) 
    {
        auto probs = res.at(i);
        if (probs.confidence > maxConfidence) 
        {
            maxConfidence = probs.confidence;
            maxIndex = i;
        }
    }

    if (maxIndex != -1) {
        auto max_probs = res.at(maxIndex);
        int predict_label = max_probs.classId;
        auto predict_name = yoloClassifier->classes[predict_label];
        float confidence = max_probs.confidence;
        max_probs.className = predict_name;
        results.push_back(max_probs);
    }


    return results;
}


void TestClassification() {
    std::string modelPath = projectRoot / "models/yibo_train_cls_best.onnx";
    std::string yamlPath = projectRoot / "configs/classnames.yaml";
    std::string imagePath = projectRoot / "images/14.jpg";
    cv::Size imageSize(416, 416);
    bool useGPU = true;

    std::cout << "[YOLO_V8]: Infer image : " << imagePath << std::endl;

    std::vector<DL_RESULT> results = ClassifyImage(imagePath, modelPath, yamlPath, imageSize, useGPU);
    for (const auto& result : results) {
        std::cout << "[YOLO_V8]: Class:" << result.className << ", Confidence: " << result.confidence << std::endl;
    }

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "[YOLO_V8]: Failed to load image" << std::endl;
        return;
    }

    for (const auto& result : results) {
        std::string text = result.className + " " + std::to_string(result.confidence);
        cv::putText(image, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }

    std::filesystem::path outputPath = projectRoot / "output/cls_result.jpg";
    cv::imwrite(outputPath.string(), image);
    std::cout << "[YOLO_V8]: Result image saved at: " << outputPath << std::endl;
    

}



void TestDetectionOrSegmentation() {
    std::string model = "yolov8l-seg.onnx"; // 可以通过修改模型名称后缀来选择检测或分割
    // std::string model = "yolov8l.onnx";
    std::string modelPath = projectRoot / "models" / model;
    std::string yamlPath = projectRoot / "configs/coco.yaml";
    std::string imagePath = projectRoot / "images/17.jpg";
    cv::Size imageSize(640, 640); 
    float rectConfidenceThreshold = 0.3;
    float iouThreshold = 0.45;
    bool useGPU = true;
    std::cout << "[YOLO_V8]: Infer image: " << imagePath << std::endl;
    std::vector<DL_RESULT> results = DetectImage(imagePath, modelPath, yamlPath, imageSize, rectConfidenceThreshold, iouThreshold, useGPU);

    for (const auto& result : results) {
        std::cout << "[YOLO_V8]: Class:" << result.className 
        << ", Confidence:" << result.confidence 
        << ", Bounding Box:" << result.box << std::endl;
    }

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "[YOLO_V8]: Failed to load image" << std::endl;
        return;
    }
    // YOLO_V8 yoloDetector;
    // yoloDetector.DrawPred(image, results);
    int detections = results.size();
    std::cout << "[YOLO_V8]: Number of detections: " << detections << std::endl;
    cv::Mat mask = image.clone();
    for (int i = 0; i < detections; ++i)
    {
        DL_RESULT detection = results[i];
        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;
        // Detection box
        cv::rectangle(image, box, color, 2);
        mask(detection.box).setTo(color, detection.boxMask);
        // Detection box text
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(image, textBox, color, cv::FILLED);
        cv::putText(image, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    // Detection mask
    if (model.find("seg") != std::string::npos) {
        cv::addWeighted(image, 0.5, mask, 0.5, 0, image); //将mask加原图上
        std::filesystem::path outputPath = projectRoot / "output/seg_result.jpg";
        cv::imwrite(outputPath, image);
        std::cout << "[YOLO_V8]: Result image saved at: " << outputPath << std::endl;

    }
    else {
        std::filesystem::path outputPath = projectRoot / "output/det_result.jpg";

        cv::imwrite(outputPath, image);
        std::cout << "[YOLO_V8]: Result image saved at: " << outputPath << std::endl;


    }
}

int main() {
    // TestDetectionOrSegmentation();
    TestClassification();
    return 0;
}
