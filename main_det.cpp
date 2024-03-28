#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <fstream>
#include "inference.h"

std::filesystem::path projectRoot = std::filesystem::current_path().parent_path();

std::vector<DL_RESULT> DetectImage(   
    const std::string& imagePath,                       
    const std::string& modelPath = projectRoot / "models/yolov8s.onnx", 
    const std::string& yamlPath = projectRoot / "configs/coco.yaml",
    const cv::Size& imgSize = {640, 640}, 
    float rectConfidenceThreshold = 0.45,
    float iouThreshold = 0.5,
    bool useGPU = false)  
    {
    std::vector<DL_RESULT> results;
    DL_INIT_PARAM params{ modelPath, YOLO_DETECT_V8, {imgSize.width, imgSize.height}, 
    rectConfidenceThreshold, iouThreshold};
    params.cudaEnable = useGPU;
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


void TestDetection() {
    std::filesystem::path projectRoot = std::filesystem::current_path().parent_path();

    std::string modelPath = projectRoot / "models/yolov8l.onnx";
    std::string yamlPath = projectRoot / "configs/coco.yaml";
    std::string imagePath = projectRoot / "images/17.jpg";
    cv::Size imageSize(640, 640); 

    std::cout << "[YOLO_V8]: Infer image : " << imagePath << std::endl;

    std::vector<DL_RESULT> results = DetectImage(imagePath, modelPath, yamlPath, imageSize, 0.3, 0.45, false);

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

    int detections = results.size();
    std::cout << "Number of detections:" << detections << std::endl;
    cv::Mat mask = image.clone();
    for (int i = 0; i < detections; ++i) {
        DL_RESULT detection = results[i];
        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;
        cv::rectangle(image, box, color, 2);
        mask(detection.box).setTo(color, detection.boxMask);
        std::string classString = detection.className + ':' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(image, textBox, color, cv::FILLED);
        cv::putText(image, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    cv::imwrite(projectRoot / "output/out.jpg", image);
}

int main() {
    TestDetection();
    return 0;
}
