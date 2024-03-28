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


std::vector<DL_RESULT> ClassifyImage(   
    const std::string& imagePath,                       
    const std::string& modelPath = projectRoot / "models/best.onnx", 
    const std::string& yamlPath = projectRoot / "configs/classnames.yaml",
    const cv::Size& imgSize = {640, 640}, 
    bool useGPU = false)  
    {
    std::vector<DL_RESULT> results;
    DL_INIT_PARAM params{ modelPath, YOLO_CLS, {imgSize.width, imgSize.height}};

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
    std::string modelPath = projectRoot / "models/best.onnx";
    std::string yamlPath = projectRoot / "configs/classnames.yaml";
    std::string imagePath = projectRoot / "images/4.jpg";
    cv::Size imageSize(416, 416);
    std::cout << "[YOLO_V8]: Infer image : " << imagePath << std::endl;

    std::vector<DL_RESULT> results = ClassifyImage(imagePath, modelPath, yamlPath, imageSize, true);
    for (const auto& result : results) {
        std::cout << "[YOLO_V8]: Class:" << result.className << ", Confidence: " << result.confidence << std::endl;
    }

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "[YOLO_V8]: Failed to load image" << std::endl;
        return;
    }

    for (const auto& result : results) {
        std::string text = result.className + ": " + std::to_string(result.confidence);
        cv::putText(image, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }

    std::filesystem::path outputPath = projectRoot / "output/cls_result.jpg";
    cv::imwrite(outputPath.string(), image);
    std::cout << "[YOLO_V8]: Result image saved at: " << outputPath << std::endl;
    

}



void TestDetection() {
    std::string modelPath = projectRoot / "models/yolov8l.onnx";
    std::string yamlPath = projectRoot / "configs/coco.yaml";
    std::string imagePath = projectRoot / "images/17.jpg";
    cv::Size imageSize(640, 640); 

    std::cout << "[YOLO_V8]: Infer image : " << imagePath << std::endl;

    std::vector<DL_RESULT> results = DetectImage(imagePath, modelPath, yamlPath, imageSize, 0.3, 0.45, true);

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
    // TestClassification();
    return 0;
}
