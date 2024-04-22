#include <iostream>
#include <getopt.h>
#include <filesystem>
#include <opencv2/core.hpp>
#include <fstream>
#include "inference.h"

namespace fs = std::filesystem;

void Test(std::string imagePath) {
    std::filesystem::path projectRoot = std::filesystem::current_path().parent_path();
    std::string model = "best.onnx"; 
    std::string modelPath = projectRoot / "models" / model;
    // std::string imagePath = projectRoot / "images/origin_data/1782-2743-03-JKE002-07-01-01-01-20240413001249876.jpg";
    std::string yamlPath = projectRoot / "configs/jke_zhuanzi.yaml"; // detect or segment choose it
    // std::string yamlPath = projectRoot / "configs/classnames.yaml"; //classify choose it
    cv::Size imageSize(768, 768); 
    MODEL_TYPE modelType = YOLO_DET_SEG_V8; // YOLO_CLS_V8
    float rectConfidenceThreshold = 0.1;
    float iouThreshold = 0.0001;
    bool useGPU = false;

    std::cout << "[YOLO_V8]: Infering image: " << imagePath << std::endl;
    std::cout << "[YOLO_V8]: Infer model: " << model << std::endl;
    std::unique_ptr<YOLO_V8> yolo(new YOLO_V8);

    auto results = yolo->Inference(imagePath, modelType, modelPath, yamlPath, imageSize, rectConfidenceThreshold, iouThreshold, useGPU);

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "[YOLO_V8]: Failed to load image" << std::endl;
        return;
    }

    if (modelType == YOLO_DET_SEG_V8) {

        for (const auto& result : results) {
            std::cout << "[YOLO_V8]: Class:" << result.className 
            << ", Confidence:" << result.confidence 
            << ", Bounding Box:" << result.box << std::endl;
        }

        int detections = results.size();
        std::cout << "[YOLO_V8]: Number of detections: " << detections << std::endl;
        cv::Mat mask = image.clone();
        for (int i = 0; i < detections; ++i)
        {
            DL_RESULT detection = results[i];
            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;
            // Detection box
            cv::rectangle(image, box, color, 1);
            // Detection box text
            std::string classString = detection.className + " " + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
            cv::rectangle(image, textBox, color, cv::FILLED);
            cv::putText(image, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2);
            // Segmentation mask contours
            std::vector<std::vector<cv::Point>> contours = detection.contours;
            cv::drawContours(image(box), contours, -1, cv::Scalar(0, 255, 0), 2);
        }
        if (model.find("seg") != std::string::npos) {
            std::filesystem::path outputPath = projectRoot / "output/seg_result.jpg";
            cv::imwrite(outputPath, image);
            std::cout << "[YOLO_V8(SEG)]: Result image saved at: " << outputPath << std::endl;

        }
        else {
            std::filesystem::path outputPath = projectRoot / "output/det_result.jpg";
            cv::imwrite(outputPath, image);
            std::cout << "[YOLO_V8(DET)]: Result image saved at: " << outputPath << std::endl;
        }
    }
    else {

        for (const auto& result : results) {
            std::cout << "[YOLO_V8]: Class:" << result.className << ", Confidence: " << result.confidence << std::endl;
            std::string text = result.className + " " + std::to_string(result.confidence).substr(0, 4);
            cv::putText(image, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);

        }

        std::filesystem::path outputPath = projectRoot / "output/cls_result.jpg";
        cv::imwrite(outputPath.string(), image);
        std::cout << "[YOLO_V8(CLS)]: Result image saved at: " << outputPath << std::endl;
    }

}
void Inference(const std::string& directoryPath) {
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (fs::is_regular_file(entry.path()) && entry.path().extension() == ".jpg") {
            std::string imagePath = entry.path().string();
            std::cout << "Infering image: " << imagePath << std::endl;
            Test(imagePath);
        }
    }
}




int main() {
    std::string directoryPath = "/home/yibo/yolov8_onnxruntime_cpp/images/origin_data";
    Inference(directoryPath);
    return 0;
}


