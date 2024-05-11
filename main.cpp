#include <iostream>
#include <getopt.h>
#include <filesystem>
#include <opencv2/core.hpp>
#include <fstream>
#include "inference.h"
#include <chrono>
namespace fs = std::filesystem;

void Test(std::string imagePath, std::string imageName) {
    std::filesystem::path projectRoot = std::filesystem::current_path().parent_path();
    std::string model = "best.onnx"; 
    std::string modelPath = projectRoot / "models" / model;
    std::string yamlPath = projectRoot / "configs/huachuan.yaml"; 
    cv::Size imageSize(512, 512); 
    MODEL_TYPE modelType = YOLO_DET_SEG_V8; 
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.45;
    bool useGPU = true;
    std::unique_ptr<YOLO_V8> yolo(new YOLO_V8);

    auto starttime_1 =  std::chrono::high_resolution_clock::now();
    auto results = yolo->Inference(imagePath, modelType, modelPath, yamlPath, imageSize, rectConfidenceThreshold, iouThreshold, useGPU);
    auto starttime_3 =  std::chrono::high_resolution_clock::now();
    auto duration_ms3 = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_3 - starttime_1).count();
    std::cout << "Inference总推理时间:" <<duration_ms3 << "ms" << std::endl;   


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
        for (int i = 0; i < detections; ++i)
        {
            DL_RESULT detection = results[i];
            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;
            cv::rectangle(image, box, color, 1);
            std::string classString = detection.className + " " + std::to_string(detection.confidence).substr(0, 4);       
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
            cv::rectangle(image, textBox, color, cv::FILLED);
            cv::putText(image, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2);
            if (!detection.contours.empty()) {
                std::vector<std::vector<cv::Point>> contours = detection.contours;
                cv::drawContours(image(box), contours, -1, cv::Scalar(0, 255, 0), 2);
            }
        }
        std::string outputDirectory = "/home/yibo/git_dir/yolov8_onnxruntime_cpp/output/";

        if (!fs::exists(outputDirectory))
            fs::create_directory(outputDirectory);

        std::filesystem::path outputImagePath = outputDirectory + imageName + "_result.jpg";

        cv::imwrite(outputImagePath.string(), image);
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
            std::string imageName = entry.path().filename().stem().string();

            std::cout << "[YOLO_V8]: Infering image: " << imageName << ".jpg" << std::endl;
            Test(imagePath, imageName);
            }
        }
    }




int main() {
    std::string directoryPath = "/home/yibo/git_dir/yolov8_onnxruntime_cpp/images/test/";
    Inference(directoryPath);
    return 0;
}


