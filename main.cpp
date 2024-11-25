#include <iostream>
#include <getopt.h>
#include <filesystem>
#include <opencv2/core.hpp>
#include <fstream>
#include "inference.h"
#include <chrono>
namespace fs = std::filesystem;


void test(const std::string& directoryPath) {
    DL_INIT_PARAM params;
    std::filesystem::path projectRoot = std::filesystem::current_path().parent_path();
    std::string labelPath = projectRoot / "huachuan/class_names_list.txt"; 
    params.modelPath = projectRoot / "huachuan/best.onnx";
    params.modelType = YOLO_DET_SEG_V8;
    params.imgSize = { 312, 312 }; // VAlgo需要暴露推理时的图片尺寸
    params.rectConfidenceThreshold = 0.7;
    params.iouThreshold = 0.0001;    
    params.cudaEnable = true;
    auto starttime_1 =  std::chrono::high_resolution_clock::now();

    std::unique_ptr<YOLO_V8> yolo(new YOLO_V8);
    yolo->CreateSession(params);
    auto starttime_3 =  std::chrono::high_resolution_clock::now();
    auto duration_ms4 = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_3 - starttime_1).count();
    std::cout << "[YOLO_V8]: 模型预热时间: " << duration_ms4 << "ms" << std::endl;

    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (fs::is_regular_file(entry.path()) && entry.path().extension() == ".jpg") {
            std::string imagePath = entry.path().string();
            std::string imageName = entry.path().filename().stem().string();
            std::cout << "\n[YOLO_V8]: 正在推理图片: " << imageName << ".jpg" << std::endl;
            auto starttime_2 =  std::chrono::high_resolution_clock::now();
            auto results = yolo->Inference(imagePath, labelPath);
            auto starttime_4 =  std::chrono::high_resolution_clock::now();
            auto duration_ms3 = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_4 - starttime_2).count();
            
            cv::Mat image = cv::imread(imagePath);
            if (image.empty()) {
                std::cerr << "[YOLO_V8]: Failed to load image" << std::endl;
                return;
            }

            if (params.modelType == YOLO_DET_SEG_V8) {

                for (const auto& result : results) {
                    std::cout << "[YOLO_V8]: 类别: " << result.className 
                    << " , 置信度: " << result.confidence 
                    << std::endl;
                }
                int detections = results.size();
                std::cout << "[YOLO_V8]: 检测数量: " << detections << std::endl;
                std::cout << "[YOLO_V8]: 总推理时间:" <<duration_ms3 << "ms" << std::endl;   

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
                std::string outputDirectory = "/home/yibo/git_dir/yolov8_onnxruntime_cpp/det_seg_output/";

                if (!fs::exists(outputDirectory))
                    fs::create_directory(outputDirectory);

                std::filesystem::path outputImagePath = outputDirectory + imageName + "_result.jpg";

                cv::imwrite(outputImagePath.string(), image);
            }
            else { // YOLO_CLS_V8

                for (const auto& result : results) {
                    std::cout << "[YOLO_V8]: 类别: " << result.className << ", 置信度: " << result.confidence << std::endl;
                    std::string text = result.className + " " + std::to_string(result.confidence).substr(0, 4);
                    cv::putText(image, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

                }
                std::string outputDirectory = "/home/yibo/git_dir/yolov8_onnxruntime_cpp/cls_output/";
                if (!fs::exists(outputDirectory))
                    fs::create_directory(outputDirectory);

                std::filesystem::create_directory(outputDirectory);

                std::filesystem::path outputImagePath = outputDirectory + imageName + "_result.jpg";
                cv::imwrite(outputImagePath.string(), image);
            }

            }
        }
    }


int main() {
    std::string dir = "/home/yibo/git_dir/yolov8_onnxruntime_cpp/images/HandianLocation_HC";
    test(dir);
    return 0;
}


