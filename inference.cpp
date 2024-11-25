#define _CRT_SECURE_NO_WARNINGS
#include "inference.h"
#include <fstream>
#include <regex>
#include <random>
#define benchmark

YOLO_V8::YOLO_V8() { 

}

YOLO_V8::~YOLO_V8() { 
    delete session;
}

#ifdef USE_CUDA
namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif


std::vector<DL_RESULT> YOLO_V8::Inference(const std::string& imagePath, const std::string& txtPath) {

    std::vector<std::string> classNames;
    std::vector<DL_RESULT> results;

    if (ReadClassNames(txtPath, classNames) != 0) {
        std::cerr << "[YOLO_V8]: Failed to read class names" << std::endl;
        return results;
    }
    classes = std::move(classNames);

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "[YOLO_V8]: Failed to load image" << std::endl;
        return results;
    }

    auto starttime_4 = std::chrono::high_resolution_clock::now();
    
    std::vector<DL_RESULT> res;
    if (RunSession(image, res) != 0) {
        std::cerr << "[YOLO_V8]: Failed to run session" << std::endl;
        return results;
    }
    auto starttime_3 = std::chrono::high_resolution_clock::now();
    auto duration_ms3 = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_3 - starttime_4).count();
    std::cout << "[YOLO_V8]: RunSession时间: " << duration_ms3 << " ms" << std::endl;

    if (modelType == YOLO_CLS_V8 ) {
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
            auto predict_name = classes[predict_label];
            float confidence = max_probs.confidence;
            max_probs.className = predict_name;
            results.push_back(max_probs);
        }
    }
    else {
        for (const auto& result : res) {
            results.push_back(result);
        }
    }

    return results;
}



template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < imgHeight; h++) {
            for (int w = 0; w < imgWidth; w++) 
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}


char* YOLO_V8::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg) {
    if (iImg.channels() == 3) {
            oImg = iImg.clone();
            cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
        }
        else {
            cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
        }


    int h = iImg.rows;
    int w = iImg.cols;
    int m = std::min(h, w);
    int top = (h - m) / 2;
    int left = (w - m) / 2;

    cv::resize(oImg(cv::Rect(left, top, m, m)), oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
    return RET_OK;
}


void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, 
    const cv::Size& newShape=cv::Size(640, 640),
    bool autoShape=true, 
    bool scaleFill=false, 
    bool scaleUp=true, 
    int stride=32, 
    const cv::Scalar& color=cv::Scalar(114, 114, 114))
{
    // if (false) {
    //     int maxLen = MAX(image.rows, image.cols);
    //     outImage = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
    //     image.copyTo(outImage(cv::Rect(0, 0, image.cols, image.rows)));
    //     params[0] = 1;
    //     params[1] = 1;
    //     params[3] = 0;
    //     params[2] = 0;
    // }

    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                        (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int new_un_pad[2] = { (int)std::round((float)shape.width * r),
                            (int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

    if (autoShape) {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1]) {
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else {
        outImage = image.clone();
    }
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}


void GetMask(
    const int* const _seg_params, 
    const float& rectConfidenceThreshold, 
    const cv::Mat& maskProposals, 
    const cv::Mat& mask_protos, 
    const cv::Vec4d& params, 
    const cv::Size& srcImgShape, 
    std::vector<DL_RESULT>& output) 
    {
    int _segChannels = *_seg_params;
    int _segHeight = *(_seg_params + 1);
    int _segWidth = *(_seg_params + 2);
    int _netHeight = *(_seg_params + 3);
    int _netWidth = *(_seg_params + 4);
    
    cv::Mat protos = mask_protos.reshape(0, { _segChannels,_segWidth * _segHeight });
    cv::Mat matmulRes = (maskProposals * protos).t();
    cv::Mat masks = matmulRes.reshape(output.size(), { _segHeight,_segWidth });
    std::vector<cv::Mat> maskChannels;
    split(masks, maskChannels);
    for (int i = 0; i < output.size(); ++i) {
        cv::Mat dest, mask;
        cv::exp(-maskChannels[i], dest);
        dest = 1.0 / (1.0 + dest);
        cv::Rect roi(
            int(params[2] / _netWidth * _segWidth), 
            int(params[3] / _netHeight * _segHeight), 
            int(_segWidth - params[2] / 2), 
            int(_segHeight - params[3] / 2));
        dest = dest(roi);
        cv::resize(dest, mask, srcImgShape, cv::INTER_NEAREST);
        cv::Rect temp_rect = output[i].box;
        mask = mask(temp_rect) > 0.5f; // 固定mask阈值，实测0.5效果最好
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        double maxArea = -1;
        int maxAreaIdx = -1;
        for (int j = 0; j < contours.size(); ++j) {
            double area = cv::contourArea(contours[j]);
            if (area > maxArea) {
                maxArea = area;
                maxAreaIdx = j;
            }
        }
        if (maxAreaIdx != -1) {
            std::vector<std::vector<cv::Point>> filteredContours;
            filteredContours.push_back(contours[maxAreaIdx]);
            output[i].contours = filteredContours;
        } else {
            output[i].contours.clear();
        }
    }
}


 
char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
    rectConfidenceThreshold = iParams.rectConfidenceThreshold;
    iouThreshold = iParams.iouThreshold;
    imgSize = iParams.imgSize;
    modelType = iParams.modelType;
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8ONNXRuntimeInference");
    Ort::SessionOptions sessionOption;

    if (iParams.cudaEnable) {
        cudaEnable = iParams.cudaEnable;
        auto providers = Ort::GetAvailableProviders();
        auto cudaAvailable = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider");
        OrtCUDAProviderOptions cudaOption;

        if (cudaAvailable != providers.end())
        {
            std::cout << "Inference device: GPU" << std::endl;
            cudaOption.device_id = 0;
            cudaOption.arena_extend_strategy = 0;
            cudaOption.do_copy_in_default_stream = 1;
            // cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchDefault;            
            // cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
            cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchHeuristic;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        else if (cudaAvailable == providers.end())
        {
            std::cout << "GPU is not supported. Fallback to CPU." << std::endl;
            std::cout << "Inference device: CPU" << std::endl;
        }
    }

    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOption.SetIntraOpNumThreads(0);
    // sessionOption.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    // sessionOption.SetInterOpNumThreads(0);
    sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

#ifdef _WIN32
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
        wide_cstr[ModelPathSize] = L'\0';
        const wchar_t* modelPath = wide_cstr;
#else
        const char* modelPath = iParams.modelPath.c_str();
#endif

    session = new Ort::Session(env, modelPath, sessionOption);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t InputNodesNum = session->GetInputCount();

    for (size_t i = 0; i < InputNodesNum; i++) {
        Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
        this->inputNodeNames.push_back(input_node_name.get());
        input_names_ptr.push_back(std::move(input_node_name));

        Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(i);
        std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        this->inputShapes.push_back(inputTensorShape);
        // this->isDynamicInputShape = false;
        // // checking if width and height are dynamic
        // if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
        // {
        //     std::cout << "Dynamic input shape" << std::endl;
        //     this->isDynamicInputShape = true;
        // }
    }
    size_t OutputNodesNum = session->GetOutputCount();
    if (OutputNodesNum > 1)
    {
        this->runSegmentation = true;        
        std::cout << "Instance Segmentation" << std::endl;
    }
    else
        std::cout << "Object Detection" << std::endl;

    for (size_t i = 0; i < OutputNodesNum; i++) {
        Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
        
        this->outputNodeNames.push_back(output_node_name.get());
        output_names_ptr.push_back(std::move(output_node_name));

        Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(i);
        std::vector<int64_t> outputTensorShape = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        this->outputShapes.push_back(outputTensorShape);
    }    

    // for (const char *x : this->inputNodeNames)
    // {
    //     std::cout << x << std::endl;
    // }
    // for (const char *x : this->outputNodeNames)
    // {
    //     std::cout << x << std::endl;
    // }

    options = Ort::RunOptions{ nullptr };
    WarmUpSession();
    return RET_OK;
}


char* YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {

#ifdef benchmark
    auto starttime_1 =  std::chrono::high_resolution_clock::now();
#endif 
    char* Ret = RET_OK;
    cv::Mat processedImg;
    cv::Vec4d params;
    //resize图片尺寸，PreProcess是resize+centercrop，LetterBox有padding
    switch (modelType) {
    case YOLO_DET_SEG_V8: {
        LetterBox(iImg, processedImg, params, cv::Size(imgSize.at(1), imgSize.at(0)));
        break;
    }
    case YOLO_CLS_V8: {
        PreProcess(iImg, imgSize, processedImg);
        break;
    }
    }        
    if (modelType < 4) {
        float* blob = new float[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        TensorProcess(starttime_1, params, iImg, blob, inputNodeDims, oResult);
    }
    else {
#ifdef USE_CUDA
        half* blob = new half[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        TensorProcess(starttime_1, params, iImg, blob, inputNodeDims, oResult);
#endif
    }
        return Ret;
    }
    


template<typename N>
char* YOLO_V8::TensorProcess(std::chrono::_V2::system_clock::time_point& starttime_1, 
cv::Vec4d& params, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims, std::vector<DL_RESULT>& oResult) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type> (
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), 
        blob, 
        3 * imgSize.at(0) * imgSize.at(1), 
        inputNodeDims.data(), 
        inputNodeDims.size()
        );
#ifdef benchmark
    auto starttime_2 =  std::chrono::high_resolution_clock::now();
#endif 
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),outputNodeNames.size());
#ifdef benchmark
    auto starttime_3 =  std::chrono::high_resolution_clock::now();
#endif 
    std::vector<int64_t> _outputTensorShape;
    _outputTensorShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    auto output = outputTensor[0].GetTensorMutableData<typename std::remove_pointer<N>::type>();
    delete[] blob;

    switch (modelType) {
    case YOLO_DET_SEG_V8: {
        // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
        std::cout << "---------------------YOLO_DET_SEG_V8---------------------" << std::endl;
        int dimensions = _outputTensorShape[1];
        int rows = _outputTensorShape[2];
        cv::Mat rowData(dimensions, rows, CV_32F, output);
        if (rows > dimensions) { 
            dimensions = _outputTensorShape[2];
            rows = _outputTensorShape[1];
            rowData = rowData.t();
        }
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<float>> picked_proposals;
       
        float* data = (float*)rowData.data;

        for (int i = 0; i < dimensions; ++i) {
            float* classesScores = data + 4;
            cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
            cv::Point class_id;
            double maxClassScore;
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > rectConfidenceThreshold) {
                if (runSegmentation) {
                    int _segChannels = outputTensor[1].GetTensorTypeAndShapeInfo().GetShape()[1];
                    std::vector<float> temp_proto(data + classes.size() + 4, data + classes.size() + 4 + _segChannels);
                    picked_proposals.push_back(temp_proto);
                }
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
                float x = (data[0] - params[2]) / params[0];
                float y = (data[1] - params[3]) / params[1];
                float w = data[2] / params[0];
                float h = data[3] / params[1];
                int left = MAX(round(x - 0.5 * w + 0.5), 0);
                int top = MAX(round(y - 0.5 * h + 0.5), 0);
                if ((left + w) > iImg.cols) { w = iImg.cols - left; }
                if ((top + h) > iImg.rows) { h = iImg.rows - top; }
                boxes.emplace_back(cv::Rect(left, top, int(w), int(h)));
            }
            data += rows;
        }
        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
        std::vector<std::vector<float>> temp_mask_proposals;
        for (int i = 0; i < nmsResult.size(); ++i) {
            int idx = nmsResult[i];
            DL_RESULT result;
            result.classId = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            result.className = classes[result.classId];
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(100, 255);
            result.color = cv::Scalar(dis(gen),dis(gen),dis(gen));
            if (result.box.width != 0 && result.box.height != 0) oResult.push_back(result);
            if (runSegmentation) temp_mask_proposals.push_back(picked_proposals[idx]);
        }
        if (!boxes.empty()) {
            if (runSegmentation) {
                cv::Mat mask_proposals;
                for (int i = 0; i < temp_mask_proposals.size(); ++i)
                    mask_proposals.push_back(cv::Mat(temp_mask_proposals[i]).t());
                std::vector<int64_t> _outputMaskTensorShape;
                _outputMaskTensorShape = outputTensor[1].GetTensorTypeAndShapeInfo().GetShape();
                int _segChannels = _outputMaskTensorShape[1];
                int _segWidth = _outputMaskTensorShape[2];
                int _segHeight = _outputMaskTensorShape[3];
                float* pdata = outputTensor[1].GetTensorMutableData<float>();
                std::vector<float> mask(pdata, pdata + _segChannels * _segWidth * _segHeight);
                int _seg_params[5] = {_segChannels, _segWidth, _segHeight, imgSize.at(0), imgSize.at(1) };
                cv::Mat mask_protos = cv::Mat(mask);
                GetMask(_seg_params, rectConfidenceThreshold, mask_proposals, mask_protos, params, iImg.size(), oResult);
            }
        }

#ifdef benchmark
    auto starttime_4 =  std::chrono::high_resolution_clock::now();

    double pre_process_time = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_2 - starttime_1).count();
    double process_time = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_3 - starttime_2).count();
    double post_process_time = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_4 - starttime_3).count();
    double total_time = pre_process_time + process_time + post_process_time;
    if (cudaEnable) {
        std::cout << "[YOLO_V8(CUDA)]: 前处理 " << pre_process_time << " ms, 推理 " << process_time 
            << " ms, 后处理 " << post_process_time << " ms. 总共耗时 " << total_time << " ms." << std::endl;
    }
    else {
        std::cout << "[YOLO_V8(CPU)]: 前处理 " << pre_process_time << " ms, 推理 " << process_time 
            << " ms, 后处理 " << post_process_time << " ms. 总共耗时 " << total_time << " ms." << std::endl;
    }
#endif
        break;
    }
    case YOLO_CLS_V8:
    {
        cv::Mat rawData;
        rawData = cv::Mat(1, this->classes.size(), CV_32F, output);
        
        float *data = (float *) rawData.data;

        DL_RESULT result;
        for (int i = 0; i < this->classes.size(); i++)
        {
            result.classId = i;
            result.confidence = data[i];
            oResult.push_back(result);
        }

#ifdef benchmark
    auto starttime_4 =  std::chrono::high_resolution_clock::now();
    double pre_process_time = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_2 - starttime_1).count();
    double process_time = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_3 - starttime_2).count();
    double post_process_time = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_4 - starttime_3).count();
    double total_time = pre_process_time + process_time + post_process_time;
    if (cudaEnable) {
        std::cout << "[YOLO_V8(CUDA)]: 前处理 " << pre_process_time << " ms, 推理 " << process_time 
            << " ms, 后处理 " << post_process_time << " ms. 总共耗时 " << total_time << " ms." << std::endl;
    }
    else {
        std::cout << "[YOLO_V8(CPU)]: 前处理 " << pre_process_time << " ms, 推理 " << process_time 
            << " ms, 后处理 " << post_process_time << " ms. 总共耗时 " << total_time << " ms." << std::endl;
    }
#endif
        break;
    }
    default:
        std::cout << "[YOLO_V8]: " << "不支持的模型类型." << std::endl;
    }
    return RET_OK;

}


char* YOLO_V8::WarmUpSession() {
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    cv::Vec4d params;
    LetterBox(iImg, processedImg, params, cv::Size(imgSize.at(1), imgSize.at(0)));
    if (modelType < 4) {
        float* blob = new float[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
            YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
        delete[] blob;
    }
    else {
        #ifdef USE_CUDA
                half* blob = new half[iImg.total() * 3];
                BlobFromImage(processedImg, blob);
                std::vector<int64_t> YOLO_input_node_dims = { 1,3,imgSize.at(0),imgSize.at(1) };
                Ort::Value input_tensor = Ort::Value::CreateTensor<half>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
                auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
                delete[] blob;
        #endif
    }
    return RET_OK;
}


int YOLO_V8::ReadClassNames(const std::string& txtPath, std::vector<std::string>& classNames) {
    std::ifstream file(txtPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open TXT file" << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), line.end());

        if (!line.empty()) {
            classNames.push_back(line);
        }
    }

    file.close();
    return 0;
}