#define _CRT_SECURE_NO_WARNINGS

#include "inference.h"
#include <fstream>
#include <regex>
#include <random>
#define benchmark
using namespace std;

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


template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < imgHeight; h++) {
            for (int w = 0; w < imgWidth; w++) {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}


char* YOLO_V8::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg) {
    cv::Mat img = iImg.clone();
    cv::resize(iImg, oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
    if (img.channels() == 1) {
        cv::cvtColor(oImg, oImg, cv::COLOR_GRAY2BGR);
    }
    cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    return RET_OK;
}


void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape=cv::Size(640, 640),
    bool autoShape=false, bool scaleFill=false, bool scaleUp=true, int stride=32, const cv::Scalar& color=cv::Scalar(114, 114, 114))
{
    if (false) {
        int maxLen = MAX(image.rows, image.cols);
        outImage = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
        image.copyTo(outImage(cv::Rect(0, 0, image.cols, image.rows)));
        params[0] = 1;
        params[1] = 1;
        params[3] = 0;
        params[2] = 0;
    }

    cv::Size shape = image.size();
    float r = min((float)newShape.height / (float)shape.height,
        (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = min(r, 1.0f);

    float ratio[2]{ r, r };
    int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

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


void GetMask(const int* const _seg_params, const float& rectConfidenceThreshold, 
const cv::Mat& maskProposals, const cv::Mat& mask_protos, 
const cv::Vec4d& params, const cv::Size& srcImgShape, std::vector<DL_RESULT>& output) {
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
        //sigmoid
        cv::exp(-maskChannels[i], dest);
        dest = 1.0 / (1.0 + dest);
        cv::Rect roi(int(params[2] / _netWidth * _segWidth), int(params[3] / _netHeight * _segHeight), int(_segWidth - params[2] / 2), int(_segHeight - params[3] / 2));
        dest = dest(roi);
        cv::resize(dest, mask, srcImgShape, cv::INTER_NEAREST);
        //crop
        cv::Rect temp_rect = output[i].box;
        mask = mask(temp_rect) > rectConfidenceThreshold;
        output[i].boxMask = mask;
    }
}


void YOLO_V8::DrawPred(cv::Mat& img, std::vector<DL_RESULT>& result) {
    std::filesystem::path projectRoot = std::filesystem::current_path().parent_path();

    int detections = result.size();
    cout << "Number of detections:" << detections << endl;
    cv::Mat mask = img.clone();
    for (int i = 0; i < detections; ++i)
    {
        DL_RESULT detection = result[i];
        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;
        // Detection box
        cv::rectangle(img, box, color, 2);
        mask(detection.box).setTo(color, detection.boxMask);
        // Detection box text
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(img, textBox, color, cv::FILLED);
        cv::putText(img, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    // Detection mask
    if(runSegmentation) cv::addWeighted(img, 0.5, mask, 0.5, 0, img); //将mask加在原图上面
    cv::imwrite(projectRoot / "output/out.jpg", img);
}

 
char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
    char* Ret = RET_OK;
    
    rectConfidenceThreshold = iParams.rectConfidenceThreshold;
    iouThreshold = iParams.iouThreshold;
    imgSize = iParams.imgSize;
    modelType = iParams.modelType;
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolov8Inference");
    Ort::SessionOptions sessionOption;
    if (iParams.cudaEnable) {
        cudaEnable = iParams.cudaEnable;
        OrtCUDAProviderOptions cudaOption;
        cudaOption.device_id = 0;
        sessionOption.AppendExecutionProvider_CUDA(cudaOption);
    }
    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
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
    size_t inputNodesNum = session->GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++) {
        Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
        char* temp_buf = new char[50];
        strcpy(temp_buf, input_node_name.get());
        inputNodeNames.push_back(temp_buf);
    }
    size_t OutputNodesNum = session->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++) {
        Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
        char* temp_buf = new char[10];
        strcpy(temp_buf, output_node_name.get());
        outputNodeNames.push_back(temp_buf);
    }
    if (outputNodeNames.size() == 2) runSegmentation = true;
    options = Ort::RunOptions{ nullptr };
    WarmUpSession();
    return RET_OK;
}


char* YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
#ifdef benchmark
    clock_t starttime_1 = clock();
#endif 
    char* Ret = RET_OK;
    cv::Mat processedImg;
    cv::Vec4d params;
    //resize图片尺寸，PreProcess是直接resize，LetterBox有padding操作
    //PreProcess(iImg, imgSize, processedImg);
    LetterBox(iImg, processedImg, params, cv::Size(imgSize.at(1), imgSize.at(0)));
    if (modelType < 4) {
        float* blob = new float[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        TensorProcess(starttime_1, params,iImg, blob, inputNodeDims, oResult);
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
char* YOLO_V8::TensorProcess(clock_t& starttime_1, cv::Vec4d& params, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims, std::vector<DL_RESULT>& oResult) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type> (
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), 
        blob, 
        3 * imgSize.at(0) * imgSize.at(1), 
        inputNodeDims.data(), 
        inputNodeDims.size()
        );
#ifdef benchmark
    clock_t starttime_2 = clock();
#endif 
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),outputNodeNames.size());
#ifdef benchmark
    clock_t starttime_3 = clock();
#endif 
    std::vector<int64_t> _outputTensorShape;
    _outputTensorShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    auto output = outputTensor[0].GetTensorMutableData<typename std::remove_pointer<N>::type>();
    delete blob;

    switch (modelType) {
    case YOLO_DET_SEG_V8: {
        // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
        // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
        // yolov5
        int dimensions = _outputTensorShape[1];
        int rows = _outputTensorShape[2];
        cv::Mat rowData(dimensions, rows, CV_32F, output);
        // yolov8
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
            // int _seg_params[5] = {_segChannels, _segWidth, _segHeight, inputNodeDims[2], inputNodeDims[3]};
            // std::cout << inputNodeDims[2] << std::endl;
            // std::cout << imgSize.at(0) << std::endl;
            int _seg_params[5] = {_segChannels, _segWidth, _segHeight, imgSize.at(0), imgSize.at(1) };
            cv::Mat mask_protos = cv::Mat(mask);
            GetMask(_seg_params, rectConfidenceThreshold, mask_proposals, mask_protos, params, iImg.size(), oResult);
        }

#ifdef benchmark
    clock_t starttime_4 = clock();
    double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
    double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
    double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
    if (cudaEnable) {
        cout << "[YOLO_V8(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time 
        << "ms inference, " << post_process_time << "ms post-process." << endl;
    }
    else {
        cout << "[YOLO_V8(CPU)]: " << pre_process_time << "ms pre-process, " << process_time 
        << "ms inference, " << post_process_time << "ms post-process." << endl;
    }
#endif
        break;
    }
    case YOLO_CLS_V8:
    {
        DL_RESULT result;
        for (int i = 0; i < this->classes.size(); i++) {
            result.classId = i;
            result.confidence = output[i];
            oResult.push_back(result);
        }

#ifdef benchmark
    clock_t starttime_4 = clock();
    double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
    double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
    double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;

    if (cudaEnable) {
        cout << "[YOLO_V8(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << endl;
    }
    else {
        cout << "[YOLO_V8(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << endl;
    }
#endif
        break;
    }
    default:
        cout << "[YOLO_V8]: " << "Not support model type." << endl;
    }
    return RET_OK;

}


char* YOLO_V8::WarmUpSession() {
    clock_t starttime_1 = clock();
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    // PreProcess(iImg, imgSize, processedImg);
    cv::Vec4d params;
    //resize图片尺寸，PreProcess是直接resize，LetterBox有padding操作
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
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable) {
            cout << "[YOLO_V8(CUDA)]: Session has Warmed Up" << endl;
            cout << "[YOLO_V8(CUDA)]: " << "CUDA warm-up cost: " << post_process_time << " ms." << endl;
        }
        else{
            cout << "[YOLO_V8(CPU)]: Session has Warmed Up" << endl;
        }
    }
    else {
        #ifdef USE_CUDA
                half* blob = new half[iImg.total() * 3];
                BlobFromImage(processedImg, blob);
                std::vector<int64_t> YOLO_input_node_dims = { 1,3,imgSize.at(0),imgSize.at(1) };
                Ort::Value input_tensor = Ort::Value::CreateTensor<half>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
                auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
                delete[] blob;
                clock_t starttime_4 = clock();
                double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
                if (cudaEnable)
                {
                    cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << endl;
                }
        #endif
    }
    return RET_OK;
}


int YOLO_V8::ReadClassNames(const std::string& yamlPath, std::vector<std::string>& classNames) {

    std::ifstream file(yamlPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open YAML file" << std::endl;
        return 1;
    }

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
        }
        else if (start > 0 && lines[i].find(':') == std::string::npos)
        {
            end = i;
            break;
        }
    }

    for (std::size_t i = start; i < end; i++)
    {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':');
        std::getline(ss, name);
        classNames.push_back(name);
    }
    return 0;
}
