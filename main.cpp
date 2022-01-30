#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/chrono.hpp>
#include "utils.h"
#include "detector.h"

/*
#include "cmdline.h"

struct Detection {
    cv::Rect box;
    float conf{};
    int classId{};
};

namespace utils {
std::wstring charToWstring(const char* str) {
    typedef std::codecvt_utf8<wchar_t> convert_type;
    std::wstring_convert<convert_type, wchar_t> converter;

    return converter.from_bytes(str);
}

std::vector<std::string> loadNames(const std::string& path) {
    // load class names
    std::vector<std::string> classNames;
    std::ifstream infile(path);
    if (infile.good()) {
        std::string line;
        while (getline(infile, line)) {
            classNames.emplace_back(line);
        }
        infile.close();
    } else {
        std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
    }

    return classNames;
}

void visualizeDetection(
    cv::Mat& image,
    std::vector<Detection>& detections,
    const std::vector<std::string>& classNames) {
        for (const Detection& detection : detections) {
            cv::rectangle(image, detection.box, cv::Scalar(229, 160, 21), 2);

            int x = detection.box.x;
            int y = detection.box.y;

            int conf = (int)(detection.conf * 100);
            int classId = detection.classId;
            std::string label = classNames[classId] + " 0." + std::to_string(conf);

            int baseline = 0;
            cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline);
            cv::rectangle(image, cv::Point(x, y - 25), cv::Point(x + size.width, y), cv::Scalar(229, 160, 21), -1);

            cv::putText(image, label, cv::Point(x, y - 3), cv::FONT_ITALIC, 0.8, cv::Scalar(255, 255, 255), 2);
        }
    }
} // namespace utils

class YOLOv5Detector {
public:
    explicit YOLOv5Detector(std::nullptr_t){};
    YOLOv5Detector(const std::string& modelPath, const bool& isGPU);

    std::vector<Detection> detect(cv::Mat& image);

private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};

    static void preprocessing(cv::Mat& image, float* blob);
    static std::vector<Detection> postprocessing(
        cv::Mat& image,
        std::vector<Ort::Value>& outputTensors
    );

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
};

YOLOv5Detector::YOLOv5Detector(const std::string& modelPath, const bool& isGPU = true) {
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (isGPU && (cudaAvailable == availableProviders.end())) {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    } else if (isGPU && (cudaAvailable != availableProviders.end())) {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    } else {
        std::cout << "Inference device: CPU" << std::endl;
    }

#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    inputNames.push_back(session.GetInputName(0, allocator));

    for (int i = 0; i < 3; ++i)
        outputNames.push_back(session.GetOutputName(i, allocator));
}

void YOLOv5Detector::preprocessing(cv::Mat& image, float* blob) {
    cv::Mat floatImage;
    cv::cvtColor(image, floatImage, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> chw(image.channels());

    floatImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    cv::Size imageSize{image.cols, image.rows};

    // hwc -> chw
    for (int i = 0; i < image.channels(); ++i) {
        chw[i] = cv::Mat(imageSize, CV_32FC1, blob + i * imageSize.width * imageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<Detection> YOLOv5Detector::postprocessing(cv::Mat& image, std::vector<Ort::Value>& outputTensors) {
    const auto* scoresTensor = outputTensors[0].GetTensorData<float>();
    const auto* classIdsTensor = outputTensors[1].GetTensorData<int64_t>();
    const auto* boxesTensor = outputTensors[2].GetTensorData<float>();

    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<Detection> detections;
    for (int i = 0; i < count; ++i) {
        Detection det;
        int x = (int)boxesTensor[i * 4];
        int y = (int)boxesTensor[i * 4 + 1];
        int width = (int)boxesTensor[i * 4 + 2] - x;
        int height = (int)boxesTensor[i * 4 + 3] - y;

        det.conf = scoresTensor[i];
        det.classId = (int)classIdsTensor[i];
        det.box = cv::Rect(x, y, width, height);
        detections.push_back(det);
    }

  return detections;
}

std::vector<Detection> YOLOv5Detector::detect(cv::Mat& image) {
    size_t inputTensorSize = image.rows * image.cols * image.channels();
    std::vector<int64_t> imageShape{image.channels(), image.rows, image.cols};
    auto* blob = new float[inputTensorSize];

    this->preprocessing(image, blob);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            imageShape.data(),
            imageShape.size()
        )
    );

    std::vector<Ort::Value> outputTensors = this->session.Run(
        Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, outputNames.data(), 3);

    std::vector<Detection> result = this->postprocessing(image, outputTensors);

    delete[] blob;

    return result;
}*/

namespace po = boost::program_options;
int main(int argc, char* argv[]) {
    /*
    cmdline::parser cmd;
    cmd.add<std::string>("model_path", 'm', "Path to onnx model.", true, "yolov5.onnx");
    cmd.add<std::string>("image", 'i', "Image source to be detected.", true, "bus.jpg");
    cmd.add<std::string>("class_names", 'c', "Path of dataset labels.", true, "coco.names");
    cmd.add("gpu", '\0', "Enable cuda device or cpu.");

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    std::string classNamesPath = cmd.get<std::string>("class_names");
    std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    std::string imagePath = cmd.get<std::string>("image");
    std::string modelPath = cmd.get<std::string>("model_path");

    if (classNames.empty()) {
        std::cout << "Empty class names file." << std::endl;
        return -1;
    }

    YOLOv5Detector detector{nullptr};
    try {
        detector = YOLOv5Detector(modelPath, isGPU);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(imagePath);

    std::vector<Detection> result = detector.detect(image);

    utils::visualizeDetection(image, result, classNames);

    cv::imshow("result", image);
    // cv::imwrite("result.jpg", image);
    cv::waitKey(0);

    return 0;
    */

    std::string filename, modelPath, classNamesPath;
    bool isGpu, useFilename;
    try {
        // define optional arguments
        po::options_description description("Lego-CNN YoloV5");
        description.add_options()
            ("help,h", "produces this help message")
            ("version,v", "get version information")
            ("gpu,g", "enable gpu")
            ("input-file,i", po::value<std::vector<std::string>>(), "input file to process")
            ("model-path,m", po::value<std::vector<std::string>>(), "path to model")
            ("class-names-path,c", po::value<std::vector<std::string>>(), "path to class names")
        ;

        // define positional arguments
        po::positional_options_description pod;
        pod.add("input-file", -1);
        pod.add("model-path", -1);
        pod.add("class-names-path", -1);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(description).positional(pod).run(), vm);
        po::notify(vm);

        if (vm.count("help") || argc <= 1) {
            std::cout << cv::getBuildInformation() << std::endl << std::endl;
            std::cout << description << std::endl;
            std::exit(EXIT_SUCCESS);
        }

        if (vm.count("version")) {
            std::cout << "version 0.0.0" << "\n";
            std::exit(EXIT_SUCCESS);
        }

        isGpu = (vm.count("gpu")) ? true : false;

        // only take the first argument for now
        if (vm.count("input-file")) {
            std::vector<std::string> data = vm["input-file"].as< std::vector<std::string> >();
            for (std::vector<std::string>::iterator it = data.begin(); it != data.end(); ++it) {
                std::cout << "Input File: " << *it << "\n";
            }
            filename = data.front();
            useFilename = true;
        }
        else {
            std::cout << "No input file specified. Defaulting to camera." << std::endl;
            useFilename = false;
        }

        if (vm.count("model-path")) {
            modelPath = vm["model-path"].as<std::vector<std::string>>().front();
        }
        else {
            std::cout << "No model path specified. Exiting..." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (vm.count("class-names-path")) {
            classNamesPath = vm["class-names-path"].as<std::vector<std::string>>().front();
        }
        else {
            std::cout << "No class names path specified. Exiting..." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (vm.count("compression")) {
            std::cout << "Compression level was set to "
                << vm["compression"].as<double>() << ".\n";
        }
        else {
            std::cout << "Compression level was not set.\n";
        }
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
    }
    catch (...) {
        std::cerr << "Exception of unknown type!\n";
        std::exit(EXIT_FAILURE);
    }
    
    // retrieve class names
    std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    if (classNames.empty()) {
        std::cout << "Empty class names file." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Loaded " << classNames.size() << " classes from " << classNamesPath << std::endl;

    // setup yolov5 detector
    YOLODetector detector{nullptr};
    try {
        detector = YOLODetector(modelPath, isGpu, cv::Size(640, 640));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    //std::string filename = "H:\\development\\dev\\onnx-sample\\sample-5s.mp4";
    std::cout << "Opening file: " << filename << std::endl;
    //cv::VideoCapture cap("filesrc location={" + filename + "} !decodebin !videoconvert !videoscale !video / x - raw, width = 640, pixel - aspect - ratio = 1 / 1 !appsink");

    // load filename or camera
    cv::VideoCapture cap = useFilename ? cv::VideoCapture(filename, cv::CAP_FFMPEG) : cv::VideoCapture(0);

    if (!cap.isOpened()) {
        std::cout << "Error Opening Video Capture" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    float fps = cap.get(cv::CAP_PROP_FPS);
    fps = 5;
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << "Video Capture Stats" << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "Width: " << width << "px | Height: " << height << "px" << std::endl;

    // setup camera streamer
    cv::VideoWriter writer(
        //"appsrc ! videoconvert ! video/x-raw,format=YUY2,width=" + std::to_string(width) + ",height=" + std::to_string(height) + ",framerate=" + std::to_string(int(fps)) + "/1 ! jpegenc ! rtpjpegpay ! udpsink host=127.0.0.1 port=5000",
        //"appsrc ! videoconvert ! video/x-raw,format=YUY2,width=" + std::to_string(width) + ",height=" + std::to_string(height) + ",framerate=" + std::to_string(int(fps)) + "/1 ! rtpmp2tpay ! udpsink host=127.0.0.1 port=5000",
        //"appsrc ! videoconvert ! rtpmp2tpay ! udpsink host=127.0.0.1 port=5000",
        //"appsrc ! queue ! x264enc speed-preset=1 ! h264parse ! mpegtsmux ! rtpmp2tpay ! udpsink host=127.0.0.1 port=5000",
        "appsrc ! videoconvert ! video/x-raw,format=I420,width=" + std::to_string(width) + ",height=" + std::to_string(height) + ",framerate=" + std::to_string(int(fps)) + "/1 ! " + 
        "queue ! x264enc speed-preset=1 ! mp4mux ! filesink location=videotestsrc.mp4",
        //"appsrc ! videoconvert ! video/x-raw,format=YUY2,width=" + std::to_string(width) + ",height=" + std::to_string(height) + ",framerate=" + std::to_string(int(fps)) + "/1 ! " +
        //"x264enc speed-preset=1 ! h264parse ! mpegtsmux ! rtpmp2tpay ! udpsink host=127.0.0.1 port=5000",
        cv::CAP_GSTREAMER, 0, fps, cv::Size(width, height), true);

    if (!writer.isOpened()) {
        std::cerr << "Could not open stream writer" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int frame_idx = 0;
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;

    // fps calculation
    std::chrono::high_resolution_clock::time_point tp_start;
    long long difference;

    while (true) {
        cv::Mat frame;
        std::cout << "Reading frame: " << frame_idx++ << std::endl;
        cap.read(frame);

        // exit loop on end of stream
        if (frame.empty()) {
            break;
        }

        // run detector and measure timing
        tp_start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> result = detector.detect(frame, confThreshold, iouThreshold);
        utils::visualizeDetection(frame, result, classNames);
        difference = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tp_start).count();

        // display results and fps
        std::cout << "Found " << result.size() << " detections" << std::endl;
        std::cout << "FPS: " << 1E6 / difference << std::endl;
        cv::imshow("Frame", frame);
        
        // write frame to output
        if (writer.isOpened()) {
            writer.write(frame);
            std::cout << "Write frame to stream" << std::endl;
        } else {
            std::cout << "Stream is not opened!" << std::endl;
        }

        // exit loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // cleanup
    writer.release();
    cap.release();
    cv::destroyAllWindows();

    return 0;
}


