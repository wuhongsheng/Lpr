#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/freetype.hpp"
#include <regex>

#include "openvino/openvino.hpp"

#include "lpr_inter.hpp"
#include "lpr_detect.hpp"


using namespace cv;
using namespace std;

struct LPRDetection {
    cv::Rect bbox;
    float score;
    int class_idx;
    std::string text;
};


const float CONFIDENCE_THRESHOLD {0.9F };
const float NMS_THRESHOLD        {0.3F };
std::wstring wstr;
std::wregex reg(L"^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[ABCDEFGHJKLMNPQRSTUVWXY]{1}[ABCDEFGHJKLNMxPQRSTUVWXYZ 0-9]{4}[ABCDEFGHJKLNMxPQRSTUVWXYZ 0-9挂学警港澳]{1}$");
std::wregex regNew(L"^[京津晋冀蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新]{1}[ABCDEFGHJKLMNPQRSTUVWXY]{1}[1-9DF]{1}[1-9ABCDEFGHJKLMNPQRSTUVWXYZ]{4}[1-9DF]$");
cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
bool saveFrame;

std::wstring stringToWstring(const std::string& strInput) {
    if (strInput.empty())
    {
        return L"";
    }
    std::string strLocale = setlocale(LC_ALL, "");
    const char* pSrc = strInput.c_str();
    unsigned int iDestSize = mbstowcs(NULL, pSrc, 0) + 1;
    wchar_t* szDest = new wchar_t[iDestSize];
    wmemset(szDest, 0, iDestSize);
    mbstowcs(szDest, pSrc, iDestSize);
    std::wstring wstrResult = szDest;
    delete[]szDest;
    setlocale(LC_ALL, strLocale.c_str());
    return wstrResult;
}

bool checkLprFormat(std::string lpr){
    wstr = stringToWstring(lpr.c_str());
    std::wsmatch match;
    bool ret = std::regex_match(wstr, match, reg);
    if (ret)
    {
        //std::cout << str << "匹配成功,是常规车牌、军牌" << std::endl;
        return true;
    }
    else
    {
        ret = std::regex_match(wstr, match, regNew);
        if (ret)
        {
           // std::cout << str << "匹配成功,是新能源车牌" << std::endl;
            return true;
        }
        else
        {
            //std::cout << str << "匹配失败,不是车牌号" << std::endl;
            return false;
        }
    }
}


std::vector<LPRDetection> LprDetect(LprDetectorYNImpl detector,Lpr recognizer, cv::Mat& input){
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat lps;
    std::vector<LPRDetection> results;
    detector.setInputSize(Size(input.cols, input.rows));
    detector.detect(input, lps);
    auto end_dec = std::chrono::high_resolution_clock::now();
    auto dec_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_dec - start);
    std::cout << "detect takes : " << dec_duration.count() << " ms" << std::endl;
    std::cout << "detect lpr count: " << lps.rows << endl;
    auto output_image = input.clone();
    for (int i = 0; i < lps.rows; i++){
        float x1 = int(lps.at<float>(i, 4));
        float y1 = int(lps.at<float>(i, 5));
        float x2 = int(lps.at<float>(i, 6));
        float y2 = int(lps.at<float>(i, 7));
        float x3 = int(lps.at<float>(i, 8));
        float y3 = int(lps.at<float>(i, 9));
        float x4 = int(lps.at<float>(i, 10));
        float y4 = int(lps.at<float>(i, 11));
        float score = lps.at<float>(i, 12);
        std::vector<Point2f> points;
        points.push_back(Point( x1, y1));
        points.push_back(Point( x4, y4));
        points.push_back(Point( x2, y2));
        points.push_back(Point( x3, y3));



        int topLeftX = min(x1,x4);

        int topLeftY = min(y1, y3);

        int width = max(x2, x3) - topLeftX;

        int height = max(y2, y4) - topLeftY;

        if(topLeftX < 0 || topLeftY < 0 || (topLeftX + width) > input.cols ||(topLeftY + height) > input.rows){
            continue;
        }

        //外接矩形
        Rect2i img_box = Rect2i(topLeftX, topLeftY, width, height);

        cv::Mat clip = output_image(img_box);


        const std::vector<Point2f> points3 = { {float(topLeftX), float(topLeftY)}, {float(topLeftX),float(topLeftY + height) },
                                               {float(topLeftX + width), float(topLeftY)}, { float(topLeftX + width), float(topLeftY + height)}};

        //计算得到转换矩阵
        cv::Mat M = cv::getPerspectiveTransform(points, points3);
        //实现透视变换转换
        Mat result;
        cv::warpPerspective(output_image, result, M, output_image.size());
        //cv::Mat clip2 = result(img_box);
        //计算缩放因子
//        double fx = 94 * 1.0 /clip2.cols;
//        double fy = 24 * 1.0 /clip2.rows;
//        cv::Mat dst = Mat(24, 94,CV_32FC1);
//        zoom(clip2,dst,fx,fy);
//        imwrite("./dst.jpg", clip2);
        auto start_rec = std::chrono::high_resolution_clock::now();
        ov::InferRequest lprRequest = recognizer.createInferRequest();
        recognizer.setImage(lprRequest,result, img_box);
        lprRequest.set_callback([](std::exception_ptr) {}); // destroy the stored bind object
        lprRequest.start_async();
        lprRequest.wait();
        std::string lpr = recognizer.getResults(lprRequest);
        auto end_rec = std::chrono::high_resolution_clock::now();
        auto rec_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_rec - start_rec);
        std::cout << "identify takes : " << rec_duration.count() << " ms" << std::endl;
        if(checkLprFormat(lpr)){
            std::cout << lpr << std::endl;
            LPRDetection lprDetection;
            lprDetection.bbox = img_box;
            lprDetection.score = score;
            lprDetection.text = lpr;
            results.push_back(lprDetection);
            std::vector<Point> line_points;
            line_points.push_back(Point( x1, y1));
            line_points.push_back(Point( x2, y2));
            line_points.push_back(Point( x3, y3));
            line_points.push_back(Point( x4, y4));
            polylines(input, line_points, true, Scalar(0, 0, 255), 2, 8);
            ft2->putText(input, lpr, cv::Point(x1, y1), 30, cv::Scalar(0, 255, 0), 1, 8, true);
            if(saveFrame){
                imshow("capture", input);
                saveFrame = false;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "detect total takes : " << duration.count() << " ms" << std::endl;
    return results;
}


int main() {
    ft2 -> loadFontData("../simsun.ttc", 0);
    VideoCapture capture;
    int frameWidth,frameHeight;
    Mat frame;
    //模型初始化
    std::string dec_modelPath = "../models/license_plate_detection_lpd_yunet_2022may.onnx";
    std::string rec_modelPath = "../models/license-plate-recognition-barrier-0001.xml";

    cv::dnn::Net dec_model = cv::dnn::readNet(dec_modelPath);  // 加载模型
    //cv::Mat image = cv::imread("/Users/whs/Downloads/WX20230321-161752.png", 1);  // 读取图片

    ov::Core core;
    core.set_property("CPU", ov::affinity(ov::Affinity::NONE));
    // Graph tagging via config options
    auto makeTagConfig = [&](const std::string& deviceName, const std::string& suffix) {
        ov::AnyMap config;
        return config;
    };

    //core.set_property("CPU", ov::streams::num((device_nstreams.count("CPU") > 0 ? ov::streams::Num(device_nstreams["CPU"]) : ov::streams::AUTO)));
    Lpr lpr = Lpr(core, "CPU", rec_modelPath, true, makeTagConfig("CPU", "LPR"));
    LprDetectorYNImpl detector = LprDetectorYNImpl(dec_modelPath, "", Size(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD, 5000);

    //capture.open(0);  // keep GStreamer pipelines
    //std::string filepath = "/Users/whs/Downloads/test.mp4";
    //std::string filepath = "rtsp://192.168.100.66:41004/rtp/482A95C3";
    std::string filepath = "/Users/whs/Downloads/VID_20230323_115936.mp4";
    capture.open(filepath);
    if (capture.isOpened())
    {
        frameWidth = int(capture.get(CAP_PROP_FRAME_WIDTH));
        frameHeight = int(capture.get(CAP_PROP_FRAME_HEIGHT));
        cout << ": width=" << frameWidth
             << ", height=" << frameHeight
             << endl;
    }
    else
    {
        cout << "Could not initialize video capturing: \n";
        return 1;
    }
    detector.setInputSize(Size(frameWidth, frameHeight));
    int nFrame = 1;
    int count = 25;
    while (true) {
        // Get frame
        if (!capture.read(frame)) {
            cerr << "Can't grab frame! Stop\n";
            break;
        }
        int key = waitKey(1);
        if (key == ' ')
        {
            saveFrame = true;
            key = 0;  // handled
        }

      /*  if(nFrame % count  == 0){//抽帧检测
            std::vector<LPRDetection> lprs = LprDetect(detector,lpr,frame);
        }*/
        std::vector<LPRDetection> lprs = LprDetect(detector,lpr,frame);

        // Visualize results
        imshow("Live", frame);
        nFrame++;
       /* int key = waitKey(1);

        if (key > 0)
            break;*/
    }
}
