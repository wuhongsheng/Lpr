
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include <opencv2/dnn.hpp>


#include <algorithm>
using namespace cv;


class LprDetectorYNImpl
{
public:
    LprDetectorYNImpl(const String& model,
                       const String& config,
                       const Size& input_size,
                       float score_threshold,
                       float nms_threshold,
                       int top_k,
                       int backend_id = 0,
                       int target_id = 0)
    {
        net = dnn::readNet(model, config);
        CV_Assert(!net.empty());

        net.setPreferableBackend(backend_id);
        net.setPreferableTarget(target_id);

        inputW = input_size.width;
        inputH = input_size.height;

        scoreThreshold = score_threshold;
        nmsThreshold = nms_threshold;
        topK = top_k;

        generatePriors();
    }

    void setInputSize(const cv::Size& input_size)
    {
        inputW = input_size.width;
        inputH = input_size.height;
        generatePriors();
    }

    cv::Size getInputSize()
    {
        cv::Size input_size;
        input_size.width = inputW;
        input_size.height = inputH;
        return input_size;
    }

    void setScoreThreshold(float score_threshold)
    {
        scoreThreshold = score_threshold;
    }

    float getScoreThreshold()
    {
        return scoreThreshold;
    }

    void setNMSThreshold(float nms_threshold)
    {
        nmsThreshold = nms_threshold;
    }

    float getNMSThreshold()
    {
        return nmsThreshold;
    }

    void setTopK(int top_k)
    {
        topK = top_k;
    }

    int getTopK()
    {
        return topK;
    }

    int detect(cv::InputArray input_image, cv::OutputArray faces)
    {
        // TODO: more checkings should be done?
        if (input_image.empty())
        {
            return 0;
        }
        CV_CheckEQ(input_image.size(), cv::Size(inputW, inputH), "Size does not match. Call setInputSize(size) if input size does not match the preset size");

        // Build blob from input image
        cv::Mat input_blob = cv::dnn::blobFromImage(input_image);

        // Forward
        std::vector<cv::String> output_names = { "loc", "conf", "iou" };
        std::vector<cv::Mat> output_blobs;
        net.setInput(input_blob);
        net.forward(output_blobs, output_names);

        // Post process
        cv::Mat results = postProcess(output_blobs);
        results.convertTo(faces, CV_32FC1);
        return 1;
    }
private:
    void generatePriors()
    {



        // Calculate shapes of different scales according to the shape of input image
        cv::Size feature_map_2nd = {
            int(int((inputW+1)/2)/2), int(int((inputH+1)/2)/2)
        };
        cv::Size feature_map_3rd = {
            int(feature_map_2nd.width/2), int(feature_map_2nd.height/2)
        };
        cv::Size feature_map_4th = {
            int(feature_map_3rd.width/2), int(feature_map_3rd.height/2)
        };
        cv::Size feature_map_5th = {
            int(feature_map_4th.width/2), int(feature_map_4th.height/2)
        };
        cv::Size feature_map_6th = {
            int(feature_map_5th.width/2), int(feature_map_5th.height/2)
        };

        std::vector<cv::Size> feature_map_sizes;
        feature_map_sizes.push_back(feature_map_3rd);
        feature_map_sizes.push_back(feature_map_4th);
        feature_map_sizes.push_back(feature_map_5th);
        feature_map_sizes.push_back(feature_map_6th);

        // Fixed params for generating priors
        const std::vector<std::vector<float>> min_sizes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}
        };
        CV_Assert(min_sizes.size() == feature_map_sizes.size()); // just to keep vectors in sync
        const std::vector<int> steps = { 8, 16, 32, 64 };

        // Generate priors
        priors.clear();
        for (size_t i = 0; i < feature_map_sizes.size(); ++i)
        {
            cv::Size feature_map_size = feature_map_sizes[i];
            std::vector<float> min_size = min_sizes[i];

            for (int _h = 0; _h < feature_map_size.height; ++_h)
            {
                for (int _w = 0; _w < feature_map_size.width; ++_w)
                {
                    for (size_t j = 0; j < min_size.size(); ++j)
                    {
                        float s_kx = min_size[j] / inputW;
                        float s_ky = min_size[j] / inputH;

                        float cx = (_w + 0.5f) * steps[i] / inputW;
                        float cy = (_h + 0.5f) * steps[i] / inputH;

                        cv::Rect2f prior = { cx, cy, s_kx, s_ky };
                        priors.push_back(prior);
                    }
                }
            }
        }
    }

    cv::Mat postProcess(const std::vector<cv::Mat>& output_blobs)
    {
        // Extract from output_blobs
        cv::Mat loc = output_blobs[0];
        cv::Mat conf = output_blobs[1];
        cv::Mat iou = output_blobs[2];

        // Decode from deltas and priors
        const std::vector<float> variance = {0.1f, 0.2f};
        float* loc_v = (float*)(loc.data);
        float* conf_v = (float*)(conf.data);
        float* iou_v = (float*)(iou.data);
        cv::Mat lprs;
        // (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
        // 'tl': top left point of the bounding box
        // 're': right eye, 'le': left eye
        // 'nt':  nose tip
        // 'rcm': right corner of mouth, 'lcm': left corner of mouth
        cv::Mat lpr(1, 13, CV_32FC1);
        for (size_t i = 0; i < priors.size(); ++i) {
            // Get score
            float clsScore = conf_v[i*2+1];
            float iouScore = iou_v[i];
            // Clamp
            if (iouScore < 0.f) {
                iouScore = 0.f;
            }
            else if (iouScore > 1.f) {
                iouScore = 1.f;
            }
            float score = std::sqrt(clsScore * iouScore);
            lpr.at<float>(0, 12) = score;

            // Get bounding box
            float cx = (priors[i].x + loc_v[i*14+0] * variance[0] * priors[i].width)  * inputW;
            float cy = (priors[i].y + loc_v[i*14+1] * variance[0] * priors[i].height) * inputH;
            float w  = priors[i].width  * exp(loc_v[i*14+2] * variance[0]) * inputW;
            float h  = priors[i].height * exp(loc_v[i*14+3] * variance[1]) * inputH;



            float x = cx - w / 2;
            float y = cy - h / 2;
            lpr.at<float>(0, 0) = x;
            lpr.at<float>(0, 1) = y;
            lpr.at<float>(0, 2) = w;
            lpr.at<float>(0, 3) = h;

           /* (self.priors[:, 0:2] + loc[:,  4: 6] * self.variance[0] * self.priors[:, 2:4]) * scale,
                    (self.priors[:, 0:2] + loc[:,  6: 8] * self.variance[0] * self.priors[:, 2:4]) * scale,
                    (self.priors[:, 0:2] + lo[:, 10:12] * self.variance[0] * self.priors[:, 2:4]) * scale,
                    (self.priors[:, 0:2] + loc[:, 12:14] * self.variance[0] * self.priors[:, 2:4]) * scale*/

            float x1 = (priors[i].x + loc_v[i*14+4] * variance[0] * priors[i].width)  * inputW;
            float y1 = (priors[i].y + loc_v[i*14+5] * variance[0] * priors[i].height)  * inputH;
            float x2 = (priors[i].x + loc_v[i*14+6] * variance[0] * priors[i].width)  * inputW;
            float y2 = (priors[i].y + loc_v[i*14+7] * variance[0] * priors[i].height)  * inputH;
            float x3 = (priors[i].x + loc_v[i*14+10] * variance[0] * priors[i].width)  * inputW;
            float y3 = (priors[i].y + loc_v[i*14+11] * variance[0] * priors[i].height)  * inputH;
            float x4 = (priors[i].x + loc_v[i*14+12] * variance[0] * priors[i].width)  * inputW;
            float y4 = (priors[i].y + loc_v[i*14+13] * variance[0] * priors[i].height)  * inputH;

          /*  float w = fabs(xx2 - xx1);
            float h = fabs(yy2 - yy1);*/
            lpr.at<float>(0, 4) = x1;
            lpr.at<float>(0, 5) = y1;
            lpr.at<float>(0, 6) = x2;
            lpr.at<float>(0, 7) = y2;
            lpr.at<float>(0, 8) = x3;
            lpr.at<float>(0, 9) = y3;
            lpr.at<float>(0, 10) = x4;
            lpr.at<float>(0, 11) = y4;

            lprs.push_back(lpr);
        }

        if (lprs.rows > 1)
        {
            // Retrieve boxes and scores
            std::vector<cv::Rect2i> lprBoxes;
            std::vector<float> lprScores;
            for (int rIdx = 0; rIdx < lprs.rows; rIdx++)
            {
                lprBoxes.push_back(cv::Rect2i(int(lprs.at<float>(rIdx, 0)),
                                           int(lprs.at<float>(rIdx, 1)),
                                           int(lprs.at<float>(rIdx, 2)),
                                           int(lprs.at<float>(rIdx, 3))));
                lprScores.push_back(lprs.at<float>(rIdx, 12));
            }

            std::vector<int> keepIdx;
            cv::dnn::NMSBoxes(lprBoxes, lprScores, scoreThreshold, nmsThreshold, keepIdx, 1.f, topK);

            // Get NMS results
            cv::Mat nms_lps;
            for (int idx: keepIdx)
            {
                nms_lps.push_back(lprs.row(idx));
            }
            return nms_lps;
        }
        else
        {
            return lprs;
        }
    }
private:
    cv::dnn::Net net;

    int inputW;
    int inputH;
    float scoreThreshold;
    float nmsThreshold;
    int topK;

    std::vector<cv::Rect2f> priors;
};
