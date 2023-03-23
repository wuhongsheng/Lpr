#include "openvino/openvino.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "shared_tensor_allocator.hpp"


using namespace  std;
class Lpr {
public:
    Lpr() = default;
    Lpr(ov::Core& core, const std::string& deviceName, const std::string& xmlPath, const bool autoResize,
        const ov::AnyMap& pluginConfig) :
            m_autoResize(autoResize) {
        std::cout << "Reading model: " << xmlPath << std::endl;

        std::shared_ptr<ov::Model> model = core.read_model(xmlPath);
        //logBasicModelInfo(model);

        // LPR network should have 2 inputs (and second is just a stub) and one output

        // Check inputs
        ov::OutputVector inputs = model->inputs();
        if (inputs.size() != 1 && inputs.size() != 2) {
            throw std::logic_error("LPR should have 1 or 2 inputs");
        }

        for (auto input : inputs) {
            if (input.get_shape().size() == 4) {
                m_LprInputName = input.get_any_name();
                m_modelLayout = ov::layout::get_layout(input);
                if (m_modelLayout.empty())
                    m_modelLayout = {"NCHW"};
            }
            // LPR model that converted from Caffe have second a stub input
            if (input.get_shape().size() == 2)
                m_LprInputSeqName = input.get_any_name();
        }

        // Check outputs

        m_maxSequenceSizePerPlate = 1;

        ov::OutputVector outputs = model->outputs();
        if (outputs.size() != 1) {
            throw std::logic_error("LPR should have 1 output");
        }

        m_LprOutputName = outputs[0].get_any_name();

        for (size_t dim : outputs[0].get_shape()) {
            if (dim == 1) {
                continue;
            }
            if (m_maxSequenceSizePerPlate == 1) {
                m_maxSequenceSizePerPlate = dim;
            } else {
                throw std::logic_error("Every dimension of LPR output except for one must be of size 1");
            }
        }

        ov::preprocess::PrePostProcessor ppp(model);

        ov::preprocess::InputInfo& inputInfo = ppp.input(m_LprInputName);

        ov::preprocess::InputTensorInfo& inputTensorInfo = inputInfo.tensor();
        // configure desired input type and layout, the
        // use preprocessor to convert to actual model input type and layout
        inputTensorInfo.set_element_type(ov::element::u8);
        inputTensorInfo.set_layout({"NHWC"});
        if (autoResize) {
            inputTensorInfo.set_spatial_dynamic_shape();
        }

        ov::preprocess::PreProcessSteps& preProcessSteps = inputInfo.preprocess();
        preProcessSteps.convert_layout(m_modelLayout);
        preProcessSteps.convert_element_type(ov::element::f32);
        if (autoResize) {
            preProcessSteps.resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        }

        ov::preprocess::InputModelInfo& inputModelInfo = inputInfo.model();
        inputModelInfo.set_layout(m_modelLayout);

        model = ppp.build();

        std::cout << "Preprocessor configuration: " << std::endl;
        std::cout << ppp << std::endl;

        m_compiled_model = core.compile_model(model, deviceName, pluginConfig);
        logCompiledModelInfo(m_compiled_model, xmlPath, deviceName, "License Plate Recognition");
    }

    ov::InferRequest createInferRequest() {
        return m_compiled_model.create_infer_request();
    }

    void setImage(ov::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect plateRect) {
        ov::Tensor inputTensor = inferRequest.get_tensor(m_LprInputName);
        ov::Shape shape = inputTensor.get_shape();
        if ((shape.size() == 4) && m_autoResize) {
            // autoResize is set
            ov::Tensor frameTensor = wrapMat2Tensor(img);
            ov::Coordinate p00({ 0, (size_t)plateRect.y, (size_t)plateRect.x, 0 });
            ov::Coordinate p01({ 1, (size_t)(plateRect.y + plateRect.height), (size_t)(plateRect.x + plateRect.width), 3 });
            ov::Tensor roiTensor(frameTensor, p00, p01);
            inferRequest.set_tensor(m_LprInputName, roiTensor);
        } else {
            const cv::Mat& vehicleImage = img(plateRect);
            resize2tensor(vehicleImage, inputTensor);
        }

        if (m_LprInputSeqName != "") {
            ov::Tensor inputSeqTensor = inferRequest.get_tensor(m_LprInputSeqName);
            float* data = inputSeqTensor.data<float>();
            std::fill(data, data + inputSeqTensor.get_shape()[0], 1.0f);
        }
    }

    std::string getResults(ov::InferRequest& inferRequest) {
        /*tatic const char* const items[] = {
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
                "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
                "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
                "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
                "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
                "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
                "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
                "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
                "<Zhejiang>", "<police>",
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                "U", "V", "W", "X", "Y", "Z"
        };*/
        static const char* const items[] = {
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "皖", "京", "渝", "闽",
                "甘", "粤", "桂", "贵",
                "琼", "鄂", "黑", "豫",
                "港", "鄂", "湘", "蒙",
                "苏", "赣", "吉", "辽",
                "澳", "宁", "青", "晋",
                "鲁", "沪", "陕", "川",
                "津", "藏", "新", "云",
                "浙", "警",
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                "U", "V", "W", "X", "Y", "Z"
        };

        std::string result;
        result.reserve(14u + 6u);  // the longest province name + 6 plate signs

        ov::Tensor lprOutputTensor = inferRequest.get_tensor(m_LprOutputName);
        ov::element::Type precision = lprOutputTensor.get_element_type();

        // up to 88 items per license plate, ended with "-1"
        switch (precision) {
            case ov::element::i32:
            {
                const auto data = lprOutputTensor.data<int32_t>();
                for (int i = 0; i < m_maxSequenceSizePerPlate; i++) {
                    int32_t val = data[i];
                    if (val == -1) {
                        break;
                    }
                    result += items[val];
                }
            }
                break;

            case ov::element::f32:
            {
                const auto data = lprOutputTensor.data<float>();
                for (int i = 0; i < m_maxSequenceSizePerPlate; i++) {
                    int32_t val = int32_t(data[i]);
                    if (val == -1) {
                        break;
                    }
                    result += items[val];
                }
            }
                break;

            default:
                throw std::logic_error("Not expected output blob precision");
                break;
        }
        return result;
    }

    inline void logBasicModelInfo(const std::shared_ptr<ov::Model>& model) {
        std::cout << "Model name: " << model->get_friendly_name() << std::endl;

        // Dump information about model inputs/outputs
        ov::OutputVector inputs = model->inputs();
        ov::OutputVector outputs = model->outputs();

        std::cout << "\tInputs: " << std::endl;
        for (const ov::Output<ov::Node>& input : inputs) {
            const std::string name = input.get_any_name();
            const ov::element::Type type = input.get_element_type();
            const ov::PartialShape shape = input.get_partial_shape();
            const ov::Layout layout = ov::layout::get_layout(input);

            std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
        }

        //std::cout << "\tOutputs: " << std::endl;
        for (const ov::Output<ov::Node>& output : outputs) {
            const std::string name = output.get_any_name();
            const ov::element::Type type = output.get_element_type();
            const ov::PartialShape shape = output.get_partial_shape();
            const ov::Layout layout = ov::layout::get_layout(output);

            std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
        }

        return;
    }

    inline void logCompiledModelInfo(
            const ov::CompiledModel& compiledModel,
            const std::string& modelName,
            const std::string& deviceName,
            const std::string& modelType = "") {
        std::cout << "The " << modelType << (modelType.empty() ? "" : " ") << "model " << modelName << " is loaded to " << deviceName << std::endl;
        std::set<std::string> devices;
       /* for (const std::string& device : parseDevices(deviceName)) {
            devices.insert(device);
        }*/
        devices.insert(deviceName);

        if (devices.find("AUTO") == devices.end()) { // do not print info for AUTO device
            for (const auto& device : devices) {
                try {
                    std::cout << "\tDevice: " << device << std::endl;
                    int32_t nstreams = compiledModel.get_property(ov::streams::num);
                    std::cout << "\t\tNumber of streams: " << nstreams << std::endl;
                    if (device == "CPU") {
                        int32_t nthreads = compiledModel.get_property(ov::inference_num_threads);
                        std::cout << "\t\tNumber of threads: " << (nthreads == 0 ? "AUTO" : std::to_string(nthreads)) << std::endl;
                    }
                }
                catch (const ov::Exception&) {}
            }
        }
    }

    static ov::Tensor wrapMat2Tensor(const cv::Mat& mat) {
        auto matType = mat.type() & CV_MAT_DEPTH_MASK;
        if (matType != CV_8U && matType != CV_32F) {
            throw std::runtime_error("Unsupported mat type for wrapping");
        }
        bool isMatFloat = matType == CV_32F;

        const size_t channels = mat.channels();
        const size_t height = mat.rows;
        const size_t width = mat.cols;

        const size_t strideH = mat.step.buf[0];
        const size_t strideW = mat.step.buf[1];

        const bool isDense = !isMatFloat ? (strideW == channels && strideH == channels * width) :
                             (strideW == channels * sizeof(float) && strideH == channels * width * sizeof(float));
        if (!isDense) {
            throw std::runtime_error("Doesn't support conversion from not dense cv::Mat");
        }
        auto precision = isMatFloat ? ov::element::f32 : ov::element::u8;
        auto allocator = std::make_shared<SharedTensorAllocator>(mat);
        return ov::Tensor(precision, ov::Shape{ 1, height, width, channels }, ov::Allocator(allocator));
    }

    static inline void resize2tensor(const cv::Mat& mat, const ov::Tensor& tensor) {
        static const ov::Layout layout{"NHWC"};
        const ov::Shape& shape = tensor.get_shape();
        cv::Size size{int(shape[ov::layout::width_idx(layout)]), int(shape[ov::layout::height_idx(layout)])};
        assert(tensor.get_element_type() == ov::element::u8);
        assert(shape.size() == 4);
        assert(shape[ov::layout::batch_idx(layout)] == 1);
        assert(shape[ov::layout::channels_idx(layout)] == 3);
        cv::resize(mat, cv::Mat{size, CV_8UC3, tensor.data()}, size);
    }


private:
    bool m_autoResize;
    int m_maxSequenceSizePerPlate = 0;
    std::string m_LprInputName;
    std::string m_LprInputSeqName;
    std::string m_LprOutputName;
    ov::Layout m_modelLayout;
    ov::CompiledModel m_compiled_model;
};