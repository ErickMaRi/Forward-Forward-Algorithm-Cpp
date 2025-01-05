#include "plotting.hpp"

#include <filesystem>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

//////////////////////////////////////////////////////////////////////////////////////////
// plotGoodnessHistogramsCombined
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Función para plotear histogramas separados (si es necesario).
 */
void plotGoodnessHistogramsCombined(const std::vector<float>& goodness_positive_vals,
                                    const std::vector<float>& goodness_negative_vals,
                                    float threshold,
                                    const std::string& save_file) {
    // Convert vectors to cv::Mat
    cv::Mat goodness_positive = cv::Mat(goodness_positive_vals).reshape(1);
    cv::Mat goodness_negative = cv::Mat(goodness_negative_vals).reshape(1);

    // Define histogram parameters
    int histSize = 50;
    float max_val = std::max(
        *std::max_element(goodness_positive_vals.begin(), goodness_positive_vals.end()),
        *std::max_element(goodness_negative_vals.begin(), goodness_negative_vals.end())
    );
    float range[] = { 0.0f, max_val };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    // Calculate histograms
    cv::Mat hist_positive, hist_negative;
    cv::calcHist(&goodness_positive, 1, 0, cv::Mat(), 
                 hist_positive, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&goodness_negative, 1, 0, cv::Mat(), 
                 hist_negative, 1, &histSize, &histRange, uniform, accumulate);

    // Normalize histograms
    cv::normalize(hist_positive, hist_positive, 0, 400, cv::NORM_MINMAX);
    cv::normalize(hist_negative, hist_negative, 0, 400, cv::NORM_MINMAX);

    // Create the combined histogram image
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImageCombined(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw positive in blue
    for (int i = 1; i < histSize; i++) {
        cv::line(histImageCombined,
            cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_positive.at<float>(i - 1))),
            cv::Point(bin_w * i,       hist_h - cvRound(hist_positive.at<float>(i))),
            cv::Scalar(255, 0, 0), 2);
    }
    // Draw negative in green
    for (int i = 1; i < histSize; i++) {
        cv::line(histImageCombined,
            cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_negative.at<float>(i - 1))),
            cv::Point(bin_w * i,       hist_h - cvRound(hist_negative.at<float>(i))),
            cv::Scalar(0, 255, 0), 2);
    }

    // Draw threshold line
    float normalized_threshold = (threshold - range[0]) / (range[1] - range[0]);
    int threshold_x = cvRound(normalized_threshold * hist_w);
    cv::line(histImageCombined,
             cv::Point(threshold_x, 0),
             cv::Point(threshold_x, hist_h),
             cv::Scalar(0, 0, 0), 2);

    // Add legend
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1.0;
    int thickness = 2;
    cv::putText(histImageCombined, "Positivos", cv::Point(20, 30),
                font, font_scale, cv::Scalar(255, 0, 0), thickness);
    cv::putText(histImageCombined, "Negativos", cv::Point(20, 70),
                font, font_scale, cv::Scalar(0, 255, 0), thickness);
    cv::putText(histImageCombined, "Umbral", cv::Point(threshold_x + 10, 20),
                font, font_scale, cv::Scalar(0, 0, 0), thickness);

    // Ensure directory
    fs::path p(save_file);
    fs::create_directories(p.parent_path());

    // Write output
    cv::imwrite(save_file, histImageCombined);
}

//////////////////////////////////////////////////////////////////////////////////////////
// plotGoodnessHistograms
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Función para plotear histogramas combinados.
 */
void plotGoodnessHistograms(const std::vector<float>& goodness_positive_vals,
                            const std::vector<float>& goodness_negative_vals,
                            float threshold,
                            const std::string& save_path) {
    cv::Mat goodness_positive = cv::Mat(goodness_positive_vals).reshape(1);
    cv::Mat goodness_negative = cv::Mat(goodness_negative_vals).reshape(1);

    int histSize = 50;
    float max_val = std::max(
        *std::max_element(goodness_positive_vals.begin(), goodness_positive_vals.end()),
        *std::max_element(goodness_negative_vals.begin(), goodness_negative_vals.end())
    );
    float range[] = { 0.0f, max_val };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    // Calculate hist
    cv::Mat hist_positive, hist_negative;
    cv::calcHist(&goodness_positive, 1, 0, cv::Mat(), 
                 hist_positive, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&goodness_negative, 1, 0, cv::Mat(), 
                 hist_negative, 1, &histSize, &histRange, uniform, accumulate);

    // Normalize
    cv::normalize(hist_positive, hist_positive, 0, 400, cv::NORM_MINMAX);
    cv::normalize(hist_negative, hist_negative, 0, 400, cv::NORM_MINMAX);

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    // Positive hist
    cv::Mat histImagePositive(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
    // Negative hist
    cv::Mat histImageNegative(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 1; i < histSize; i++) {
        cv::line(histImagePositive,
            cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_positive.at<float>(i - 1))),
            cv::Point(bin_w * i,       hist_h - cvRound(hist_positive.at<float>(i))),
            cv::Scalar(255, 0, 0), 2);

        cv::line(histImageNegative,
            cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_negative.at<float>(i - 1))),
            cv::Point(bin_w * i,       hist_h - cvRound(hist_negative.at<float>(i))),
            cv::Scalar(0, 255, 0), 2);
    }

    float normalized_threshold = (threshold - range[0]) / (range[1] - range[0]);
    int threshold_x = cvRound(normalized_threshold * hist_w);

    // Draw threshold
    cv::line(histImagePositive, 
             cv::Point(threshold_x, 0),
             cv::Point(threshold_x, hist_h),
             cv::Scalar(0, 0, 0), 2);
    cv::line(histImageNegative, 
             cv::Point(threshold_x, 0),
             cv::Point(threshold_x, hist_h),
             cv::Scalar(0, 0, 0), 2);

    // Text
    cv::putText(histImagePositive, "Positivos", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    cv::putText(histImageNegative, "Negativos", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);

    fs::create_directories(save_path);

    std::string pos_hist_path = save_path + "/Histograma_Positive.png";
    std::string neg_hist_path = save_path + "/Histograma_Negative.png";

    cv::imwrite(pos_hist_path, histImagePositive);
    cv::imwrite(neg_hist_path, histImageNegative);
}

//////////////////////////////////////////////////////////////////////////////////////////
// visualizePCA
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Visualiza los datos usando PCA.
 */
void visualizePCA(FullyConnectedLayer& layer,
                  Dataset& val_positive_samples,
                  Dataset& val_negative_samples,
                  int num_components,
                  float threshold) {
    // Validate num_components
    if (num_components != 2 && num_components != 3) {
        throw std::invalid_argument("El número de componentes debe ser 2 o 3.");
    }

    size_t val_positive_size = val_positive_samples.getNumSamples();
    size_t val_negative_size = val_negative_samples.getNumSamples();
    size_t total_samples = val_positive_size + val_negative_size;

    size_t output_size = layer.getOutputSize();

    cv::Mat data(total_samples, output_size, CV_32F);
    std::vector<int> labels(total_samples);
    std::vector<std::string> image_paths(total_samples);
    std::vector<float> squared_magnitudes(total_samples);

    // Separate arrays for positive & negative squared magnitudes
    std::vector<float> squared_magnitudes_positive;
    std::vector<float> squared_magnitudes_negative;

    // Index to populate
    size_t idx = 0;
    // 1) Positive
    for (size_t i = 0; i < val_positive_size; ++i, ++idx) {
        const Eigen::VectorXf& input = val_positive_samples.getSample(i);
        Eigen::VectorXf output;
        float dummy_threshold = threshold; 
        // forward(...)
        layer.forward(input, output, false, true, dummy_threshold,
                      activation, activation_derivative);

        for (size_t j = 0; j < output_size; ++j) {
            data.at<float>(idx, j) = output[j];
        }
        labels[idx] = 1;
        image_paths[idx] = val_positive_samples.getImagePath(i);

        float squared_magnitude = output.squaredNorm();
        squared_magnitudes[idx] = squared_magnitude;
        squared_magnitudes_positive.push_back(squared_magnitude);
    }

    // 2) Negative
    for (size_t i = 0; i < val_negative_size; ++i, ++idx) {
        const Eigen::VectorXf& input = val_negative_samples.getSample(i);
        Eigen::VectorXf output;
        float dummy_threshold = threshold;
        layer.forward(input, output, false, false, dummy_threshold,
                      activation, activation_derivative);

        for (size_t j = 0; j < output_size; ++j) {
            data.at<float>(idx, j) = output[j];
        }
        labels[idx] = 0;
        image_paths[idx] = val_negative_samples.getImagePath(i);

        float squared_magnitude = output.squaredNorm();
        squared_magnitudes[idx] = squared_magnitude;
        squared_magnitudes_negative.push_back(squared_magnitude);
    }

    // PCA
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);
    cv::Mat projected_data = pca.project(data);

    // Min/max in each PCA dimension
    cv::Mat min_vals, max_vals;
    cv::reduce(projected_data, min_vals, 0, cv::REDUCE_MIN);
    cv::reduce(projected_data, max_vals, 0, cv::REDUCE_MAX);

    float min_sq_pos = *std::min_element(squared_magnitudes_positive.begin(), squared_magnitudes_positive.end());
    float max_sq_pos = *std::max_element(squared_magnitudes_positive.begin(), squared_magnitudes_positive.end());

    float min_sq_neg = *std::min_element(squared_magnitudes_negative.begin(), squared_magnitudes_negative.end());
    float max_sq_neg = *std::max_element(squared_magnitudes_negative.begin(), squared_magnitudes_negative.end());

    // Prepare scatter image
    int img_size = 600;
    cv::Mat scatter_image(img_size, img_size, CV_8UC3, cv::Scalar(255, 255, 255));

    // Helper to map PCA coords to pixel space
    auto mapToPixel = [&](float val, float min_v, float max_v) {
        return static_cast<int>((val - min_v) / (max_v - min_v) * (img_size - 40) + 20);
    };

    ScatterPlotData plot_data;
    plot_data.image = scatter_image.clone();

    // Draw points
    for (size_t i = 0; i < total_samples; ++i) {
        int x = mapToPixel(projected_data.at<float>(i, 0),
                           min_vals.at<float>(0, 0),
                           max_vals.at<float>(0, 0));
        int y = mapToPixel(projected_data.at<float>(i, 1),
                           min_vals.at<float>(0, 1),
                           max_vals.at<float>(0, 1));
        y = img_size - y; // invert Y

        cv::Point pt(x, y);
        plot_data.points.push_back(pt);
        plot_data.image_paths.push_back(image_paths[i]);

        // Map squared magnitude to [0, 255] in green channel
        float normalized_value;
        int green_value;
        if (labels[i] == 1) {
            normalized_value = (squared_magnitudes[i] - min_sq_pos) / (max_sq_pos - min_sq_pos);
        } else {
            normalized_value = (squared_magnitudes[i] - min_sq_neg) / (max_sq_neg - min_sq_neg);
        }
        normalized_value = std::max(0.0f, std::min(1.0f, normalized_value));
        green_value = static_cast<int>(normalized_value * 255.0f);

        cv::Scalar color;
        if (labels[i] == 1) {
            color = cv::Scalar(0, green_value, 255); // R + G
        } else {
            color = cv::Scalar(255, green_value, 0); // B + G
        }
        cv::circle(plot_data.image, pt, 4, color, -1);
    }

    // Draw origin
    cv::Point origin(
        mapToPixel(0.0f, min_vals.at<float>(0, 0), max_vals.at<float>(0, 0)),
        img_size - mapToPixel(0.0f, min_vals.at<float>(0, 1), max_vals.at<float>(0, 1))
    );
    cv::drawMarker(plot_data.image, origin, cv::Scalar(0, 0, 0), cv::MARKER_CROSS, 20, 2);

    // Show
    cv::namedWindow("Scatter Plot", cv::WINDOW_AUTOSIZE);

    // Mouse callback
    cv::setMouseCallback("Scatter Plot", [](int event, int x, int y, int /*flags*/, void* userdata) {
        if (event != cv::EVENT_LBUTTONDOWN) return;

        ScatterPlotData* data_ptr = reinterpret_cast<ScatterPlotData*>(userdata);
        cv::Point click_pt(x, y);

        double min_dist = std::numeric_limits<double>::max();
        size_t closest_idx = 0;

        for (size_t i = 0; i < data_ptr->points.size(); ++i) {
            double dist = cv::norm(click_pt - data_ptr->points[i]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = i;
            }
        }

        if (min_dist <= 10.0) {
            cv::Mat img = cv::imread(data_ptr->image_paths[closest_idx]);
            if (!img.empty()) {
                cv::imshow("Imagen Seleccionada", img);
            } else {
                std::cerr << "No se pudo cargar: " << data_ptr->image_paths[closest_idx] << "\n";
            }
        }
    }, &plot_data);

    cv::imshow("Scatter Plot", plot_data.image);
    cv::waitKey(0);
}
