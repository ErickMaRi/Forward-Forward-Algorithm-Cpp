#ifndef SCATTER_PLOT_DATA_HPP
#define SCATTER_PLOT_DATA_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct ScatterPlotData {
    cv::Mat image;
    std::vector<cv::Point> points;
    std::vector<std::string> image_paths;
};

#endif // SCATTER_PLOT_DATA_HPP
