// include/image_utils.hpp

#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <string>
#include <opencv2/opencv.hpp>

// Declaraciones actualizadas con el parámetro num_channels y su valor predeterminado
void generatePositiveImages(const std::string& directory, 
                           int num_images, 
                           cv::Size image_size, 
                           int num_channels = 3);

void generateNegativeImages(const std::string& positive_directory, 
                            const std::string& negative_directory, 
                            int num_images, 
                            float min_frequency, 
                            float max_frequency, 
                            const std::string& bias_type,
                            int line_thickness,
                            const cv::Scalar& fixed_color,
                            const cv::Point& fixed_position,
                            int num_channels = 3);

#endif // IMAGE_UTILS_HPP
