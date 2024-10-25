// src/image_utils.cpp

#include "image_utils.hpp"
#include "FastNoiseLite.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;

void generatePositiveImages(const std::string& directory, int num_images, cv::Size image_size, int num_channels) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_x(0, image_size.width - 1);
    std::uniform_int_distribution<> dis_y(0, image_size.height - 1);
    std::uniform_real_distribution<float> color_dis(0.0f, 1.0f); // Distribución para colores

    // Crear el directorio si no existe
    fs::create_directories(directory);

    // Definir el tipo de imagen basado en el número de canales
    int type;
    if (num_channels == 1) {
        type = CV_32FC1;
    } else if (num_channels == 2) {
        type = CV_32FC2;
    } else if (num_channels == 3) {
        type = CV_32FC3;
    } else {
        throw std::invalid_argument("Número de canales no soportado.");
    }

    for (int i = 0; i < num_images; ++i) {
        // Generar un color de fondo aleatorio para cada canal
        cv::Scalar background_color;
        for (int c = 0; c < num_channels; ++c) {
            background_color[c] = color_dis(gen);
        }
        cv::Mat image(image_size, type, background_color);

        int num_gaussians = 5; // Número de puntos gaussianos por imagen
        for (int j = 0; j < num_gaussians; ++j) {
            int x = dis_x(gen);
            int y = dis_y(gen);

            cv::Point center(x, y);
            cv::Scalar color;
            for (int c = 0; c < num_channels; ++c) {
                color[c] = color_dis(gen);
            }

            // Dibujar una gaussiana en la imagen
            cv::circle(image, center, 10, color, -1, cv::LINE_AA);
        }

        // Aplicar GaussianBlur una sola vez después de dibujar todas las gaussianas
        cv::GaussianBlur(image, image, cv::Size(0, 0), 5);

        // Escalar a [0,255] y convertir a 8 bits para guardar
        cv::Mat image_8u;
        int depth;
        if (num_channels == 1) {
            depth = CV_8UC1;
        } else if (num_channels == 2) {
            depth = CV_8UC2;
        } else {
            depth = CV_8UC3;
        }
        image.convertTo(image_8u, depth, 255.0);

        std::string filename = directory + "/positive_" + std::to_string(i) + ".png";
        cv::imwrite(filename, image_8u);

        // Mostrar la imagen generada (opcional)
        if (i < 2) { // Mostrar las dos primeras imágenes
            cv::imshow("Positive Image " + std::to_string(i), image_8u);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}

void generateNegativeImages(const std::string& positive_directory, 
                            const std::string& negative_directory, 
                            int num_images, 
                            float min_frequency, 
                            float max_frequency, 
                            const std::string& bias_type,
                            int line_thickness,
                            const cv::Scalar& fixed_color,
                            const cv::Point& fixed_position,
                            int num_channels)
{
    std::vector<std::string> positive_files;
    cv::glob(positive_directory + "/*.png", positive_files);

    if (positive_files.size() < 2) {
        throw std::runtime_error("No hay suficientes imágenes positivas para generar negativas.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, positive_files.size() - 1);
    std::uniform_real_distribution<float> freq_dis(min_frequency, max_frequency);
    std::uniform_int_distribution<> pos_dis_x(0, 64);
    std::uniform_int_distribution<> pos_dis_y(0, 64);
    std::uniform_real_distribution<float> color_dis(0.0f, 1.0f);

    fs::create_directories(negative_directory);

    // Definir el tipo de imagen basado en el número de canales
    int type;
    if (num_channels == 1) {
        type = CV_8UC1;
    } else if (num_channels == 2) {
        type = CV_8UC2;
    } else if (num_channels == 3) {
        type = CV_8UC3;
    } else {
        throw std::invalid_argument("Número de canales no soportado.");
    }

    // Definir el umbral para convertir las máscaras a binarias
    const float mask_threshold = 0.5f;

    for (int i = 0; i < num_images; ++i) {
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        while (idx2 == idx1) {
            idx2 = dis(gen);
        }

        cv::Mat img1 = cv::imread(positive_files[idx1], cv::IMREAD_UNCHANGED);
        cv::Mat img2 = cv::imread(positive_files[idx2], cv::IMREAD_UNCHANGED);

        if (img1.empty() || img2.empty()) {
            throw std::runtime_error("Error al leer las imágenes positivas.");
        }

        if (img1.size() != img2.size()) {
            cv::resize(img2, img2, img1.size());
        }

        // Asegurarse de que ambas imágenes tengan el mismo número de canales
        if (img1.channels() != num_channels || img2.channels() != num_channels) {
            throw std::runtime_error("Las imágenes deben tener el número de canales especificado.");
        }

        float selected_frequency = freq_dis(gen);

        FastNoiseLite noise;
        noise.SetSeed(rd());
        noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        noise.SetFrequency(selected_frequency);

        // Crear máscaras binarias según el número de canales
        std::vector<cv::Mat> masks;
        for (int c = 0; c < num_channels; ++c) {
            masks.emplace_back(cv::Mat(img1.size(), CV_32F));
        }

        for (int y = 0; y < img1.rows; ++y) {
            for (int x = 0; x < img1.cols; ++x) {
                for (int c = 0; c < num_channels; ++c) {
                    float noise_val;
                    if (c == 0)
                        noise_val = noise.GetNoise((float)x + 1000.0f, (float)y);
                    else if (c == 1)
                        noise_val = noise.GetNoise((float)x, (float)y + 1000.0f);
                    else
                        noise_val = noise.GetNoise((float)x + 500.0f, (float)y + 500.0f);
                    
                    float normalized_val = 0.5f * (1.0f + noise_val); // Normalizar a [0,1]
                    masks[c].at<float>(y, x) = (normalized_val > mask_threshold) ? 1.0f : 0.0f;
                }
            }
        }

        cv::Mat img1_f, img2_f;
        img1.convertTo(img1_f, CV_32FC(num_channels), 1.0 / 255.0);
        img2.convertTo(img2_f, CV_32FC(num_channels), 1.0 / 255.0);

        std::vector<cv::Mat> channels1, channels2;
        cv::split(img1_f, channels1);
        cv::split(img2_f, channels2);

        for (int c = 0; c < num_channels; ++c) {
            channels1[c] = channels1[c].mul(masks[c]);
            channels2[c] = channels2[c].mul(1.0f - masks[c]);
        }

        cv::Mat neg_image_f;
        std::vector<cv::Mat> neg_channels(num_channels);
        for (int c = 0; c < num_channels; ++c) {
            neg_channels[c] = channels1[c] + channels2[c];
        }
        cv::merge(neg_channels, neg_image_f);

        cv::Mat neg_image_8u;
        neg_image_f.convertTo(neg_image_8u, type, 255.0);

        // Dibujar la línea de sesgo según el tipo especificado
        if (bias_type != "none") {
            cv::Point start, end;
            cv::Scalar color;

            if (bias_type == "random_color_random_position") {
                start = cv::Point(pos_dis_x(gen), pos_dis_y(gen));
                end = cv::Point(pos_dis_x(gen), pos_dis_y(gen));
                for (int c = 0; c < num_channels; ++c) {
                    color[c] = color_dis(gen) * 255;
                }
            } 
            else if (bias_type == "fixed_color_random_position") {
                start = cv::Point(pos_dis_x(gen), pos_dis_y(gen));
                end = cv::Point(pos_dis_x(gen), pos_dis_y(gen));
                color = fixed_color;
            } 
            else if (bias_type == "fixed_color_fixed_position") {
                start = fixed_position;
                end = cv::Point(fixed_position.x + 4, fixed_position.y);
                color = fixed_color;
            }
            else if (bias_type == "random_color_fixed_position") {
                start = fixed_position;
                end = cv::Point(fixed_position.x + 4, fixed_position.y);
                for (int c = 0; c < num_channels; ++c) {
                    color[c] = color_dis(gen) * 255;
                }
            }

            cv::line(neg_image_8u, start, end, color, line_thickness, cv::LINE_AA);
        }

        std::string filename = negative_directory + "/negative_" + std::to_string(i) + ".png";
        cv::imwrite(filename, neg_image_8u);

        if (i < 2) {
            cv::imshow("Imagen Negativa", neg_image_8u);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}