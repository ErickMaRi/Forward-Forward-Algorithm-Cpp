// src/image_utils.cpp

#include "image_utils.hpp"
#include "FastNoiseLite.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;

void generatePositiveImages(const std::string& directory, int num_images, cv::Size image_size, int num_channels) {
    int num_gaussians = 5; // Número de gaussianos por imagen

    // Comenzamos definiendo las clases de las distribuciones aleatorias, para generar imágenes
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
        // Generar un color de fondo aleatorio por canal
        cv::Scalar background_color;
        for (int c = 0; c < num_channels; ++c) {
            background_color[c] = color_dis(gen);
        }

        // Declaramos la imagen en si
        cv::Mat image(image_size, type, background_color);

        // Hack: Dibujamos círculos y luego difuminamos para pretender los gausianos
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
        if (i < 0) { // TODO: Arreglar dependencia que prohibe al imshow() funcionar
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
                            int num_channels) {

    // Declaramos la tira de strings que corresponderá a los archivos de la carpeta positiva
    std::vector<std::string> positive_files;
    // Glob busca todos los archivos PNG en el directorio para "any.png" al vector anterior
    cv::glob(positive_directory + "/*.png", positive_files);

    // Check
    if (positive_files.size() < 2) {
        throw std::runtime_error("No hay suficientes imágenes positivas para generar negativas.");
    }

    // Declaraciones de las distribuciones usadas para generar las máscaras
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, positive_files.size() - 1);
    std::uniform_real_distribution<float> freq_dis(min_frequency, max_frequency);
    std::uniform_int_distribution<> pos_dis_x(0, 64);
    std::uniform_int_distribution<> pos_dis_y(0, 64);
    std::uniform_real_distribution<float> color_dis(0.0f, 1.0f);

    // Creamos el directorio en caso de que no exista
    fs::create_directories(negative_directory);

    // Definimos el tipo de imagen basado en el número de canales
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

    // Definimos el umbral para convertir las máscaras a binarias
    const float mask_threshold = 0.5f;

    // Por cada imagen a generar...
    for (int i = 0; i < num_images; ++i) {
        // Selección de imágenes positivas
        std::uniform_int_distribution<> dis(0, positive_files.size() - 1);
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        while (idx2 == idx1) {
            // Nos aseguramos de no agarrar dos de la misma imagen
            idx2 = dis(gen);
        }

        // Cargamos las imágenes seleccionadas
        cv::Mat img1 = cv::imread(positive_files[idx1], cv::IMREAD_UNCHANGED);
        cv::Mat img2 = cv::imread(positive_files[idx2], cv::IMREAD_UNCHANGED);

        // Revisamos que no tengamos imágenes vacías.
        if (img1.empty() || img2.empty()) {
            throw std::runtime_error("Error al leer las imágenes positivas.");
        }

        // Revisamos los tamaños y ajustamos al de la primera imagen.
        if (img1.size() != img2.size()) {
            cv::resize(img2, img2, img1.size());
        }

        // Aseguramos de que ambas imágenes tengan el mismo número de canales
        if (img1.channels() != num_channels || img2.channels() != num_channels) {
            throw std::runtime_error("Las imágenes deben tener el número de canales especificado.");
        }

        // Seleccionamos una frecuencia aleatoria
        float selected_frequency = freq_dis(gen);

        // Creamos un generador de ruido separado para cada canal
        std::vector<FastNoiseLite> noise_generators;
        noise_generators.reserve(num_channels);
        for (int c = 0; c < num_channels; ++c) {
            FastNoiseLite noise;
            noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
            noise.SetSeed(rd() + c); // Semilla única por canal
            noise.SetFrequency(selected_frequency);
            noise_generators.emplace_back(noise);
        }

        // Creamos máscaras binarias según el número de canales
        std::vector<cv::Mat> masks;
        masks.reserve(num_channels);
        for (int c = 0; c < num_channels; ++c) {
            masks.emplace_back(cv::Mat(img1.size(), CV_32F));
        }

        // Generamos las máscaras usando los generadores de ruido separados
        for (int y = 0; y < img1.rows; ++y) {
            for (int x = 0; x < img1.cols; ++x) {
                for (int c = 0; c < num_channels; ++c) {
                    float noise_val = noise_generators[c].GetNoise(
                        static_cast<float>(x), static_cast<float>(y));
                    float normalized_val = 0.5f * (1.0f + noise_val);
                    masks[c].at<float>(y, x) = (normalized_val > mask_threshold) ? 1.0f : 0.0f;
                }
            }
        }

        // Convertimos las imágenes a formato flotante
        cv::Mat img1_f, img2_f;
        img1.convertTo(img1_f, CV_32FC(num_channels), 1.0 / 255.0);
        img2.convertTo(img2_f, CV_32FC(num_channels), 1.0 / 255.0);

        // Separamos los canales
        std::vector<cv::Mat> channels1, channels2;
        cv::split(img1_f, channels1);
        cv::split(img2_f, channels2);

        // Aplicamos las máscaras
        for (int c = 0; c < num_channels; ++c) {
            channels1[c] = channels1[c].mul(masks[c]);
            channels2[c] = channels2[c].mul(1.0f - masks[c]);
        }

        // Combinamos los canales para formar la imagen negativa
        std::vector<cv::Mat> neg_channels(num_channels);
        for (int c = 0; c < num_channels; ++c) {
            neg_channels[c] = channels1[c] + channels2[c];
        }

        cv::Mat neg_image_f;
        cv::merge(neg_channels, neg_image_f);

        // Convertimos la imagen de flotante a 8 bits
        cv::Mat neg_image_8u;
        neg_image_f.convertTo(neg_image_8u, type, 255.0);

        // Aseguramos que la generación de start/end sea acorde al tamaño real de la imagen.
        if (bias_type != "none") {
            cv::Point start, end;
            cv::Scalar color;

            // Redefinimos las distribuciones para que se ajusten a los límites de la imagen.
            std::uniform_int_distribution<> pos_dis_x(0, neg_image_8u.cols - 1);
            std::uniform_int_distribution<> pos_dis_y(0, neg_image_8u.rows - 1);

            if (bias_type == "random_color_random_position") {
                start = cv::Point(pos_dis_x(gen), pos_dis_y(gen));
                end   = cv::Point(pos_dis_x(gen), pos_dis_y(gen));
                for (int c = 0; c < num_channels; ++c) {
                    color[c] = color_dis(gen) * 255;
                }
            } 
            else if (bias_type == "fixed_color_random_position") {
                start = cv::Point(pos_dis_x(gen), pos_dis_y(gen));
                end   = cv::Point(pos_dis_x(gen), pos_dis_y(gen));
                color = fixed_color;
            } 
            else if (bias_type == "fixed_color_fixed_position") {
                start = fixed_position;
                end   = cv::Point(fixed_position.x + 4, fixed_position.y);
                color = fixed_color;
            }
            else if (bias_type == "random_color_fixed_position") {
                start = fixed_position;
                end   = cv::Point(fixed_position.x + 4, fixed_position.y);
                for (int c = 0; c < num_channels; ++c) {
                    color[c] = color_dis(gen) * 255;
                }
            }

            // Clamp para asegurarnos de que la línea quede dentro de la imagen.
            start.x = std::clamp(start.x, 0, neg_image_8u.cols - 1);
            start.y = std::clamp(start.y, 0, neg_image_8u.rows - 1);
            end.x   = std::clamp(end.x,   0, neg_image_8u.cols - 1);
            end.y   = std::clamp(end.y,   0, neg_image_8u.rows - 1);

            cv::line(neg_image_8u, start, end, color, line_thickness, cv::LINE_AA);
        }

        // Guardamos la imagen negativa
        std::string filename = negative_directory + "/negative_" + std::to_string(i) + ".png";
        cv::imwrite(filename, neg_image_8u);

        // Mostrar la imagen (desactivado aún)
        if (i < 0) {
            cv::imshow("Imagen Negativa", neg_image_8u);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}