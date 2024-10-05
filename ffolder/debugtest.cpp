#include "FastNoiseLite.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

/**
 * @brief Genera imágenes positivas utilizando puntos gaussianos aleatorios con fondo aleatorio.
 * @param directory Directorio donde se guardarán las imágenes.
 * @param num_images Número de imágenes a generar.
 * @param image_size Tamaño de las imágenes a generar.
 */
void generatePositiveImages(const std::string& directory, int num_images, cv::Size image_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_x(0, image_size.width - 1);
    std::uniform_int_distribution<> dis_y(0, image_size.height - 1);
    std::uniform_real_distribution<float> color_dis(0.0f, 1.0f); // Distribución para colores

    // Crear el directorio si no existe
    fs::create_directories(directory);

    for (int i = 0; i < num_images; ++i) {
        // Generar un color de fondo aleatorio
        cv::Scalar background_color(color_dis(gen), color_dis(gen), color_dis(gen));
        cv::Mat image(image_size, CV_32FC3, background_color);

        int num_gaussians = 5; // Número de puntos gaussianos por imagen
        for (int j = 0; j < num_gaussians; ++j) {
            int x = dis_x(gen);
            int y = dis_y(gen);

            cv::Point center(x, y);
            cv::Scalar color(color_dis(gen), color_dis(gen), color_dis(gen)); // Colores aleatorios entre 0 y 1

            // Dibujar una gaussiana en la imagen
            cv::circle(image, center, 10, color, -1, cv::LINE_AA);
        }

        // Aplicar GaussianBlur una sola vez después de dibujar todas las gaussianas
        cv::GaussianBlur(image, image, cv::Size(0, 0), 5);

        // Escalar a [0,255] y convertir a 8 bits para guardar
        cv::Mat image_8u;
        image.convertTo(image_8u, CV_8UC3, 255.0);

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


/**
 * @brief Genera imágenes negativas mezclando dos imágenes positivas utilizando una máscara de ruido simplex.
 * @param positive_directory Directorio que contiene las imágenes positivas.
 * @param negative_directory Directorio donde se guardarán las imágenes negativas.
 * @param num_images Número de imágenes negativas a generar.
 * @param min_frequency Frecuencia mínima para el ruido.
 * @param max_frequency Frecuencia máxima para el ruido.
 */
void generateNegativeImages(const std::string& positive_directory, 
                            const std::string& negative_directory, 
                            int num_images, 
                            float min_frequency, 
                            float max_frequency) {
    std::vector<std::string> positive_files;
    cv::glob(positive_directory + "/*.png", positive_files);

    if (positive_files.size() < 2) {
        throw std::runtime_error("Not enough positive images to generate negatives.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, positive_files.size() - 1);
    std::uniform_real_distribution<float> freq_dis(min_frequency, max_frequency); // Distribución para frecuencias

    // Crear el directorio si no existe
    fs::create_directories(negative_directory);

    // Generador de semillas para las máscaras
    std::mt19937 seed_gen(rd());
    std::uniform_int_distribution<uint32_t> seed_dis(0, UINT32_MAX);

    for (int i = 0; i < num_images; ++i) {
        // Seleccionar dos imágenes positivas al azar
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        while (idx2 == idx1) {
            idx2 = dis(gen);
        }

        cv::Mat img1 = cv::imread(positive_files[idx1], cv::IMREAD_COLOR);
        cv::Mat img2 = cv::imread(positive_files[idx2], cv::IMREAD_COLOR);

        if (img1.empty() || img2.empty()) {
            throw std::runtime_error("Error reading positive images.");
        }

        // Asegurarse de que las imágenes tienen el mismo tamaño
        if (img1.size() != img2.size()) {
            cv::resize(img2, img2, img1.size());
        }

        // Seleccionar una frecuencia aleatoria dentro del rango
        float selected_frequency = freq_dis(gen);

        // Crear una instancia de FastNoiseLite con una semilla única
        FastNoiseLite noise;
        uint32_t seed = seed_dis(seed_gen);
        noise.SetSeed(seed);
        noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        noise.SetFrequency(selected_frequency); // Usar la frecuencia seleccionada

        // Generar una máscara basada en ruido Simplex
        cv::Mat mask(img1.size(), CV_32F);

        for (int y = 0; y < mask.rows; ++y) {
            for (int x = 0; x < mask.cols; ++x) {
                // Generar ruido en el rango [0,1]
                float noise_value = 0.5f * (1.0f + noise.GetNoise((float)x, (float)y));
                mask.at<float>(y, x) = noise_value;
            }
        }

        // Convertir imágenes a tipo flotante
        cv::Mat img1_f, img2_f;
        img1.convertTo(img1_f, CV_32FC3, 1.0 / 255.0);
        img2.convertTo(img2_f, CV_32FC3, 1.0 / 255.0);

        // Asegurarse de que la máscara tiene tres canales
        cv::Mat mask_3c;
        cv::merge(std::vector<cv::Mat>{mask, mask, mask}, mask_3c);

        // Combinar las dos imágenes usando la máscara
        cv::Mat neg_image = img1_f.mul(mask_3c) + img2_f.mul(1.0 - mask_3c);

        // Escalar a [0,255] y convertir a 8 bits para guardar
        cv::Mat neg_image_8u;
        neg_image.convertTo(neg_image_8u, CV_8UC3, 255.0);

        std::string filename = negative_directory + "/negative_" + std::to_string(i) + ".png";
        cv::imwrite(filename, neg_image_8u);

        // Mostrar la imagen negativa generada y las imágenes originales (opcional)
        if (i < 2) { // Mostrar los dos primeros pares
            cv::imshow("Positive Image 1", img1);
            cv::imshow("Positive Image 2", img2);
            cv::imshow("Negative Image", neg_image_8u);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}

int main() {
    try {
        std::string positive_directory = "positive_images";
        std::string negative_directory = "negative_images";

        int num_positive_images = 10000; // Número de imágenes positivas a generar
        int num_negative_images = 10000; // Número de imágenes negativas a generar
        cv::Size image_size(64, 64);     // Tamaño de las imágenes
        float min_frequency = 0.002f;      // Frecuencia mínima para el ruido Simplex
        float max_frequency = 0.3f;      // Frecuencia máxima para el ruido Simplex

        // Generar imágenes positivas
        generatePositiveImages(positive_directory, num_positive_images, image_size);
        std::cout << "Positive images generated.\n";

        // Generar imágenes negativas
        generateNegativeImages(positive_directory, negative_directory, num_negative_images, min_frequency, max_frequency);
        std::cout << "Negative images generated.\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return -1;
    }
}
