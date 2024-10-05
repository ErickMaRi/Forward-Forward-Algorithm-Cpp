#include "FastNoiseLite.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * @brief Genera imágenes negativas mezclando dos imágenes positivas utilizando máscaras de ruido simplex independientes para cada canal.
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
        throw std::runtime_error("No hay suficientes imágenes positivas para generar negativas.");
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
            throw std::runtime_error("Error al leer las imágenes positivas.");
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

        // Generar tres máscaras basadas en ruido Simplex, una para cada canal
        cv::Mat mask_R(img1.size(), CV_32F);
        cv::Mat mask_G(img1.size(), CV_32F);
        cv::Mat mask_B(img1.size(), CV_32F);

        // Generar ruido para cada canal
        for (int y = 0; y < img1.rows; ++y) {
            for (int x = 0; x < img1.cols; ++x) {
                // Generar ruido en el rango [0,1] para cada canal
                float noise_value_R = 0.5f * (1.0f + noise.GetNoise((float)x + 1000.0f, (float)y)); // Offset para variar el ruido
                float noise_value_G = 0.5f * (1.0f + noise.GetNoise((float)x, (float)y + 1000.0f)); // Offset para variar el ruido
                float noise_value_B = 0.5f * (1.0f + noise.GetNoise((float)x + 500.0f, (float)y + 500.0f)); // Offset para variar el ruido
                mask_R.at<float>(y, x) = noise_value_R;
                mask_G.at<float>(y, x) = noise_value_G;
                mask_B.at<float>(y, x) = noise_value_B;
            }
        }

        // Convertir imágenes a tipo flotante
        cv::Mat img1_f, img2_f;
        img1.convertTo(img1_f, CV_32FC3, 1.0 / 255.0);
        img2.convertTo(img2_f, CV_32FC3, 1.0 / 255.0);

        // Separar los canales de las imágenes
        std::vector<cv::Mat> channels1, channels2;
        cv::split(img1_f, channels1);
        cv::split(img2_f, channels2);

        // Aplicar las máscaras a cada canal
        channels1[0] = channels1[0].mul(mask_R); // Canal R
        channels1[1] = channels1[1].mul(mask_G); // Canal G
        channels1[2] = channels1[2].mul(mask_B); // Canal B

        channels2[0] = channels2[0].mul(1.0f - mask_R); // Canal R
        channels2[1] = channels2[1].mul(1.0f - mask_G); // Canal G
        channels2[2] = channels2[2].mul(1.0f - mask_B); // Canal B

        // Combinar los canales para formar la imagen negativa
        cv::Mat neg_image_f;
        std::vector<cv::Mat> neg_channels(3);
        for (int c = 0; c < 3; ++c) {
            neg_channels[c] = channels1[c] + channels2[c];
        }
        cv::merge(neg_channels, neg_image_f);

        // Escalar a [0,255] y convertir a 8 bits para guardar
        cv::Mat neg_image_8u;
        neg_image_f.convertTo(neg_image_8u, CV_8UC3, 255.0);

        std::string filename = negative_directory + "/negative_" + std::to_string(i) + ".png";
        cv::imwrite(filename, neg_image_8u);

        // Mostrar la imagen negativa generada y las imágenes originales (opcional)
        if (i < 2) { // Mostrar los dos primeros pares
            cv::imshow("Imagen Positiva 1", img1);
            cv::imshow("Imagen Positiva 2", img2);
            cv::imshow("Imagen Negativa", neg_image_8u);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}

int main() {
    try {
        std::string positive_directory = "positive_images";
        std::string negative_directory = "negative_images";
        int num_negative_images = 10000; // Número de imágenes negativas a generar
        float min_frequency = 0.002f;      // Frecuencia mínima para el ruido Simplex
        float max_frequency = 0.1f;       // Frecuencia máxima para el ruido Simplex

        // Generar imágenes negativas
        generateNegativeImages(positive_directory, negative_directory, num_negative_images, min_frequency, max_frequency);
        std::cout << "Imágenes negativas generadas correctamente.\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return -1;
    }
}
