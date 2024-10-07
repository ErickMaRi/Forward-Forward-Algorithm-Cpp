// src/main_noise.cpp

#include "image_utils.hpp"
#include <iostream>

int main() {
    try {
        std::string positive_directory = "data/positive_images";
        std::string negative_directory = "data/negative_images";
        int num_positive_images = 10000; // Número de imágenes positivas a generar
        int num_negative_images = 10000; // Número de imágenes negativas a generar
        cv::Size image_size(64, 64);     // Tamaño de las imágenes
        float min_frequency = 0.002f;     // Frecuencia mínima para el ruido Simplex
        float max_frequency = 0.3f;       // Frecuencia máxima para el ruido Simplex

        // Generar imágenes positivas
        generatePositiveImages(positive_directory, num_positive_images, image_size);
        std::cout << "Imágenes positivas generadas correctamente en: " << positive_directory << "\n";

        // Generar imágenes negativas
        generateNegativeImages(positive_directory, negative_directory, num_negative_images, min_frequency, max_frequency);
        std::cout << "Imágenes negativas generadas correctamente en: " << negative_directory << "\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return -1;
    }
}
