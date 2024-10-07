// src/main_noise.cpp

#include "image_utils.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        // Configuraciones por defecto
        std::string positive_directory = "data/positive_images";
        std::string negative_directory = "data/negative_images";
        int num_positive_images = 10000; // Número de imágenes positivas a generar
        int num_negative_images = 10000; // Número de imágenes negativas a generar
        cv::Size image_size(64, 64);     // Tamaño de las imágenes
        float min_frequency = 0.002f;     // Frecuencia mínima para el ruido Simplex
        float max_frequency = 0.3f;       // Frecuencia máxima para el ruido Simplex

        // Determinar la acción basada en los argumentos
        if (argc < 2) {
            std::cerr << "Uso: " << argv[0] << " [all|positive|negative]\n";
            return -1;
        }

        std::string mode = argv[1];

        if (mode == "all") {
            // Generar imágenes positivas
            generatePositiveImages(positive_directory, num_positive_images, image_size);
            std::cout << "Imágenes positivas generadas correctamente en: " << positive_directory << "\n";

            // Generar imágenes negativas
            generateNegativeImages(positive_directory, negative_directory, num_negative_images, min_frequency, max_frequency);
            std::cout << "Imágenes negativas generadas correctamente en: " << negative_directory << "\n";
        }
        else if (mode == "positive") {
            // Generar solo imágenes positivas
            generatePositiveImages(positive_directory, num_positive_images, image_size);
            std::cout << "Imágenes positivas generadas correctamente en: " << positive_directory << "\n";
        }
        else if (mode == "negative") {
            // Generar solo imágenes negativas
            generateNegativeImages(positive_directory, negative_directory, num_negative_images, min_frequency, max_frequency);
            std::cout << "Imágenes negativas generadas correctamente en: " << negative_directory << "\n";
        }
        else {
            std::cerr << "Modo desconocido: " << mode << "\n";
            std::cerr << "Usos válidos: all, positive, negative\n";
            return -1;
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return -1;
    }
}
