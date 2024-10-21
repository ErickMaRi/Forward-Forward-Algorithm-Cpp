//src/main_noise.cpp
#include "image_utils.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        std::string positive_directory = "data/positive_images";
        std::string negative_directory = "data/negative_images";
        int num_positive_images = 10000;
        int num_negative_images = 10000;
        cv::Size image_size(32, 32);
        float min_frequency = 0.05f;
        float max_frequency = 0.4f;
        std::string bias_type = "none"; // Opciones: none, random_color_random_position, fixed_color_random_position, fixed_color_fixed_position
        int line_thickness = 2;
        cv::Scalar fixed_color(255, 255, 255); // Color por defecto para el sesgo
        cv::Point fixed_position(0, 0); // Posición por defecto para el sesgo fijo

        if (argc < 2) {
            std::cerr << "Uso: " << argv[0] << " [all|positive|negative] [bias_type] [line_thickness]\n";
            return -1;
        }

        std::string mode = argv[1];
        if (argc > 2) {
            bias_type = argv[2];
        }
        if (argc > 3) {
            line_thickness = std::stoi(argv[3]);
        }

        if (mode == "all") {
            generatePositiveImages(positive_directory, num_positive_images, image_size);
            std::cout << "Imágenes positivas generadas correctamente en: " << positive_directory << "\n";
            generateNegativeImages(positive_directory, negative_directory, num_negative_images, min_frequency, max_frequency, bias_type, line_thickness, fixed_color, fixed_position);
            std::cout << "Imágenes negativas generadas correctamente en: " << negative_directory << "\n";
        }
        else if (mode == "positive") {
            generatePositiveImages(positive_directory, num_positive_images, image_size);
            std::cout << "Imágenes positivas generadas correctamente en: " << positive_directory << "\n";
        }
        else if (mode == "negative") {
            generateNegativeImages(positive_directory, negative_directory, num_negative_images, min_frequency, max_frequency, bias_type, line_thickness, fixed_color, fixed_position);
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
