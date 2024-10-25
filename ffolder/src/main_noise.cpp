// src/main_noise.cpp
#include "image_utils.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    try {
        std::string positive_directory = "data/positive_images";
        std::string negative_directory = "data/negative_images";
        int num_positive_images = 60000;
        int num_negative_images = 60000;
        cv::Size image_size(32, 32);
        float min_frequency = 0.05f;
        float max_frequency = 0.1f;
        std::string bias_type = "none"; // Opciones: none, random_color_random_position, fixed_color_random_position, fixed_color_fixed_position
        int line_thickness = 2;
        cv::Scalar fixed_color; // Inicializar sin valores
        cv::Point fixed_position(0, 0); // Posición por defecto para el sesgo fijo
        int num_channels = 1; // Valor predeterminado

        // Asignar fixed_color según el número de canales
        // Esto se hará más adelante después de conocer num_channels

        if (argc < 2) {
            std::cerr << "Uso: " << argv[0] << " [all|positive|negative] [bias_type] [line_thickness] [num_channels]\n";
            std::cerr << "Ejemplo: " << argv[0] << " all none 2 3\n";
            return -1;
        }

        std::string mode = argv[1];
        if (argc > 2) {
            bias_type = argv[2];
        }
        if (argc > 3) {
            line_thickness = std::stoi(argv[3]);
        }
        if (argc > 4) {
            num_channels = std::stoi(argv[4]);
            if (num_channels < 1 || num_channels > 3) {
                std::cerr << "Número de canales inválido: " << num_channels << ". Debe ser 1, 2 o 3.\n";
                return -1;
            }
        }

        // Asignar fixed_color según num_channels
        if (num_channels == 1) {
            fixed_color = cv::Scalar(255);
        } else if (num_channels == 2) {
            fixed_color = cv::Scalar(255, 255);
        } else { // num_channels == 3
            fixed_color = cv::Scalar(255, 255, 255);
        }

        if (mode == "all") {
            generatePositiveImages(positive_directory, num_positive_images, image_size, num_channels);
            std::cout << "Imágenes positivas generadas correctamente en: " << positive_directory << "\n";
            generateNegativeImages(positive_directory, negative_directory, num_negative_images, min_frequency, max_frequency, bias_type, line_thickness, fixed_color, fixed_position, num_channels);
            std::cout << "Imágenes negativas generadas correctamente en: " << negative_directory << "\n";
        }
        else if (mode == "positive") {
            generatePositiveImages(positive_directory, num_positive_images, image_size, num_channels);
            std::cout << "Imágenes positivas generadas correctamente en: " << positive_directory << "\n";
        }
        else if (mode == "negative") {
            generateNegativeImages(positive_directory, negative_directory, num_negative_images, min_frequency, max_frequency, bias_type, line_thickness, fixed_color, fixed_position, num_channels);
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
