// src/main_noise.cpp
#include "image_utils.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    try {
        // Directorios raíz
        std::string positive_directory_root = "data/positive_images";
        std::string negative_directory_root = "data/negative_images";
        
        // Directorios para train y val
        std::string positive_train_dir = positive_directory_root + "/train";
        std::string positive_val_dir = positive_directory_root + "/val";
        std::string negative_train_dir = negative_directory_root + "/train";
        std::string negative_val_dir = negative_directory_root + "/val";

        // Este valor es la suma de la cantidad de archivos entre validación y entrenamiento
        int num_positive_images = 70000;
        int num_negative_images = 70000;

        // La variable propor controla cuantos archivos son de validación
        float propor = 6.0f / 7.0f;

        // Demás parámetros para la síntesis
        cv::Size image_size(32, 32);
        float min_frequency = 0.1f;
        float max_frequency = 0.15f;
        std::string bias_type = "none"; 
        int line_thickness = 2;
        cv::Scalar fixed_color; 
        cv::Point fixed_position(0, 0); 
        int num_channels = 1; 

        // Le avisamos al usuario si usó mal el script
        if (argc < 2) {
            std::cerr << "Uso: " << argv[0] << " [all|positive|negative] [bias_type] [line_thickness] [num_channels]\n";
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

        if (num_channels == 1) {
            fixed_color = cv::Scalar(255);
        } else if (num_channels == 2) {
            fixed_color = cv::Scalar(255, 255);
        } else {
            fixed_color = cv::Scalar(255, 255, 255);
        }

        // Dividir imágenes: según propor
        int num_positive_train = static_cast<int>(num_positive_images * propor );
        int num_positive_val = num_positive_images - num_positive_train;
        int num_negative_train = static_cast<int>(num_negative_images * propor );
        int num_negative_val = num_negative_images - num_negative_train;

        if (mode == "all" || mode == "positive") {
            generatePositiveImages(positive_train_dir, num_positive_train, image_size, num_channels);
            generatePositiveImages(positive_val_dir, num_positive_val, image_size, num_channels);
            std::cout << "Imágenes positivas generadas correctamente en: " << positive_directory_root << "\n";
        }

        if (mode == "all" || mode == "negative") {
            generateNegativeImages(positive_train_dir, negative_train_dir, num_negative_train, min_frequency, max_frequency, bias_type, line_thickness, fixed_color, fixed_position, num_channels);
            generateNegativeImages(positive_val_dir, negative_val_dir, num_negative_val, min_frequency, max_frequency, bias_type, line_thickness, fixed_color, fixed_position, num_channels);
            std::cout << "Imágenes negativas generadas correctamente en: " << negative_directory_root << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return -1;
    }
}
