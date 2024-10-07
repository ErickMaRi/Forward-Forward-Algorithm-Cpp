// include/image_utils.hpp

#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief Genera imágenes positivas utilizando puntos gaussianos aleatorios con fondo aleatorio.
 * @param directory Directorio donde se guardarán las imágenes.
 * @param num_images Número de imágenes a generar.
 * @param image_size Tamaño de las imágenes a generar.
 */
void generatePositiveImages(const std::string& directory, int num_images, cv::Size image_size);

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
                            float max_frequency);

#endif // IMAGE_UTILS_HPP
