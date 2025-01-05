// include/image_utils.hpp

#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <string>
#include <opencv2/opencv.hpp>

/**
 * @file image_utils.hpp
 * @brief Funciones para la generación de imágenes sintéticas positivas y negativas.
 */

/**
 * @brief Genera imágenes positivas sintéticas con gaussianas superpuestas.
 *
 * Esta función crea un conjunto de imágenes positivas almacenadas en el directorio especificado.
 * Cada imagen contiene un número determinado de gaussianas dibujadas sobre un color de fondo aleatorio.
 *
 * @param directory Directorio donde se guardarán las imágenes positivas generadas.
 * @param num_images Número de imágenes positivas a generar.
 * @param image_size Tamaño de cada imagen (ancho x alto).
 * @param num_channels Número de canales de color de las imágenes (por defecto 3).
 *
 * @throws std::invalid_argument Si el número de canales no es soportado.
 * @throws cv::Exception Si ocurre un error al guardar las imágenes.
 */
void generatePositiveImages(const std::string& directory, 
                           int num_images, 
                           cv::Size image_size, 
                           int num_channels = 3);

/**
 * @brief Genera imágenes negativas sintéticas basadas en imágenes positivas existentes.
 *
 * Esta función combina aleatoriamente pares de imágenes positivas para crear imágenes negativas.
 * Además, aplica máscaras de ruido generadas dinámicamente para mezclar los canales de las imágenes
 * positivas seleccionadas. Opcionalmente, se pueden dibujar líneas con diferentes tipos de sesgo
 * (bias) sobre las imágenes negativas generadas.
 *
 * @param positive_directory Directorio que contiene las imágenes positivas existentes.
 * @param negative_directory Directorio donde se guardarán las imágenes negativas generadas.
 * @param num_images Número de imágenes negativas a generar.
 * @param min_frequency Frecuencia mínima para el generador de ruido.
 * @param max_frequency Frecuencia máxima para el generador de ruido.
 * @param bias_type Tipo de sesgo a aplicar a las líneas dibujadas en las imágenes negativas.
 *        - "none": No se dibujan líneas.
 *        - "random_color_random_position": Líneas con color y posición aleatorios.
 *        - "fixed_color_random_position": Líneas con color fijo y posición aleatoria.
 *        - "fixed_color_fixed_position": Líneas con color y posición fijos.
 *        - "random_color_fixed_position": Líneas con color aleatorio y posición fija.
 * @param line_thickness Grosor de las líneas dibujadas (si se aplica un sesgo).
 * @param fixed_color Color fijo para las líneas (utilizado en ciertos tipos de sesgo).
 * @param fixed_position Posición fija para las líneas (utilizado en ciertos tipos de sesgo).
 * @param num_channels Número de canales de color de las imágenes negativas (por defecto 3).
 *
 * @throws std::runtime_error Si no hay suficientes imágenes positivas o si ocurre un error al leerlas.
 * @throws std::invalid_argument Si el número de canales no es soportado.
 * @throws cv::Exception Si ocurre un error al guardar las imágenes.
 */
void generateNegativeImages(const std::string& positive_directory, 
                            const std::string& negative_directory, 
                            int num_images, 
                            float min_frequency, 
                            float max_frequency, 
                            const std::string& bias_type,
                            int line_thickness,
                            const cv::Scalar& fixed_color,
                            const cv::Point& fixed_position,
                            int num_channels = 3);

#endif // IMAGE_UTILS_HPP
