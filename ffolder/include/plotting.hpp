#ifndef PLOTTING_HPP
#define PLOTTING_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "scatter_plot_data.hpp"
#include "neural_network.hpp" // for Dataset, FullyConnectedLayer

/**
 * @brief Grafica un histograma combinado de "goodness" para positivos y negativos.
 * @param goodness_positive_vals Vector con valores de bondad de muestras positivas.
 * @param goodness_negative_vals Vector con valores de bondad de muestras negativas.
 * @param threshold Umbral para dibujar la línea vertical.
 * @param save_file Ruta del archivo PNG donde se guardará el resultado.
 */
void plotGoodnessHistogramsCombined(const std::vector<float>& goodness_positive_vals,
                                    const std::vector<float>& goodness_negative_vals,
                                    float threshold,
                                    const std::string& save_file);

/**
 * @brief Grafica histogramas separados de "goodness" para positivos y negativos.
 * @param goodness_positive_vals Vector con valores de bondad de muestras positivas.
 * @param goodness_negative_vals Vector con valores de bondad de muestras negativas.
 * @param threshold Umbral para dibujar la línea vertical.
 * @param save_path Carpeta o ruta donde se guardan los histogramas (se generan 2 archivos).
 */
void plotGoodnessHistograms(const std::vector<float>& goodness_positive_vals,
                            const std::vector<float>& goodness_negative_vals,
                            float threshold,
                            const std::string& save_path);

/**
 * @brief Visualiza las salidas de una capa con PCA y muestra un scatter plot interactivo.
 * @param layer Capa que se evaluará.
 * @param val_positive_samples Muestras positivas de validación.
 * @param val_negative_samples Muestras negativas de validación.
 * @param num_components 2 o 3.
 * @param threshold Umbral utilizado para calcular la bondad.
 */
void visualizePCA(FullyConnectedLayer& layer, 
                  Dataset& val_positive_samples,
                  Dataset& val_negative_samples,
                  int num_components,
                  float threshold);

#endif // PLOTTING_HPP
