//src/main_train.cpp

#define EIGEN_USE_THREADS
#include "neural_network.hpp"
#include "optimizer.hpp"
#include "scatter_plot_data.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <thread> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <omp.h>
#include <fstream>

namespace fs = std::filesystem;

// ================================
// Funciones Auxiliares y Principal
// ================================

// Definir la función de activación y su derivada (Leaky ReLU) correctamente
std::function<float(float)> activation = [](float x) -> float {
    const float alpha = 1.0f / 64.0f; // 2^-6
    return (x >= 0.0f) ? x : alpha * x; // Leaky ReLU
};

std::function<float(float)> activation_derivative = [](float x) -> float {
    const float alpha = 1.0f / 64.0f; // 2^-6
    return (x >= 0.0f) ? 1.0f : alpha; // Derivada de Leaky ReLU
};

/**
 * @brief Genera una permutación fija para los datos de entrada.
 * @param input_size Tamaño del vector de entrada.
 * @return Vector que representa la permutación.
 */
std::vector<size_t> generateFixedPermutation(size_t input_size) {
    std::vector<size_t> permutation(input_size);
    std::iota(permutation.begin(), permutation.end(), 0); // Inicializa con 0, 1, 2, ..., input_size-1

    // Utiliza un generador con semilla fija para reproducibilidad
    std::mt19937 rng(42); // Semilla fija
    std::shuffle(permutation.begin(), permutation.end(), rng);

    return permutation;
}

/**
 * @brief Aplica una permutación fija a una imagen.
 * @param image Vector de entrada que representa la imagen.
 * @param permutation Vector que define la permutación de los índices.
 * @return Vector permutado.
 */
Eigen::VectorXf applyPermutation(const Eigen::VectorXf& image, const std::vector<size_t>& permutation) {
    Eigen::VectorXf permuted_image(image.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        permuted_image[i] = image[permutation[i]];
    }
    return permuted_image;
}

/**
 * @brief Permite al usuario seleccionar un optimizador.
 * @return Puntero compartido al optimizador seleccionado.
 */
std::shared_ptr<Optimizer> selectOptimizer() {
    int optimizer_choice;
    std::cout << "\nSeleccione el optimizador:\n";
    std::cout << "1. SGD Optimizer\n";
    std::cout << "2. Adam Optimizer\n";
    std::cout << "3. Low Pass Filter Optimizer\n";
    std::cout << "4. AdaBelief Optimizer\n";
    std::cout << "Ingrese su elección: ";
    std::cin >> optimizer_choice;

    switch (optimizer_choice) {
        case 1: {
            float learning_rate, momentum;

            std::cout << "Ingrese el learning rate para SGD Optimizer (ej. 0.0001): ";
            std::cin >> learning_rate;
            std::cout << "Ingrese el valor de momentum (ej. 0.9 o 0): ";
            std::cin >> momentum;

            return std::make_shared<SGDOptimizer>(learning_rate, momentum);
        }
        case 2: {
            float learning_rate, beta1, beta2, epsilon;

            std::cout << "Ingrese el learning rate para Adam Optimizer (ej. 0.0001): ";
            std::cin >> learning_rate;

            std::cout << "Ingrese beta1 (ej. 0.9): ";
            std::cin >> beta1;
            std::cout << "Ingrese beta2 (ej. 0.999): ";
            std::cin >> beta2;
            std::cout << "Ingrese epsilon (ej. 1e-8): ";
            std::cin >> epsilon;

            return std::make_shared<AdamOptimizer>(learning_rate, beta1, beta2, epsilon);
        }
        case 3: {
            float learning_rate, alpha;

            std::cout << "Ingrese el learning rate para Low Pass Filter Optimizer (ej. 0.0001): ";
            std::cin >> learning_rate;
            std::cout << "Ingrese el valor de alpha (ej. 0.1): ";
            std::cin >> alpha;

            return std::make_shared<LowPassFilterOptimizer>(learning_rate, alpha);
        }
        case 4: {
            float learning_rate, beta1, beta2, epsilon;

            std::cout << "Ingrese el learning rate para AdaBelief Optimizer (ej. 0.0001): ";
            std::cin >> learning_rate;
            std::cout << "Ingrese beta1 (ej. 0.9): ";
            std::cin >> beta1;
            std::cout << "Ingrese beta2 (ej. 0.999): ";
            std::cin >> beta2;
            std::cout << "Ingrese epsilon (ej. 1e-8): ";
            std::cin >> epsilon;

            return std::make_shared<AdaBeliefOptimizer>(learning_rate, beta1, beta2, epsilon);
        }
        default: {
            std::cout << "Opción inválida. Usando Adam Optimizer con valores por defecto.\n";
            return std::make_shared<AdamOptimizer>();
        }
    }
}


// ================================
// Funciones para Visualización
// ================================

/**
 * @brief Visualiza los datos usando PCA.
 */
void visualizePCA(FullyConnectedLayer& layer, Dataset& val_positive_samples,
                  Dataset& val_negative_samples, int num_components,
                  float threshold) {
    // Verificar que num_components sea 2 o 3
    if (num_components != 2 && num_components != 3) {
        throw std::invalid_argument("El número de componentes debe ser 2 o 3.");
    }

    size_t val_positive_size = val_positive_samples.getNumSamples();
    size_t val_negative_size = val_negative_samples.getNumSamples();
    size_t total_samples = val_positive_size + val_negative_size;

    size_t output_size = layer.getOutputSize();

    // Crear matriz para almacenar las salidas
    cv::Mat data(total_samples, output_size, CV_32F);
    std::vector<int> labels(total_samples);
    std::vector<std::string> image_paths(total_samples);
    std::vector<float> squared_magnitudes(total_samples);

    // Vectores para las magnitudes al cuadrado de cada conjunto
    std::vector<float> squared_magnitudes_positive;
    std::vector<float> squared_magnitudes_negative;

    // Recopilar salidas para muestras positivas
    size_t idx = 0;
    for (size_t i = 0; i < val_positive_size; ++i, ++idx) {
        const Eigen::VectorXf& input = val_positive_samples.getSample(i);
        Eigen::VectorXf output;
        layer.forward(input, output, false, true, threshold,
                      activation, activation_derivative);

        // Copiar la salida a la matriz de datos
        for (size_t j = 0; j < output_size; ++j) {
            data.at<float>(idx, j) = output[j];
        }
        labels[idx] = 1; // Positivo
        image_paths[idx] = val_positive_samples.getImagePath(i);

        // Calcular y almacenar la magnitud al cuadrado
        float squared_magnitude = output.squaredNorm();
        squared_magnitudes[idx] = squared_magnitude;
        squared_magnitudes_positive.push_back(squared_magnitude);
    }

    // Recopilar salidas para muestras negativas
    for (size_t i = 0; i < val_negative_size; ++i, ++idx) {
        const Eigen::VectorXf& input = val_negative_samples.getSample(i);
        Eigen::VectorXf output;
        layer.forward(input, output, false, false, threshold,
                      activation, activation_derivative);

        // Copiar la salida a la matriz de datos
        for (size_t j = 0; j < output_size; ++j) {
            data.at<float>(idx, j) = output[j];
        }
        labels[idx] = 0; // Negativo
        image_paths[idx] = val_negative_samples.getImagePath(i);

        // Calcular y almacenar la magnitud al cuadrado
        float squared_magnitude = output.squaredNorm();
        squared_magnitudes[idx] = squared_magnitude;
        squared_magnitudes_negative.push_back(squared_magnitude);
    }

    // Realizar PCA
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);

    // Proyectar datos
    cv::Mat projected_data = pca.project(data);

    // Encontrar mínimos y máximos para el escalado de las coordenadas PCA
    cv::Mat min_vals, max_vals;
    cv::reduce(projected_data, min_vals, 0, cv::REDUCE_MIN);
    cv::reduce(projected_data, max_vals, 0, cv::REDUCE_MAX);

    // Encontrar mínimos y máximos de las magnitudes al cuadrado para cada conjunto
    float min_squared_magnitude_positive = *std::min_element(squared_magnitudes_positive.begin(), squared_magnitudes_positive.end());
    float max_squared_magnitude_positive = *std::max_element(squared_magnitudes_positive.begin(), squared_magnitudes_positive.end());

    float min_squared_magnitude_negative = *std::min_element(squared_magnitudes_negative.begin(), squared_magnitudes_negative.end());
    float max_squared_magnitude_negative = *std::max_element(squared_magnitudes_negative.begin(), squared_magnitudes_negative.end());

    // Crear imagen para el scatter plot
    int img_size = 600;
    cv::Mat scatter_image(img_size, img_size, CV_8UC3,
                          cv::Scalar(255, 255, 255));

    // Función para mapear coordenadas PCA a píxeles
    auto mapToPixel = [&](float val, float min_val, float max_val) {
        return static_cast<int>((val - min_val) / (max_val - min_val) *
                                (img_size - 40) + 20);
    };

    ScatterPlotData plot_data;
    plot_data.image = scatter_image.clone();

    // Dibujar puntos
    for (size_t i = 0; i < total_samples; ++i) {
        int x = mapToPixel(projected_data.at<float>(i, 0),
                           min_vals.at<float>(0, 0),
                           max_vals.at<float>(0, 0));
        int y = mapToPixel(projected_data.at<float>(i, 1),
                           min_vals.at<float>(0, 1),
                           max_vals.at<float>(0, 1));

        // Invertir y para que el origen esté en la parte inferior
        y = img_size - y;

        cv::Point pt(x, y);
        plot_data.points.push_back(pt);
        plot_data.image_paths.push_back(image_paths[i]);

        // Mapear la magnitud al cuadrado al rango [0, 255] para el canal verde
        float normalized_value;
        int green_value;
        if (labels[i] == 1) {
            // Muestras positivas
            normalized_value = (squared_magnitudes[i] - min_squared_magnitude_positive) /
                               (max_squared_magnitude_positive - min_squared_magnitude_positive);
        } else {
            // Muestras negativas
            normalized_value = (squared_magnitudes[i] - min_squared_magnitude_negative) /
                               (max_squared_magnitude_negative - min_squared_magnitude_negative);
        }

        // Asegurarse de que el valor esté entre 0 y 1
        normalized_value = std::max(0.0f, std::min(1.0f, normalized_value));

        green_value = static_cast<int>(normalized_value * 255.0f);

        // Establecer color según la etiqueta y la magnitud
        cv::Scalar color;
        if (labels[i] == 1) {
            // Rojo y canal verde para la magnitud
            color = cv::Scalar(0, green_value, 255);
        } else {
            // Azul y canal verde para la magnitud
            color = cv::Scalar(255, green_value, 0);
        }

        cv::circle(plot_data.image, pt, 4, color, -1);
    }

    // Dibujar el origen (0,0)
    cv::Point origin(
        mapToPixel(0.0f, min_vals.at<float>(0, 0), max_vals.at<float>(0, 0)),
        img_size - mapToPixel(0.0f, min_vals.at<float>(0, 1),
                              max_vals.at<float>(0, 1))
    );
    cv::drawMarker(plot_data.image, origin, cv::Scalar(0, 0, 0),
                   cv::MARKER_CROSS, 20, 2);

    // Mostrar la imagen
    cv::namedWindow("Scatter Plot", cv::WINDOW_AUTOSIZE);

    // Función de callback para manejar los clics del mouse
    cv::setMouseCallback("Scatter Plot", [](int event, int x, int y,
                                            int flags, void* userdata) {
        if (event != cv::EVENT_LBUTTONDOWN) return;

        ScatterPlotData* plot_data = reinterpret_cast<ScatterPlotData*>(
                                     userdata);
        cv::Point click_point(x, y);

        // Encontrar el punto más cercano
        double min_dist = std::numeric_limits<double>::max();
        size_t closest_idx = 0;

        for (size_t i = 0; i < plot_data->points.size(); ++i) {
            double dist = cv::norm(click_point - plot_data->points[i]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = i;
            }
        }

        // Si el clic está cerca de un punto
        if (min_dist <= 10.0) {
            // Cargar y mostrar la imagen correspondiente
            cv::Mat img = cv::imread(plot_data->image_paths[closest_idx]);
            if (!img.empty()) {
                cv::imshow("Imagen Seleccionada", img);
            } else {
                std::cerr << "No se pudo cargar la imagen: "
                          << plot_data->image_paths[closest_idx] << "\n";
            }
        }
    }, &plot_data);

    cv::imshow("Scatter Plot", plot_data.image);
    cv::waitKey(0);
}

/**
 * @brief Función para plotear histogramas combinados.
 */
void plotGoodnessHistogramsCombined(const std::vector<float>& goodness_positive_vals,
                                    const std::vector<float>& goodness_negative_vals,
                                    float threshold,
                                    const std::string& save_file) { // Renombrado a 'save_file'
    // Convertir los vectores a Mat de OpenCV
    cv::Mat goodness_positive = cv::Mat(goodness_positive_vals).reshape(1);
    cv::Mat goodness_negative = cv::Mat(goodness_negative_vals).reshape(1);

    // Definir los parámetros del histograma
    int histSize = 50; // Número de bins
    float max_val = std::max(*std::max_element(
                             goodness_positive_vals.begin(),
                             goodness_positive_vals.end()),
                             *std::max_element(
                             goodness_negative_vals.begin(),
                             goodness_negative_vals.end()));
    float range[] = { 0.0f, max_val };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    // Calcular los histogramas
    cv::Mat hist_positive, hist_negative;
    cv::calcHist(&goodness_positive, 1, 0, cv::Mat(), hist_positive, 1,
                &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&goodness_negative, 1, 0, cv::Mat(), hist_negative, 1,
                &histSize, &histRange, uniform, accumulate);

    // Normalizar los histogramas
    cv::normalize(hist_positive, hist_positive, 0, 400, cv::NORM_MINMAX);
    cv::normalize(hist_negative, hist_negative, 0, 400, cv::NORM_MINMAX);

    // Crear la imagen para el histograma combinado
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImageCombined(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Dibujar el histograma positivo en azul
    for (int i = 1; i < histSize; i++) {
        cv::line(histImageCombined,
            cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_positive.at<float>(i - 1))),
            cv::Point(bin_w * i, hist_h - cvRound(hist_positive.at<float>(i))),
            cv::Scalar(255, 0, 0), 2); // Azul para positivos
    }

    // Dibujar el histograma negativo en verde
    for (int i = 1; i < histSize; i++) {
        cv::line(histImageCombined,
            cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_negative.at<float>(i - 1))),
            cv::Point(bin_w * i, hist_h - cvRound(hist_negative.at<float>(i))),
            cv::Scalar(0, 255, 0), 2); // Verde para negativos
    }

    // Dibujar la línea del umbral
    float normalized_threshold = (threshold - range[0]) / (range[1] - range[0]);
    int threshold_x = cvRound(normalized_threshold * hist_w);
    cv::line(histImageCombined,
             cv::Point(threshold_x, 0),
             cv::Point(threshold_x, hist_h),
             cv::Scalar(0, 0, 0), 2); // Línea negra para el umbral

    // Añadir leyenda
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1.0;
    int thickness = 2;
    cv::putText(histImageCombined, "Positivos", cv::Point(20, 30),
                font, font_scale, cv::Scalar(255, 0, 0), thickness); // Azul
    cv::putText(histImageCombined, "Negativos", cv::Point(20, 70),
                font, font_scale, cv::Scalar(0, 255, 0), thickness); // Verde
    cv::putText(histImageCombined, "Umbral", cv::Point(threshold_x + 10, 20),
                font, font_scale, cv::Scalar(0, 0, 0), thickness); // Negro

    // Extraer la carpeta de la ruta del archivo
    fs::path p(save_file);
    fs::create_directories(p.parent_path());

    // Guardar la imagen combinada
    cv::imwrite(save_file, histImageCombined);
}

/**
 * @brief Función para plotear histogramas separados (si es necesario).
 */
void plotGoodnessHistograms(const std::vector<float>& goodness_positive_vals,
                            const std::vector<float>& goodness_negative_vals,
                            float threshold,
                            const std::string& save_path) { // Añadido 'save_path'
    // Convertir los vectores a Mat de OpenCV
    cv::Mat goodness_positive = cv::Mat(goodness_positive_vals).reshape(1);
    cv::Mat goodness_negative = cv::Mat(goodness_negative_vals).reshape(1);

    // Definir los parámetros del histograma
    int histSize = 50; // Número de bins
    float max_val = std::max(*std::max_element(
                             goodness_positive_vals.begin(),
                             goodness_positive_vals.end()),
                             *std::max_element(
                             goodness_negative_vals.begin(),
                             goodness_negative_vals.end()));
    float range[] = { 0.0f, max_val }; // Ajustar el rango máximo
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    // Calcular los histogramas
    cv::Mat hist_positive, hist_negative;
    cv::calcHist(&goodness_positive, 1, 0, cv::Mat(), hist_positive, 1,
                 &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&goodness_negative, 1, 0, cv::Mat(), hist_negative, 1,
                 &histSize, &histRange, uniform, accumulate);

    // Normalizar los histogramas
    cv::normalize(hist_positive, hist_positive, 0, 400, cv::NORM_MINMAX);
    cv::normalize(hist_negative, hist_negative, 0, 400, cv::NORM_MINMAX);

    // Crear las imágenes para los histogramas
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImagePositive(hist_h, hist_w, CV_8UC3,
                              cv::Scalar(255, 255, 255));
    cv::Mat histImageNegative(hist_h, hist_w, CV_8UC3,
                              cv::Scalar(255, 255, 255));

    // Dibujar los histogramas
    for (int i = 1; i < histSize; i++) {
        // Histograma positivo
        cv::line(histImagePositive,
            cv::Point(bin_w * (i - 1), hist_h -
                      cvRound(hist_positive.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h -
                      cvRound(hist_positive.at<float>(i))),
            cv::Scalar(255, 0, 0), 2);

        // Histograma negativo
        cv::line(histImageNegative,
            cv::Point(bin_w * (i - 1), hist_h -
                      cvRound(hist_negative.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h -
                      cvRound(hist_negative.at<float>(i))),
            cv::Scalar(0, 255, 0), 2);
    }

    // Dibujar la línea del umbral
    float normalized_threshold = (threshold - range[0]) / (range[1] - range[0]);
    int threshold_x = cvRound(normalized_threshold * hist_w);
    cv::line(histImagePositive,
             cv::Point(threshold_x, 0),
             cv::Point(threshold_x, hist_h),
             cv::Scalar(0, 0, 0), 2);

    cv::line(histImageNegative,
             cv::Point(threshold_x, 0),
             cv::Point(threshold_x, hist_h),
             cv::Scalar(0, 0, 0), 2);

    // Añadir texto a las imágenes
    cv::putText(histImagePositive, "Positivos", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    cv::putText(histImageNegative, "Negativos", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);

    // Crear la carpeta de destino si no existe
    fs::create_directories(save_path);

    // Guardar las imágenes
    std::string pos_hist_path = save_path + "/Histograma_Positive.png";
    std::string neg_hist_path = save_path + "/Histograma_Negative.png";
    cv::imwrite(pos_hist_path, histImagePositive);
    cv::imwrite(neg_hist_path, histImageNegative);
}

/**
 * @brief Función principal que maneja el flujo de entrenamiento del modelo.
 */
void trainModel() {
    try {
        std::cout << "Cargando conjuntos de datos...\n";
        
        // Cargamos directamente datos de entrenamiento y validación de carpetas separadas
        Dataset train_positive_samples("data/positive_images/train/");
        Dataset val_positive_samples("data/positive_images/val/");
        Dataset train_negative_samples("data/negative_images/train/");
        Dataset val_negative_samples("data/negative_images/val/");

        if (train_positive_samples.getNumSamples() == 0 ||
            val_positive_samples.getNumSamples() == 0 ||
            train_negative_samples.getNumSamples() == 0 ||
            val_negative_samples.getNumSamples() == 0) {
            throw std::runtime_error("Algún conjunto de datos (train o val) está vacío.");
        }

        size_t input_size = train_positive_samples.getInputSize(); // Asegúrate de que esto corresponde al tamaño de cada imagen
        std::vector<size_t> permutation = generateFixedPermutation(input_size);

        std::shared_ptr<Optimizer> optimizer = selectOptimizer();

        bool dynamic_threshold = false;
        std::cout << "¿Desea utilizar un umbral dinámico? (1 = Sí, 0 = No): ";
        int threshold_choice;
        std::cin >> threshold_choice;
        dynamic_threshold = (threshold_choice == 1);

        float threshold;
        std::cout << "Ingrese el umbral inicial para determinar la bondad: ";
        std::cin >> threshold;

        float best_overall_threshold = threshold;

        size_t total_epochs_long;
        std::cout << "Ingrese el número total de épocas de entrenamiento largo: ";
        std::cin >> total_epochs_long;

        std::vector<float> goodness_positive_vals;
        std::vector<float> goodness_negative_vals;

        size_t output_size;
        std::cout << "Ingrese el tamaño de la capa (número de neuronas): ";
        std::cin >> output_size;

        size_t num_initializations;
        size_t initial_epochs;
        std::cout << "Ingrese el número de inicializaciones iniciales: ";
        std::cin >> num_initializations;
        std::cout << "Ingrese el número de épocas por inicialización (1 a 4): ";
        std::cin >> initial_epochs;

        size_t ntop;
        std::cout << "Ingrese el número de mejores modelos a recordar (ntop): ";
        std::cin >> ntop;

        size_t patience;
        std::cout << "Ingrese el número de épocas de tolerancia sin mejora (patience): ";
        std::cin >> patience;

        fs::create_directories("histograms");
        fs::create_directories("ntop_cache");
        // Lista para almacenar los mejores ntop modelos y sus puntuaciones
        std::vector<std::pair<double, std::string>> top_models; // <score, filepath>

        // Realizar múltiples inicializaciones
        for (size_t init = 0; init < num_initializations; ++init) {
            std::cout << "\n--- Inicialización " << (init + 1) << "/" << num_initializations << " ---\n";

            // Crear una nueva capa con pesos inicializados aleatoriamente
            FullyConnectedLayer current_layer(input_size, output_size, optimizer);

            // Inicialización de variables para el seguimiento del mejor modelo en esta inicialización
            double best_score_init = -std::numeric_limits<double>::infinity();
            FullyConnectedLayer best_init_layer = current_layer; // Copia inicial del modelo
            float best_init_threshold = threshold;
            size_t epochs_without_improvement_init = 0;

            // Entrenar por el número de épocas especificado para la inicialización
            for (size_t epoch = 0; epoch < initial_epochs; ++epoch) {
                std::cout << "\n--- Inicialización " << (init + 1) << " - Época " << (epoch + 1) << "/" << initial_epochs << " ---\n";

                // Mezclar y barajar los datos positivos y negativos
                train_positive_samples.shuffle();
                train_negative_samples.shuffle();

                // Crear una lista combinada de todas las muestras con sus etiquetas
                std::vector<std::pair<std::reference_wrapper<const Eigen::VectorXf>, bool>> combined_train_samples;
                combined_train_samples.reserve(train_positive_samples.getNumSamples() + train_negative_samples.getNumSamples());

                for (size_t i = 0; i < train_positive_samples.getNumSamples(); ++i) {
                    combined_train_samples.emplace_back(train_positive_samples.getSample(i), true); // Positivo
                }
                for (size_t i = 0; i < train_negative_samples.getNumSamples(); ++i) {
                    combined_train_samples.emplace_back(train_negative_samples.getSample(i), false); // Negativo
                }

                // Barajar la lista combinada
                {
                    std::random_device rd;
                    std::mt19937 g(rd());
                    std::shuffle(combined_train_samples.begin(), combined_train_samples.end(), g);
                }

                // Entrenamiento con paralelización y sección crítica para actualizar pesos
                // #pragma omp parallel for
                for (size_t i = 0; i < combined_train_samples.size(); ++i) {
                    const Eigen::VectorXf& input = combined_train_samples[i].first.get();
                    bool is_positive = combined_train_samples[i].second;
                    Eigen::VectorXf output;
                    // #pragma omp critical
                    {
                        current_layer.forward(input, output, true, is_positive, threshold, activation, activation_derivative);
                    }
                }

                // Evaluación en conjunto de validación
                size_t correct_positive = 0;
                size_t correct_negative = 0;

                goodness_positive_vals.clear();
                goodness_negative_vals.clear();

                // Evaluación en muestras positivas - paralelizado
                {
                    std::vector<float> local_goodness_positive;
                    // #pragma omp parallel for reduction(+:correct_positive)
                    for (size_t i = 0; i < val_positive_samples.getNumSamples(); ++i) {
                        const Eigen::VectorXf& input = val_positive_samples.getSample(i);
                        Eigen::VectorXf output;
                        current_layer.forward(input, output, false, true, threshold, activation, activation_derivative);

                        float goodness = output.squaredNorm();
                        if (goodness > threshold) {
                            correct_positive++;
                        }

                        // #pragma omp critical
                        {
                            local_goodness_positive.push_back(goodness);
                        }
                    }
                    // Mover los datos locales fuera de la sección paralela
                    goodness_positive_vals = std::move(local_goodness_positive);
                }

                // Evaluación en muestras negativas - paralelizado
                {
                    std::vector<float> local_goodness_negative;
                    // #pragma omp parallel for reduction(+:correct_negative)
                    for (size_t i = 0; i < val_negative_samples.getNumSamples(); ++i) {
                        const Eigen::VectorXf& input = val_negative_samples.getSample(i);
                        Eigen::VectorXf output;
                        current_layer.forward(input, output, false, false, threshold, activation, activation_derivative);

                        float goodness = output.squaredNorm();
                        if (goodness < threshold) {
                            correct_negative++;
                        }

                        // #pragma omp critical
                        {
                            local_goodness_negative.push_back(goodness);
                        }
                    }
                    goodness_negative_vals = std::move(local_goodness_negative);
                }

                // Calcula la precisión
                double accuracy = (static_cast<double>(correct_positive + correct_negative) /
                                  (val_positive_samples.getNumSamples() + val_negative_samples.getNumSamples())) * 100.0;

                double accuracy_positive = (static_cast<double>(correct_positive) /
                                           val_positive_samples.getNumSamples()) * 100.0;

                double accuracy_negative = (static_cast<double>(correct_negative) /
                                           val_negative_samples.getNumSamples()) * 100.0;

                std::cout << "Precisión en validación: " << accuracy << "%\n"
                          << "Precisión en positivos: " << accuracy_positive << "%\n"
                          << "Precisión en negativos: " << accuracy_negative << "%\n";

                // Verifica si esta es la mejor precisión hasta ahora en esta inicialización
                if (accuracy > best_score_init) {
                    best_score_init = accuracy;
                    epochs_without_improvement_init = 0;

                    // Guarda el mejor modelo de esta inicialización
                    best_init_layer = current_layer;
                    best_init_threshold = threshold;
                } else {
                    epochs_without_improvement_init++;
                }

                // Revertir al mejor modelo si no hay mejora en 'patience' épocas
                if (epochs_without_improvement_init >= patience) {
                    std::cout << "No hay mejora en las últimas " << patience << " épocas. Revirtiendo al mejor modelo de esta inicialización.\n";
                    current_layer = best_init_layer;
                    threshold = best_init_threshold;
                    epochs_without_improvement_init = 0; // Reinicia el contador
                    break; // Salir del bucle de épocas para esta inicialización
                }

                // Ajusta dinámicamente el umbral si está habilitado
                if (dynamic_threshold && !goodness_positive_vals.empty() && !goodness_negative_vals.empty()) {
                    float avg_goodness_positive = std::accumulate(goodness_positive_vals.begin(),
                                                                  goodness_positive_vals.end(), 0.0f) / goodness_positive_vals.size();
                    float avg_goodness_negative = std::accumulate(goodness_negative_vals.begin(),
                                                                  goodness_negative_vals.end(), 0.0f) / goodness_negative_vals.size();

                    threshold = (avg_goodness_positive + avg_goodness_negative) / 2.0f;
                    std::cout << "Umbral ajustado a: " << threshold << "\n";
                }

                // Guardar los histogramas combinados de esta época con el nombre único
                std::string hist_filename = "histograms/Init_" + std::to_string(init + 1) +
                                            "_Epoch_" + std::to_string(epoch + 1) + ".png";

                plotGoodnessHistogramsCombined(goodness_positive_vals,
                                               goodness_negative_vals,
                                               threshold,
                                               hist_filename);

                std::cout << "Histograma combinado guardado en: " << hist_filename << "\n";
            }

            // Guardar el mejor modelo de esta inicialización en la carpeta ntop_cache
            std::string model_filename = "ntop_cache/model_init_" + std::to_string(init + 1) + ".bin";
            best_init_layer.saveModel(model_filename);

            // Agregar el modelo y su puntuación a la lista de top_models
            top_models.emplace_back(best_score_init, model_filename);

            // Ordenar la lista de top_models y mantener solo los ntop mejores
            std::sort(top_models.begin(), top_models.end(),
                      [](const std::pair<double, std::string>& a, const std::pair<double, std::string>& b) {
                          return a.first > b.first; // Orden descendente por puntuación
                      });

            if (top_models.size() > ntop) {
                // Eliminar los modelos adicionales y sus archivos para liberar espacio
                for (size_t i = ntop; i < top_models.size(); ++i) {
                    fs::remove(top_models[i].second); // Eliminar el archivo del modelo
                }
                top_models.resize(ntop); // Mantener solo los ntop mejores
            }

            std::cout << "Modelo guardado en: " << model_filename << "\n";
        }

        // Evaluación del mejor modelo individual
        std::cout << "\n--- Evaluando el mejor modelo individual en el conjunto de validación ---\n";
        FullyConnectedLayer best_individual_model(input_size, output_size, optimizer);
        best_individual_model.loadModel(top_models.front().second);

        size_t correct_positive_individual = 0;
        size_t correct_negative_individual = 0;

        // Evaluación en muestras positivas (paralelizado)
        {
            // #pragma omp parallel for reduction(+:correct_positive_individual)
            for (size_t i = 0; i < val_positive_samples.getNumSamples(); ++i) {
                const Eigen::VectorXf& input = val_positive_samples.getSample(i);
                Eigen::VectorXf output;
                best_individual_model.forward(input, output, false, true, threshold, activation, activation_derivative);

                float goodness = output.squaredNorm();

                if (goodness > threshold) {
                    correct_positive_individual++;
                }
            }
        }

        // Evaluación en muestras negativas (paralelizado)
        {
            // #pragma omp parallel for reduction(+:correct_negative_individual)
            for (size_t i = 0; i < val_negative_samples.getNumSamples(); ++i) {
                const Eigen::VectorXf& input = val_negative_samples.getSample(i);
                Eigen::VectorXf output;
                best_individual_model.forward(input, output, false, false, threshold, activation, activation_derivative);

                float goodness = output.squaredNorm();

                if (goodness < threshold) {
                    correct_negative_individual++;
                }
            }
        }

        double accuracy_individual = (static_cast<double>(correct_positive_individual + correct_negative_individual) /
                                     (val_positive_samples.getNumSamples() + val_negative_samples.getNumSamples())) * 100.0;

        std::cout << "Precisión del mejor modelo individual: " << accuracy_individual << "%\n";

        // Cargar los ntop mejores modelos para el ensemble
        std::vector<FullyConnectedLayer> ensemble_models;
        for (const auto& model_info : top_models) {
            FullyConnectedLayer model(input_size, output_size, optimizer);
            model.loadModel(model_info.second);
            ensemble_models.push_back(model);
            std::cout << "Modelo cargado desde: " << model_info.second << " con precisión: " << model_info.first << "%\n";
        }

        // Evaluación del ensemble mediante votación en el conjunto de validación
        std::cout << "\n--- Evaluando el ensemble mediante votación en el conjunto de validación ---\n";
        size_t correct_positive_ensemble_vote = 0;
        size_t correct_negative_ensemble_vote = 0;

        // Evaluación en muestras positivas (votación, paralelizado)
        {
            // #pragma omp parallel for reduction(+:correct_positive_ensemble_vote)
            for (size_t i = 0; i < val_positive_samples.getNumSamples(); ++i) {
                const Eigen::VectorXf& input = val_positive_samples.getSample(i);
                int votes = 0;

                for (auto& model : ensemble_models) {
                    Eigen::VectorXf output;
                    model.forward(input, output, false, true, threshold, activation, activation_derivative);
                    float goodness = output.squaredNorm();

                    if (goodness > threshold) {
                        votes++;
                    }
                }

                if (votes > static_cast<int>(ensemble_models.size() / 2)) {
                    correct_positive_ensemble_vote++;
                }
            }
        }

        // Evaluación en muestras negativas (votación, paralelizado)
        {
            // #pragma omp parallel for reduction(+:correct_negative_ensemble_vote)
            for (size_t i = 0; i < val_negative_samples.getNumSamples(); ++i) {
                const Eigen::VectorXf& input = val_negative_samples.getSample(i);
                int votes = 0;

                for (auto& model : ensemble_models) {
                    Eigen::VectorXf output;
                    model.forward(input, output, false, false, threshold, activation, activation_derivative);
                    float goodness = output.squaredNorm();

                    if (goodness < threshold) {
                        votes++;
                    }
                }

                if (votes > static_cast<int>(ensemble_models.size() / 2)) {
                    correct_negative_ensemble_vote++;
                }
            }
        }

        double accuracy_ensemble_vote = (static_cast<double>(correct_positive_ensemble_vote + correct_negative_ensemble_vote) /
                                        (val_positive_samples.getNumSamples() + val_negative_samples.getNumSamples())) * 100.0;
        std::cout << "Precisión del ensemble mediante votación: " << accuracy_ensemble_vote << "%\n";

        // Crear el modelo promedio (promediando pesos y biases)
        std::cout << "\n--- Creando el modelo promedio (promedio de pesos y biases) ---\n";
        FullyConnectedLayer averaged_model = ensemble_models.front(); // Inicializar con el primer modelo

        Eigen::MatrixXf accumulated_weights = averaged_model.getWeights();
        Eigen::VectorXf accumulated_biases = averaged_model.getBiases();

        for (size_t i = 1; i < ensemble_models.size(); ++i) {
            accumulated_weights += ensemble_models[i].getWeights();
            accumulated_biases += ensemble_models[i].getBiases();
        }

        accumulated_weights /= ensemble_models.size();
        accumulated_biases /= ensemble_models.size();

        averaged_model.setWeights(accumulated_weights);
        averaged_model.setBiases(accumulated_biases);

        // Evaluación del modelo promedio en el conjunto de validación
        std::cout << "\n--- Evaluando el modelo promedio en el conjunto de validación ---\n";
        size_t correct_positive_averaged = 0;
        size_t correct_negative_averaged = 0;

        // Evaluación positivas (promedio, paralelizado)
        {
            // #pragma omp parallel for reduction(+:correct_positive_averaged)
            for (size_t i = 0; i < val_positive_samples.getNumSamples(); ++i) {
                const Eigen::VectorXf& input = val_positive_samples.getSample(i);
                Eigen::VectorXf output;
                averaged_model.forward(input, output, false, true, threshold, activation, activation_derivative);

                float goodness = output.squaredNorm();

                if (goodness > threshold) {
                    correct_positive_averaged++;
                }
            }
        }

        // Evaluación negativas (promedio, paralelizado)
        {
            // #pragma omp parallel for reduction(+:correct_negative_averaged)
            for (size_t i = 0; i < val_negative_samples.getNumSamples(); ++i) {
                const Eigen::VectorXf& input = val_negative_samples.getSample(i);
                Eigen::VectorXf output;
                averaged_model.forward(input, output, false, false, threshold, activation, activation_derivative);

                float goodness = output.squaredNorm();

                if (goodness < threshold) {
                    correct_negative_averaged++;
                }
            }
        }

        double accuracy_averaged = (static_cast<double>(correct_positive_averaged + correct_negative_averaged) /
                                   (val_positive_samples.getNumSamples() + val_negative_samples.getNumSamples())) * 100.0;

        std::cout << "Precisión del modelo promedio: " << accuracy_averaged << "%\n";

        // Comparar las precisiones y decidir con cuál modelo continuar
        std::cout << "\n--- Comparación de precisiones ---\n";
        std::cout << "Precisión del mejor modelo individual: " << accuracy_individual << "%\n";
        std::cout << "Precisión del ensemble mediante votación: " << accuracy_ensemble_vote << "%\n";
        std::cout << "Precisión del modelo promedio: " << accuracy_averaged << "%\n";

        // Determinar el modelo con mejor precisión
        double max_accuracy = std::max({ accuracy_individual, accuracy_ensemble_vote, accuracy_averaged });

        FullyConnectedLayer layer(input_size, output_size, optimizer); // Modelo para el entrenamiento largo

        if (max_accuracy == accuracy_averaged) {
            std::cout << "\nEl modelo promedio tiene la mejor precisión.\n";
            layer = averaged_model;
            best_overall_threshold = threshold;
        } else if (max_accuracy == accuracy_ensemble_vote) {
            std::cout << "\nEl ensemble mediante votación tiene la mejor precisión.\n";
            // No podemos entrenar directamente un ensemble, así que usaremos el modelo promedio
            layer = averaged_model;
            best_overall_threshold = threshold;
        } else {
            std::cout << "\nEl mejor modelo individual tiene la mejor precisión.\n";
            layer = best_individual_model;
            best_overall_threshold = threshold;
        }

        // Continuar con el entrenamiento largo utilizando el modelo seleccionado
        std::cout << "\n--- Comenzando el entrenamiento largo ---\n";

        size_t epochs_without_improvement = 0; // Contador de épocas sin mejora
        double best_score = max_accuracy;
        FullyConnectedLayer best_overall_layer = layer;

        size_t val_positive_size = val_positive_samples.getNumSamples();
        size_t val_negative_size = val_negative_samples.getNumSamples();

        for (size_t epoch = 0; epoch < total_epochs_long; ++epoch) {
            std::cout << "\n--- Entrenamiento Largo - Época " << (epoch + 1) << "/" << total_epochs_long << " ---\n";

            // Mezclar y barajar los datos positivos y negativos
            train_positive_samples.shuffle();
            train_negative_samples.shuffle();

            // Crear una lista combinada de todas las muestras con sus etiquetas
            std::vector<std::pair<std::reference_wrapper<const Eigen::VectorXf>, bool>> combined_train_samples;
            combined_train_samples.reserve(train_positive_samples.getNumSamples() + train_negative_samples.getNumSamples());

            for (size_t i = 0; i < train_positive_samples.getNumSamples(); ++i) {
                combined_train_samples.emplace_back(train_positive_samples.getSample(i), true); // Positivo
            }
            for (size_t i = 0; i < train_negative_samples.getNumSamples(); ++i) {
                combined_train_samples.emplace_back(train_negative_samples.getSample(i), false); // Negativo
            }

            // Barajar la lista combinada
            {
                std::random_device rd;
                std::mt19937 g(rd());
                std::shuffle(combined_train_samples.begin(), combined_train_samples.end(), g);
            }

            // Entrenamiento largo con paralelización
            // #pragma omp parallel for
            for (size_t i = 0; i < combined_train_samples.size(); ++i) {
                const Eigen::VectorXf& input = combined_train_samples[i].first.get();
                bool is_positive = combined_train_samples[i].second;
                Eigen::VectorXf output;
                // #pragma omp critical
                {
                    layer.forward(input, output, true, is_positive, threshold, activation, activation_derivative);
                }
            }

            // Evaluación en conjunto de validación
            size_t correct_positive = 0;
            size_t correct_negative = 0;

            goodness_positive_vals.clear();
            goodness_negative_vals.clear();

            // Evaluación en muestras positivas - paralelizado
            {
                std::vector<float> local_goodness_positive;
                // #pragma omp parallel for reduction(+:correct_positive)
                for (size_t i = 0; i < val_positive_size; ++i) {
                    const Eigen::VectorXf& input = val_positive_samples.getSample(i);
                    Eigen::VectorXf output;
                    layer.forward(input, output, false, true, threshold, activation, activation_derivative);

                    float goodness = output.squaredNorm();
                    if (goodness > threshold) {
                        correct_positive++;
                    }

                    // #pragma omp critical
                    {
                        local_goodness_positive.push_back(goodness);
                    }
                }
                goodness_positive_vals = std::move(local_goodness_positive);
            }

            // Evaluación en muestras negativas - paralelizado
            {
                std::vector<float> local_goodness_negative;
                // #pragma omp parallel for reduction(+:correct_negative)
                for (size_t i = 0; i < val_negative_size; ++i) {
                    const Eigen::VectorXf& input = val_negative_samples.getSample(i);
                    Eigen::VectorXf output;
                    layer.forward(input, output, false, false, threshold, activation, activation_derivative);

                    float goodness = output.squaredNorm();
                    if (goodness < threshold) {
                        correct_negative++;
                    }

                    // #pragma omp critical
                    {
                        local_goodness_negative.push_back(goodness);
                    }
                }
                goodness_negative_vals = std::move(local_goodness_negative);
            }

            // Calcula la precisión
            double accuracy = (static_cast<double>(correct_positive + correct_negative) /
                              (val_positive_size + val_negative_size)) * 100.0;

            double accuracy_positive = (static_cast<double>(correct_positive) /
                                       val_positive_size) * 100.0;

            double accuracy_negative = (static_cast<double>(correct_negative) /
                                       val_negative_size) * 100.0;

            std::cout << "Precisión en validación: " << accuracy << "%\n"
                      << "Precisión en positivos: " << accuracy_positive << "%\n"
                      << "Precisión en negativos: " << accuracy_negative << "%\n";

            // Verifica si esta es la mejor precisión hasta ahora
            if (accuracy > best_score) {
                best_score = accuracy;
                epochs_without_improvement = 0;

                // Guarda el mejor modelo
                best_overall_layer = layer;
                best_overall_threshold = threshold; // Actualizamos el mejor umbral
                std::cout << "Nuevo mejor modelo general encontrado con precisión: " << best_score << "%\n";
            } else {
                epochs_without_improvement++;
            }

            // Revertir al mejor modelo si no hay mejora en 'patience' épocas
            if (epochs_without_improvement >= patience) {
                std::cout << "No hay mejora en las últimas " << patience << " épocas. Revirtiendo al mejor modelo general.\n";
                layer = best_overall_layer;
                threshold = best_overall_threshold; // Revertimos al mejor umbral
                epochs_without_improvement = 0; // Reinicia el contador
            }

            // Ajusta dinámicamente el umbral si está habilitado
            if (dynamic_threshold && !goodness_positive_vals.empty() && !goodness_negative_vals.empty()) {
                float avg_goodness_positive = std::accumulate(goodness_positive_vals.begin(),
                                                              goodness_positive_vals.end(), 0.0f) / goodness_positive_vals.size();
                float avg_goodness_negative = std::accumulate(goodness_negative_vals.begin(),
                                                              goodness_negative_vals.end(), 0.0f) / goodness_negative_vals.size();

                threshold = (avg_goodness_positive + avg_goodness_negative) / 2.0f;
                std::cout << "Umbral ajustado a: " << threshold << "\n";
            }

            // Guardar los histogramas combinados de esta época con el nombre único
            std::string hist_filename = "histograms/LongTraining_Epoch_" + std::to_string(epoch + 1) + ".png";

            plotGoodnessHistogramsCombined(goodness_positive_vals,
                                           goodness_negative_vals,
                                           threshold,
                                           hist_filename);

            std::cout << "Histograma combinado guardado en: " << hist_filename << "\n";
        }

        // Establecer la capa y umbral al mejor encontrado durante el entrenamiento largo
        FullyConnectedLayer final_layer = best_overall_layer;
        threshold = best_overall_threshold; // Usamos el mejor umbral encontrado
        double final_best_score = best_score;

        // Solicita al usuario la ruta para guardar el modelo final
        std::string model_path;
        std::cout << "\nIngrese la ruta para guardar el mejor modelo final (ej., best_model_final.bin): ";
        std::cin >> model_path;
        final_layer.saveModel(model_path);
        std::cout << "Mejor modelo final guardado en: " << model_path << "\n";

        // Construir el nombre de archivo para el histograma final
        std::string final_hist_filename = "histograms/Histograma_Combined_final.png";

        // Visualización de histogramas de bondad (última época)
        plotGoodnessHistogramsCombined(goodness_positive_vals,
                                       goodness_negative_vals,
                                       threshold,
                                       final_hist_filename);
        std::cout << "Histograma final combinado guardado en: " << final_hist_filename << "\n";

        // Visualización PCA
        int num_components;
        std::cout << "\nIngrese el número de componentes PCA (2 o 3): ";
        std::cin >> num_components;
        visualizePCA(final_layer, val_positive_samples, val_negative_samples, num_components, threshold);

    } catch (const std::exception& ex) {
        std::cerr << "Error durante el entrenamiento: " << ex.what() << "\n";
    }
}

// =====================================================
// Modo "Bayesiano" (realmente random) con CSV de salida
// =====================================================
/**
 * @brief Genera un flotante aleatorio en [min_val, max_val].
 */
float randomFloatInRange(float min_val, float max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    return dist(gen);
}

/**
 * @brief Pequeña función de entrenamiento simplificado para
 *        usar en la búsqueda “bayesiana”: sin histogramas ni PCA.
 * @param train_positive_samples, val_positive_samples, etc. Conjuntos de datos
 * @param optimizer Puntero al optimizador
 * @param layerSize Tamaño de la capa
 * @param threshold Valor de umbral inicial
 * @param dynamic_threshold Indica si se ajusta automáticamente
 * @param num_init, init_epochs, ntop, patience, total_epochs Largo
 * @return Precisión final en la validación
 */
double trainAndEvaluateBayes(Dataset& train_positive_samples,
                             Dataset& val_positive_samples,
                             Dataset& train_negative_samples,
                             Dataset& val_negative_samples,
                             std::shared_ptr<Optimizer> optimizer,
                             size_t layerSize,
                             float threshold,
                             bool dynamic_threshold,
                             size_t num_init,
                             size_t init_epochs,
                             size_t ntop,
                             size_t patience,
                             size_t total_epochs_long) {
    size_t input_size = train_positive_samples.getInputSize();

    // Lista para almacenar los mejores ntop modelos y sus puntuaciones
    std::vector<std::pair<double, std::string>> top_models; // <score, filepath>

    std::cout << "\n[Bayes] *** Nuevo Trial ***" 
              << "\n      layerSize=" << layerSize
              << " threshold=" << threshold
              << " dynamic_threshold=" << (dynamic_threshold ? "Yes":"No")
              << " #init=" << num_init 
              << " init_epochs=" << init_epochs
              << " total_long=" << total_epochs_long
              << "\n";

    double best_score_init_global = -std::numeric_limits<double>::infinity();

    // ---------------------------
    // 3) Múltiples inicializaciones cortas
    // ---------------------------
    for (size_t init_i = 0; init_i < num_init; ++init_i) {
        std::cout << "[Bayes] Inicialización " << (init_i + 1) << "/" << num_init << "\n";

        // Capa con pesos aleatorios
        FullyConnectedLayer current_layer(input_size, layerSize, optimizer);

        // Track del mejor de esta inicialización
        double best_score_local = -std::numeric_limits<double>::infinity();
        FullyConnectedLayer best_local_layer = current_layer;
        float best_local_threshold = threshold;
        size_t epochs_no_improve_local = 0;

        // --- Entrenamiento corto con 'init_epochs' ---
        for (size_t epoch = 0; epoch < init_epochs; ++epoch) {
            std::cout << "   [Init=" << (init_i+1) 
                      << "] Época " << (epoch+1) << "/" << init_epochs << "\n";

            // Barajar
            train_positive_samples.shuffle();
            train_negative_samples.shuffle();

            // Combinar datos
            std::vector<std::pair<std::reference_wrapper<const Eigen::VectorXf>, bool>> combined;
            combined.reserve(train_positive_samples.getNumSamples() + train_negative_samples.getNumSamples());
            for (size_t i = 0; i < train_positive_samples.getNumSamples(); ++i) {
                combined.emplace_back(train_positive_samples.getSample(i), true);
            }
            for (size_t i = 0; i < train_negative_samples.getNumSamples(); ++i) {
                combined.emplace_back(train_negative_samples.getSample(i), false);
            }

            // Re-barajamos la lista combinada
            {
                std::random_device rd;
                std::mt19937 g(rd());
                std::shuffle(combined.begin(), combined.end(), g);
            }

            // Forward + training
            for (size_t i = 0; i < combined.size(); ++i) {
                bool is_pos = combined[i].second;
                const Eigen::VectorXf& input_data = combined[i].first.get();
                Eigen::VectorXf out;
                current_layer.forward(input_data, out, true, is_pos, threshold, activation, activation_derivative);
            }

            // Evaluación en validación
            size_t correct_pos = 0;
            size_t correct_neg = 0;
            std::vector<float> goodness_positive_vals;
            std::vector<float> goodness_negative_vals;

            // Positivos
            for (size_t i = 0; i < val_positive_samples.getNumSamples(); ++i) {
                Eigen::VectorXf out;
                current_layer.forward(val_positive_samples.getSample(i), out, false, true, threshold, activation, activation_derivative);
                float good = out.squaredNorm();
                if (good > threshold) {
                    correct_pos++;
                }
                goodness_positive_vals.push_back(good);
            }

            // Negativos
            for (size_t i = 0; i < val_negative_samples.getNumSamples(); ++i) {
                Eigen::VectorXf out;
                current_layer.forward(val_negative_samples.getSample(i), out, false, false, threshold, activation, activation_derivative);
                float good = out.squaredNorm();
                if (good < threshold) {
                    correct_neg++;
                }
                goodness_negative_vals.push_back(good);
            }

            // Cálculo de accuracy
            double accuracy = 100.0 * (double)(correct_pos + correct_neg)
                              / (val_positive_samples.getNumSamples() + val_negative_samples.getNumSamples());

            std::cout << "      Accuracy val = " << accuracy << "%\n";

            // Early stopping local
            if (accuracy > best_score_local) {
                best_score_local = accuracy;
                epochs_no_improve_local = 0;
                best_local_layer = current_layer;
                best_local_threshold = threshold;
            } else {
                epochs_no_improve_local++;
            }

            if (epochs_no_improve_local >= patience) {
                std::cout << "   [Init=" << (init_i+1) 
                          << "] No mejora en " << patience << " épocas. Revierto.\n";
                current_layer = best_local_layer;
                threshold = best_local_threshold;
                epochs_no_improve_local = 0;
                break;
            }

            // Ajuste dinámico de umbral si corresponde
            if (dynamic_threshold &&
                !goodness_positive_vals.empty() && !goodness_negative_vals.empty()) {
                float avg_pos = std::accumulate(goodness_positive_vals.begin(), goodness_positive_vals.end(), 0.0f)
                                / goodness_positive_vals.size();
                float avg_neg = std::accumulate(goodness_negative_vals.begin(), goodness_negative_vals.end(), 0.0f)
                                / goodness_negative_vals.size();
                threshold = (avg_pos + avg_neg) * 0.5f;
            }
        } // Fin for epoch

        // Al terminar las init_epochs, guardar el mejor local en top_models
        // Guardar el modelo en un archivo temporal
        std::string model_filename = "ntop_cache/bayes_trial_layerSize_" + std::to_string(layerSize) + "_init_" + std::to_string(init_i+1) + ".bin";
        best_local_layer.saveModel(model_filename);

        // Agregar el modelo y su puntuación a top_models
        top_models.emplace_back(best_score_local, model_filename);

        // Ordenar top_models y mantener solo ntop mejores
        std::sort(top_models.begin(), top_models.end(),
                  [](const std::pair<double, std::string>& a, const std::pair<double, std::string>& b) {
                      return a.first > b.first; // Orden descendente
                  });

        if (top_models.size() > ntop) {
            // Eliminar modelos sobrantes
            for (size_t i = ntop; i < top_models.size(); ++i) {
                fs::remove(top_models[i].second); // Eliminar archivo del modelo
            }
            top_models.resize(ntop);
        }

        // Mantener el mejor score global dentro del trial
        if (top_models[0].first > best_score_init_global) {
            best_score_init_global = top_models[0].first;
        }
    } // Fin for init_i

    // -------------------------------------
    // 4) Tomar el mejor de las inicializaciones y 
    //    (Opcional) Entrenamiento largo
    // -------------------------------------
    // El top_models[0] es el de mayor score
    double best_score_local = top_models[0].first;
    std::string best_model_path = top_models[0].second;

    // Cargar el mejor modelo
    FullyConnectedLayer best_layer(input_size, layerSize, optimizer);
    best_layer.loadModel(best_model_path);

    // Entrenamiento largo si es necesario
    if (total_epochs_long > 0) {
        size_t epochs_no_improve = 0;
        double best_score_ever = best_score_local;
        FullyConnectedLayer best_saved_layer = best_layer;
        float best_saved_threshold = threshold;

        for (size_t epoch = 0; epoch < total_epochs_long; ++epoch) {
            std::cout << "[Bayes] Entrenamiento Largo - Época " 
                      << (epoch+1) << "/" << total_epochs_long << "\n";

            // Mezclar
            train_positive_samples.shuffle();
            train_negative_samples.shuffle();

            // Combinar datos
            std::vector<std::pair<std::reference_wrapper<const Eigen::VectorXf>, bool>> combined;
            combined.reserve(train_positive_samples.getNumSamples() + train_negative_samples.getNumSamples());
            for (size_t i = 0; i < train_positive_samples.getNumSamples(); ++i) {
                combined.emplace_back(train_positive_samples.getSample(i), true);
            }
            for (size_t i = 0; i < train_negative_samples.getNumSamples(); ++i) {
                combined.emplace_back(train_negative_samples.getSample(i), false);
            }
            {
                std::random_device rd;
                std::mt19937 g(rd());
                std::shuffle(combined.begin(), combined.end(), g);
            }

            // Forward + training
            for (size_t i = 0; i < combined.size(); ++i) {
                bool is_pos = combined[i].second;
                const Eigen::VectorXf& input_data = combined[i].first.get();
                Eigen::VectorXf out;
                best_layer.forward(input_data, out, true, is_pos, threshold, activation, activation_derivative);
            }

            // Evaluación en validación
            size_t correct_pos = 0;
            size_t correct_neg = 0;
            std::vector<float> goodness_positive_vals;
            std::vector<float> goodness_negative_vals;

            // Positivos
            for (size_t i = 0; i < val_positive_samples.getNumSamples(); ++i) {
                Eigen::VectorXf out;
                best_layer.forward(val_positive_samples.getSample(i), out, false, true, threshold, activation, activation_derivative);
                float good = out.squaredNorm();
                if (good > threshold) {
                    correct_pos++;
                }
                goodness_positive_vals.push_back(good);
            }

            // Negativos
            for (size_t i = 0; i < val_negative_samples.getNumSamples(); ++i) {
                Eigen::VectorXf out;
                best_layer.forward(val_negative_samples.getSample(i), out, false, false, threshold, activation, activation_derivative);
                float good = out.squaredNorm();
                if (good < threshold) {
                    correct_neg++;
                }
                goodness_negative_vals.push_back(good);
            }

            // Cálculo de accuracy
            double accuracy = 100.0 * (double)(correct_pos + correct_neg)
                              / (val_positive_samples.getNumSamples() + val_negative_samples.getNumSamples());

            std::cout << "   [Bayes] Acc. val: " << accuracy << "%\n";

            if (accuracy > best_score_ever) {
                best_score_ever = accuracy;
                epochs_no_improve = 0;
                best_saved_layer = best_layer;
                best_saved_threshold = threshold;
                std::cout << "   [Bayes] Nuevo mejor local: " << best_score_ever << "%\n";
            } else {
                epochs_no_improve++;
            }

            if (epochs_no_improve >= patience) {
                std::cout << "   [Bayes] No hay mejora en " << patience 
                          << " épocas. Revierto al mejor.\n";
                best_layer = best_saved_layer;
                threshold = best_saved_threshold;
                epochs_no_improve = 0;
                break;
            }

            // Ajuste dinámico de umbral si corresponde
            if (dynamic_threshold &&
                !goodness_positive_vals.empty() && !goodness_negative_vals.empty()) {
                float avg_pos = std::accumulate(goodness_positive_vals.begin(), goodness_positive_vals.end(), 0.0f)
                                / goodness_positive_vals.size();
                float avg_neg = std::accumulate(goodness_negative_vals.begin(), goodness_negative_vals.end(), 0.0f)
                                / goodness_negative_vals.size();
                threshold = (avg_pos + avg_neg) * 0.5f;
            }
        } // Fin for epoch

        // Al acabar, best_score_ever es la precisión final
        best_score_local = best_score_ever;
        best_layer = best_saved_layer;
    } // Fin Entrenamiento largo

    // 5) Retornar la precisión final en validación
    std::cout << "[Bayes] *** Final accuracy del TRIAL = " << best_score_local << "% ***\n";
    return best_score_local;
}

/**
 * @brief Función que simula la búsqueda bayesiana de hiperparámetros
 *        (realmente: random search) y guarda resultados en CSV.
 */
/**
 * @brief Función que simula la búsqueda bayesiana de hiperparámetros
 *        (realmente: random search) y guarda resultados en CSV.
 */
void runBayesianOptimization() {
    try {
        // Parámetros globales
        size_t total_epochs_long;
        std::cout << "Ingrese # total de epochs (entrenamiento largo): ";
        std::cin >> total_epochs_long;

        bool dynamic_threshold = false;
        std::cout << "¿Desea umbral dinámico? (1=Sí,0=No): ";
        int tmp;
        std::cin >> tmp;
        dynamic_threshold = (tmp == 1);

        size_t num_init;
        std::cout << "Ingrese # de inicializaciones: ";
        std::cin >> num_init;

        size_t init_epochs;
        std::cout << "Ingrese # de epochs por inicialización: ";
        std::cin >> init_epochs;

        size_t ntop;
        std::cout << "Ingrese # de mejores modelos a recordar (ntop): ";
        std::cin >> ntop;

        size_t patience;
        std::cout << "Ingrese # de épocas de tolerancia (patience): ";
        std::cin >> patience;

        // Umbral base
        float base_threshold;
        std::cout << "Ingrese el umbral base: ";
        std::cin >> base_threshold;

        // Rango learning rate
        float min_lr, max_lr;
        std::cout << "Rango min de LR: ";
        std::cin >> min_lr;
        std::cout << "Rango max de LR: ";
        std::cin >> max_lr;

        // Rango momentum (para SGD)
        float min_mom, max_mom;
        std::cout << "Rango min de momentum: ";
        std::cin >> min_mom;
        std::cout << "Rango max de momentum: ";
        std::cin >> max_mom;

        // Rango alpha (para LowPassFilter)
        float min_alpha, max_alpha;
        std::cout << "Rango min de alpha: ";
        std::cin >> min_alpha;
        std::cout << "Rango max de alpha: ";
        std::cin >> max_alpha;

        int num_trials;
        std::cout << "Ingrese el # de trials por cada tamaño de capa [16,32,64,128]: ";
        std::cin >> num_trials;

        // Cargar datasets
        Dataset train_positive_samples("data/positive_images/train/");
        Dataset val_positive_samples("data/positive_images/val/");
        Dataset train_negative_samples("data/negative_images/train/");
        Dataset val_negative_samples("data/negative_images/val/");

        if (train_positive_samples.getNumSamples() == 0 ||
            val_positive_samples.getNumSamples() == 0 ||
            train_negative_samples.getNumSamples() == 0 ||
            val_negative_samples.getNumSamples() == 0) {
            throw std::runtime_error("Algún conjunto de datos (train o val) está vacío.");
        }

        fs::create_directories("results");
        std::string csv_path = "results/bayes_results.csv";
        std::ofstream csvFile(csv_path, std::ios::out);
        if (!csvFile.is_open()) {
            throw std::runtime_error("No se pudo crear " + csv_path);
        }
        // Escribir encabezados con valores predeterminados para parámetros irrelevantes
        csvFile << "Optimizer,LayerSize,LR,Momentum,Alpha,BaseThreshold,DynamicThreshold,Accuracy\n";

        // Lista de tamaños de capa
        std::vector<size_t> layerSizes = {16, 32, 64};

        // Lista de optimizadores a intercalar
        std::vector<std::string> optimizer_names = {"SGD", "LowPassFilter"};
        size_t optimizer_count = optimizer_names.size();

        // Iterar sobre cada tamaño de capa
        for (auto layerSize : layerSizes) {
            // Ajustar umbral en función del tamaño de capa
            float scaled_threshold = base_threshold;
            if (layerSize == 32) {
                scaled_threshold = base_threshold * 2.0f; 
            } else if (layerSize == 64) {
                scaled_threshold = base_threshold * 4.0f; 
            } else if (layerSize == 128) {
                scaled_threshold = base_threshold * 8.0f; 
            }

            // Para cada trial
            for (int t = 0; t < num_trials; ++t) {
                std::cout << "\n=== Trial " << (t+1) << " para LayerSize=" << layerSize << " ===\n";

                // Seleccionar optimizador intercalado
                std::string optName = optimizer_names[t % optimizer_count];
                std::shared_ptr<Optimizer> trial_optimizer;
                float trial_lr = randomFloatInRange(min_lr, max_lr);
                float trial_mom = 0.0f;   // Valor predeterminado para optimizadores que no lo usan
                float trial_alpha = 0.0f; // Valor predeterminado para optimizadores que no lo usan

                if (optName == "SGD") {
                    trial_mom = randomFloatInRange(min_mom, max_mom);
                    trial_optimizer = std::make_shared<SGDOptimizer>(trial_lr, trial_mom);
                }
                else if (optName == "Adam") {
                    // Para simplificar, se asignan valores predeterminados a parámetros de Adam
                    float beta1 = 0.9f;
                    float beta2 = 0.999f;
                    float epsilon = 1e-8f;
                    trial_optimizer = std::make_shared<AdamOptimizer>(trial_lr, beta1, beta2, epsilon);
                }
                else if (optName == "LowPassFilter") {
                    trial_alpha = randomFloatInRange(min_alpha, max_alpha);
                    trial_optimizer = std::make_shared<LowPassFilterOptimizer>(trial_lr, trial_alpha);
                }
                else if (optName == "AdaBelief") {
                    // Para simplificar, se asignan valores predeterminados a parámetros de AdaBelief
                    float beta1 = 0.9f;
                    float beta2 = 0.999f;
                    float epsilon = 1e-8f;
                    trial_optimizer = std::make_shared<AdaBeliefOptimizer>(trial_lr, beta1, beta2, epsilon);
                }
                else {
                    throw std::runtime_error("Optimizador desconocido: " + optName);
                }

                std::cout << "[Trial " << (t+1) << "] Optimizer: " << optName
                          << ", LR: " << trial_lr;

                if (optName == "SGD") {
                    std::cout << ", Momentum: " << trial_mom;
                }
                else if (optName == "LowPassFilter") {
                    std::cout << ", Alpha: " << trial_alpha;
                }

                std::cout << "\n";

                // Entrenar y evaluar el modelo
                double accuracy = trainAndEvaluateBayes(
                    train_positive_samples,
                    val_positive_samples,
                    train_negative_samples,
                    val_negative_samples,
                    trial_optimizer,
                    layerSize,
                    scaled_threshold,
                    dynamic_threshold,
                    num_init,
                    init_epochs,
                    ntop,
                    patience,
                    total_epochs_long
                );

                // Guardar en CSV con manejo de parámetros irrelevantes
                csvFile << optName << ","
                        << layerSize << ","
                        << trial_lr << ",";

                if (optName == "SGD") {
                    csvFile << trial_mom << "," << "N/A" << ",";
                }
                else if (optName == "LowPassFilter") {
                    csvFile << "N/A" << "," << trial_alpha << ",";
                }
                else { // Adam y AdaBelief
                    csvFile << "N/A" << "," << "N/A" << ",";
                }

                csvFile << base_threshold << ","
                        << (dynamic_threshold ? 1 : 0) << ","
                        << accuracy << "\n";

                std::cout << "[Trial " << (t+1) << "] Accuracy: " << accuracy << "%\n";
            }
        }
        csvFile.close();
        std::cout << "Resultados guardados en: " << csv_path << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error durante la optimización bayesiana: " << ex.what() << "\n";
    }
}



// =================================
// Función principal del programa
// =================================
int main() {
    // Configurar número de hilos
    Eigen::setNbThreads(std::thread::hardware_concurrency());

    std::cout << "Seleccione el modo:\n";
    std::cout << "1) Entrenamiento normal\n";
    std::cout << "2) Busqueda Bayesiana Hiperparametros (simulada)\n";
    int mode;
    std::cin >> mode;

    if (mode == 1) {
        trainModel();
    } else if (mode == 2) {
        runBayesianOptimization();
    } else {
        std::cout << "Opción inválida.\n";
    }

    return 0;
}
