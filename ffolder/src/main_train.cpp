// src/main_train.cpp

#include "neural_network.hpp"
#include "optimizer.hpp"
#include "scatter_plot_data.hpp"

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

namespace fs = std::filesystem;

// ================================
// Funciones Auxiliares y Principal
// ================================

/**
 * @brief Divide el conjunto de datos en entrenamiento y validación.
 * @param dataset Conjunto de datos completo.
 * @param train_fraction Fracción de datos para entrenamiento.
 * @param train_set Referencia al conjunto de entrenamiento.
 * @param val_set Referencia al conjunto de validación.
 */
void splitDataset(const Dataset& dataset, float train_fraction,
                  Dataset& train_set, Dataset& val_set) {
    size_t total_samples = dataset.getNumSamples();
    size_t train_samples = static_cast<size_t>(total_samples * train_fraction);

    std::vector<size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

    for (size_t i = 0; i < train_samples; ++i) {
        train_set.addSample(dataset.getSample(indices[i]),
                            dataset.getImagePath(indices[i]));
    }
    for (size_t i = train_samples; i < total_samples; ++i) {
        val_set.addSample(dataset.getSample(indices[i]),
                          dataset.getImagePath(indices[i]));
    }

    // Copia las propiedades de imagen
    train_set.setImageProperties(dataset.getImageHeight(),
                                 dataset.getImageWidth(),
                                 dataset.getNumChannels());
    val_set.setImageProperties(dataset.getImageHeight(),
                               dataset.getImageWidth(),
                               dataset.getNumChannels());
}

/**
 * @brief Permite al usuario seleccionar un optimizador.
 * @return Puntero compartido al optimizador seleccionado.
 */
std::shared_ptr<Optimizer> selectOptimizer() {
    int optimizer_choice;
    std::cout << "\nSeleccione el optimizador:\n";
    std::cout << "1. Adam Optimizer\n";
    std::cout << "Ingrese su elección: ";
    std::cin >> optimizer_choice;

    if (optimizer_choice == 1) {
        float learning_rate;
        std::cout << "Ingrese el learning rate para Adam Optimizer: ";
        std::cin >> learning_rate;
        return std::make_shared<AdamOptimizer>(learning_rate);
    } else {
        std::cout << "Opción inválida. Usando Adam Optimizer con learning rate por defecto.\n";
        return std::make_shared<AdamOptimizer>();
    }
}

// ================================
// Funciones para Visualización
// ================================

void visualizePCA(FullyConnectedLayer& layer, Dataset& val_positive_samples,
                  Dataset& val_negative_samples, int num_components,
                  float threshold) { // Añadido 'threshold'
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

    // Definir la función de activación y su derivada (identidad en este caso)
    auto activation = [](float x) -> float {
        return x;
    };
    auto activation_derivative = [](float x) -> float {
        return 1.0f;
    };

    // Recopilar salidas para muestras positivas
    size_t idx = 0;
    for (size_t i = 0; i < val_positive_size; ++i, ++idx) {
        const Eigen::VectorXf& input = val_positive_samples.getSample(i);
        Eigen::VectorXf output;
        layer.forward(input, output, false, true, threshold, // Usar 'threshold'
                     activation, activation_derivative); // Activación identidad

        // Copiar la salida a la matriz de datos
        for (size_t j = 0; j < output_size; ++j) {
            data.at<float>(idx, j) = output[j];
        }
        labels[idx] = 1; // Positivo
        image_paths[idx] = val_positive_samples.getImagePath(i);
    }

    // Recopilar salidas para muestras negativas
    for (size_t i = 0; i < val_negative_size; ++i, ++idx) {
        const Eigen::VectorXf& input = val_negative_samples.getSample(i);
        Eigen::VectorXf output;
        layer.forward(input, output, false, false, threshold, // Usar 'threshold'
                     activation, activation_derivative); // Activación identidad

        // Copiar la salida a la matriz de datos
        for (size_t j = 0; j < output_size; ++j) {
            data.at<float>(idx, j) = output[j];
        }
        labels[idx] = 0; // Negativo
        image_paths[idx] = val_negative_samples.getImagePath(i);
    }

    // Realizar PCA
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);

    // Proyectar datos
    cv::Mat projected_data = pca.project(data);

    // Encontrar mínimos y máximos para el escalado
    cv::Mat min_vals, max_vals;
    cv::reduce(projected_data, min_vals, 0, cv::REDUCE_MIN);
    cv::reduce(projected_data, max_vals, 0, cv::REDUCE_MAX);

    // Crear imagen para el scatter plot
    int img_size = 800;
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

        cv::Scalar color = labels[i] == 1 ? cv::Scalar(255, 0, 0) :
                                            cv::Scalar(0, 255, 0);
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

// ================================
// Funciones de Entrenamiento y Evaluación
// ================================

void trainAndEvaluate(Dataset& train_positive_samples,
                      Dataset& train_negative_samples,
                      Dataset& val_positive_samples,
                      Dataset& val_negative_samples,
                      FullyConnectedLayer& layer, float& threshold,
                      size_t epochs,
                      const std::function<float(float)>& activation,
                      const std::function<float(float)>& activation_derivative,
                      bool verbose,
                      double& best_score,
                      bool dynamic_threshold,
                      std::vector<float>& goodness_positive_vals,
                      std::vector<float>& goodness_negative_vals,
                      size_t patience, // Número de épocas para esperar antes de revertir
                      FullyConnectedLayer& best_layer, // Capa para almacenar el mejor modelo
                      float& best_threshold) {
    size_t train_positive_size = train_positive_samples.getNumSamples();
    size_t train_negative_size = train_negative_samples.getNumSamples();
    size_t val_positive_size = val_positive_samples.getNumSamples();
    size_t val_negative_size = val_negative_samples.getNumSamples();

    size_t epochs_without_improvement = 0; // Contador de épocas sin mejora

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        if (verbose) {
            std::cout << "\n--- Época " << (epoch + 1) << "/" << epochs << " ---\n";
        }

        train_positive_samples.shuffle();
        train_negative_samples.shuffle();

        // Entrenamiento en muestras positivas
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < train_positive_size; ++i) {
            const Eigen::VectorXf& input = train_positive_samples.getSample(i);
            Eigen::VectorXf output;
            layer.forward(input, output, true, true, threshold, activation, activation_derivative);
        }

        // Entrenamiento en muestras negativas
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < train_negative_size; ++i) {
            const Eigen::VectorXf& input = train_negative_samples.getSample(i);
            Eigen::VectorXf output;
            layer.forward(input, output, true, false, threshold, activation, activation_derivative);
        }

        // Evaluación en conjunto de validación
        size_t correct_positive = 0;
        size_t correct_negative = 0;

        goodness_positive_vals.clear();
        goodness_negative_vals.clear();

        // Evaluación en muestras positivas
        #pragma omp parallel for reduction(+:correct_positive)
        for (size_t i = 0; i < val_positive_size; ++i) {
            const Eigen::VectorXf& input = val_positive_samples.getSample(i);
            Eigen::VectorXf output;
            layer.forward(input, output, false, true, threshold, activation, activation_derivative);

            float goodness = output.squaredNorm();
            #pragma omp critical
            {
                goodness_positive_vals.push_back(goodness);
            }

            if (goodness > threshold) {
                ++correct_positive;
            }
        }

        // Evaluación en muestras negativas
        #pragma omp parallel for reduction(+:correct_negative)
        for (size_t i = 0; i < val_negative_size; ++i) {
            const Eigen::VectorXf& input = val_negative_samples.getSample(i);
            Eigen::VectorXf output;
            layer.forward(input, output, false, false, threshold, activation, activation_derivative);

            float goodness = output.squaredNorm();
            #pragma omp critical
            {
                goodness_negative_vals.push_back(goodness);
            }

            if (goodness < threshold) {
                ++correct_negative;
            }
        }

        // Calcula la precisión
        double accuracy = (static_cast<double>(correct_positive + correct_negative) /
                          (val_positive_size + val_negative_size)) * 100.0;

        if (verbose) {
            std::cout << "Precisión en validación: " << accuracy << "%\n";
        }

        // Verifica si esta es la mejor precisión hasta ahora
        if (accuracy > best_score) {
            best_score = accuracy;
            epochs_without_improvement = 0;

            // Guarda el mejor modelo
            best_layer = layer; // Asumiendo que la clase FullyConnectedLayer tiene un operador de asignación
            best_threshold = threshold;
        } else {
            epochs_without_improvement++;
        }

        // Revertir al mejor modelo si no hay mejora en 'patience' épocas
        if (epochs_without_improvement >= patience) {
            if (verbose) {
                std::cout << "No hay mejora en las últimas " << patience << " épocas. Revirtiendo al mejor modelo.\n";
            }
            layer = best_layer;
            threshold = best_threshold;
            epochs_without_improvement = 0; // Reinicia el contador
        }

        // Ajusta dinámicamente el umbral si está habilitado
        if (dynamic_threshold) {
            float avg_goodness_positive = std::accumulate(goodness_positive_vals.begin(),
                                                          goodness_positive_vals.end(), 0.0f) / val_positive_size;
            float avg_goodness_negative = std::accumulate(goodness_negative_vals.begin(),
                                                          goodness_negative_vals.end(), 0.0f) / val_negative_size;

            threshold = (avg_goodness_positive + avg_goodness_negative) / 2.0f;
            if (verbose) {
                std::cout << "Umbral ajustado a: " << threshold << "\n";
            }
        }

        // Construir el nombre de archivo único para la época actual
        std::string hist_filename = "histograms/Histograma_Combined_epoch_" + std::to_string(epoch + 1) + ".png";

        // Guardar los histogramas combinados de esta época con el nombre único
        plotGoodnessHistogramsCombined(goodness_positive_vals,
                                       goodness_negative_vals,
                                       threshold,
                                       hist_filename); // Pasar 'hist_filename' como ruta de guardado

        if (verbose) {
            std::cout << "Histograma combinado guardado en: " << hist_filename << "\n";
        }
    }
}

/**
 * @brief Función principal que maneja el flujo de entrenamiento del modelo.
 */
void trainModel() {
    try {
        std::cout << "Cargando conjuntos de datos...\n";
        Dataset positive_samples("data/positive_images/"); // Directorio de imágenes positivas
        Dataset negative_samples("data/negative_images/"); // Directorio de imágenes negativas

        if (positive_samples.getNumSamples() == 0 ||
            negative_samples.getNumSamples() == 0) {
            throw std::runtime_error("Conjuntos de datos positivos o negativos están vacíos.");
        }

        Dataset train_positive_samples, val_positive_samples;
        Dataset train_negative_samples, val_negative_samples;

        // Divide los conjuntos de datos en entrenamiento (80%) y validación (20%)
        splitDataset(positive_samples, 0.8f, train_positive_samples,
                     val_positive_samples);
        splitDataset(negative_samples, 0.8f, train_negative_samples,
                     val_negative_samples);

        // Selecciona el optimizador
        std::shared_ptr<Optimizer> optimizer = selectOptimizer();

        // Pregunta al usuario si desea utilizar un umbral dinámico
        bool dynamic_threshold = false;
        std::cout << "¿Desea utilizar un umbral dinámico? (1 = Sí, 0 = No): ";
        int threshold_choice;
        std::cin >> threshold_choice;
        dynamic_threshold = (threshold_choice == 1);

        // Solicita al usuario el umbral inicial
        float threshold;
        std::cout << "Ingrese el umbral inicial para determinar la bondad: ";
        std::cin >> threshold;

        // Solicita al usuario el número de épocas de entrenamiento
        size_t epochs;
        std::cout << "Ingrese el número de épocas de entrenamiento: ";
        std::cin >> epochs;

        // Define la función de activación y su derivada (Leaky ReLU)
        auto activation = [](float x) -> float {
            return x > 0.0f ? x : 0.01f * x; // Leaky ReLU
        };

        auto activation_derivative = [](float x) -> float {
            return x > 0.0f ? 1.0f : 0.01f;
        };

        // Vectores para almacenar las bondades durante la evaluación
        std::vector<float> goodness_positive_vals;
        std::vector<float> goodness_negative_vals;

        // Solicita al usuario el tamaño de la capa completamente conectada
        size_t input_size = train_positive_samples.getInputSize();
        size_t output_size;
        std::cout << "Ingrese el tamaño de la capa (número de neuronas): ";
        std::cin >> output_size;

        // Inicializa la capa completamente conectada
        FullyConnectedLayer layer(input_size, output_size, optimizer);

        // Inicializa variables para el seguimiento del mejor modelo
        double best_score = -std::numeric_limits<double>::infinity();
        FullyConnectedLayer best_layer = layer; // Copia inicial del modelo
        float best_threshold = threshold;

        // Configuración de la paciencia (tolerancia)
        size_t patience;
        std::cout << "Ingrese el número de épocas de tolerancia sin mejora (patience): ";
        std::cin >> patience;

        // Asegurarse de que la carpeta principal para los histogramas exista
        fs::create_directories("histograms");

        // Entrena y evalúa la capa
        trainAndEvaluate(train_positive_samples, train_negative_samples,
                         val_positive_samples, val_negative_samples,
                         layer, threshold, epochs,
                         activation, activation_derivative,
                         true, best_score, dynamic_threshold,
                         goodness_positive_vals, goodness_negative_vals,
                         patience, best_layer, best_threshold);

        // Restaurar el mejor modelo después del entrenamiento
        layer = best_layer;
        threshold = best_threshold;

        // Solicita al usuario la ruta para guardar el modelo
        std::string model_path;
        std::cout << "\nIngrese la ruta para guardar el mejor modelo (ej., best_model.bin): ";
        std::cin >> model_path;
        layer.saveModel(model_path);
        std::cout << "Mejor modelo guardado en: " << model_path << "\n";

        // Construir el nombre de archivo para el histograma final
        std::string final_hist_filename = "histograms/Histograma_Combined_final.png";

        // Visualización de histogramas de bondad (última época)
        plotGoodnessHistogramsCombined(goodness_positive_vals,
                                       goodness_negative_vals,
                                       threshold,
                                       final_hist_filename); // Guardar el histograma final
        std::cout << "Histograma final combinado guardado en: " << final_hist_filename << "\n";

        // Visualización PCA
        int num_components;
        std::cout << "\nIngrese el número de componentes PCA (2 o 3): ";
        std::cin >> num_components;
        visualizePCA(layer, val_positive_samples, val_negative_samples, num_components, threshold); // Pasar 'threshold'

    } catch (const std::exception& ex) {
        std::cerr << "Error durante el entrenamiento: " << ex.what() << "\n";
    }
}

// Función principal del programa
int main() {
    trainModel();
    return 0;
}
