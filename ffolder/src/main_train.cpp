//src/main_train.cpp

#include "neural_network.hpp"
#include "optimizer.hpp"
#include "scatter_plot_data.hpp"
#include <Eigen/Dense>
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
 * @brief Divide el conjunto de datos en entrenamiento y validación, aplicando una permutación fija.
 * @param dataset Conjunto de datos completo.
 * @param train_fraction Fracción de datos para entrenamiento.
 * @param train_set Referencia al conjunto de entrenamiento.
 * @param val_set Referencia al conjunto de validación.
 * @param permutation Vector de permutación a aplicar.
 */
void splitDataset(const Dataset& dataset, float train_fraction,
                 Dataset& train_set, Dataset& val_set,
                 const std::vector<size_t>& permutation) {
    size_t total_samples = dataset.getNumSamples();
    size_t train_samples = static_cast<size_t>(total_samples * train_fraction);

    std::vector<size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

    for (size_t i = 0; i < train_samples; ++i) {
        Eigen::VectorXf permuted_sample = applyPermutation(dataset.getSample(indices[i]), permutation);
        train_set.addSample(permuted_sample, dataset.getImagePath(indices[i]));
    }
    for (size_t i = train_samples; i < total_samples; ++i) {
        Eigen::VectorXf permuted_sample = applyPermutation(dataset.getSample(indices[i]), permutation);
        val_set.addSample(permuted_sample, dataset.getImagePath(indices[i]));
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
    std::cout << "1. SGD Optimizer\n";
    std::cout << "2. Adam Optimizer\n";
    std::cout << "3. RMSProp Optimizer\n";
    std::cout << "4. Adagrad Optimizer\n";
    std::cout << "5. Adadelta Optimizer\n";
    std::cout << "6. AdamW Optimizer\n";
    std::cout << "7. Low Pass Filter Optimizer\n";
    std::cout << "Ingrese su elección: ";
    std::cin >> optimizer_choice;

    switch (optimizer_choice) {
        case 1: {
            float learning_rate, momentum;

            std::cout << "Ingrese el learning rate para SGD Optimizer (ej. 0.0001): ";
            std::cin >> learning_rate;

            std::cout << "Ingrese el valor de momentum (ej. 0 para sin momentum): ";
            std::cin >> momentum;

            return std::make_shared<SGDOptimizer>(learning_rate, momentum);
        }
        case 2: {
            float learning_rate, beta1, beta2, epsilon;

            std::cout << "Ingrese el learning rate para Adam Optimizer (ej. 0.00001): ";
            std::cin >> learning_rate;

            std::cout << "Ingrese el valor de beta1 (ej. 0.9): ";
            std::cin >> beta1;

            std::cout << "Ingrese el valor de beta2 (ej. 0.999): ";
            std::cin >> beta2;

            std::cout << "Ingrese el valor de epsilon (ej. 1e-8): ";
            std::cin >> epsilon;

            return std::make_shared<AdamOptimizer>(learning_rate, beta1, beta2, epsilon);
        }
        case 3: {
            float learning_rate, beta, epsilon;

            std::cout << "Ingrese el learning rate para RMSProp Optimizer (ej. 0.00001): ";
            std::cin >> learning_rate;

            std::cout << "Ingrese el valor de beta (ej. 0.9): ";
            std::cin >> beta;

            std::cout << "Ingrese el valor de epsilon (ej. 1e-8): ";
            std::cin >> epsilon;

            return std::make_shared<RMSPropOptimizer>(learning_rate, beta, epsilon);
        }
        case 4: {
            float learning_rate, epsilon;

            std::cout << "Ingrese el learning rate para Adagrad Optimizer (ej. 0.0001): ";
            std::cin >> learning_rate;

            std::cout << "Ingrese el valor de epsilon (ej. 1e-8): ";
            std::cin >> epsilon;

            return std::make_shared<AdagradOptimizer>(learning_rate, epsilon);
        }
        case 5: {
            float rho, epsilon;

            std::cout << "Ingrese el valor de rho para Adadelta Optimizer (ej. 0.95): ";
            std::cin >> rho;

            std::cout << "Ingrese el valor de epsilon (ej. 1e-6): ";
            std::cin >> epsilon;

            return std::make_shared<AdadeltaOptimizer>(rho, epsilon);
        }
        case 6: {
            float learning_rate, beta1, beta2, epsilon, weight_decay;

            std::cout << "Ingrese el learning rate para AdamW Optimizer (ej. 0.00001): ";
            std::cin >> learning_rate;

            std::cout << "Ingrese el valor de beta1 (ej. 0.9): ";
            std::cin >> beta1;

            std::cout << "Ingrese el valor de beta2 (ej. 0.999): ";
            std::cin >> beta2;

            std::cout << "Ingrese el valor de epsilon (ej. 1e-8): ";
            std::cin >> epsilon;

            std::cout << "Ingrese el valor de weight decay (ej. 0.0001): ";
            std::cin >> weight_decay;

            return std::make_shared<AdamWOptimizer>(learning_rate, beta1, beta2, epsilon, weight_decay);
        }
        case 7: {
            float learning_rate, alpha;

            std::cout << "Ingrese el learning rate para Low Pass Filter Optimizer (ej. 0.0001): ";
            std::cin >> learning_rate;

            std::cout << "Ingrese el valor de alpha (ej. 0.1): ";
            std::cin >> alpha;

            return std::make_shared<LowPassFilterOptimizer>(learning_rate, alpha);
        }
        default:
            std::cout << "Opción inválida. Usando Adam Optimizer con valores por defecto.\n";
            return std::make_shared<AdamOptimizer>();
    }
}

// ================================
// Funciones para Visualización
// ================================
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

        // Generar una permutación fija basada en el tamaño de entrada
        size_t input_size = positive_samples.getInputSize(); // Asegúrate de que esto corresponde al tamaño de cada imagen
        std::vector<size_t> permutation = generateFixedPermutation(input_size);

        Dataset train_positive_samples, val_positive_samples;
        Dataset train_negative_samples, val_negative_samples;

        // Divide los conjuntos de datos en entrenamiento (80%) y validación (20%) aplicando la permutación
        splitDataset(positive_samples, 0.8f, train_positive_samples,
                     val_positive_samples, permutation);
        splitDataset(negative_samples, 0.8f, train_negative_samples,
                     val_negative_samples, permutation);

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

        // Declaramos la variable para almacenar el mejor umbral global
        float best_overall_threshold = threshold;

        // Solicita al usuario el número total de épocas de entrenamiento largo
        size_t total_epochs_long;
        std::cout << "Ingrese el número total de épocas de entrenamiento largo: ";
        std::cin >> total_epochs_long;

        // Vectores para almacenar las bondades durante la evaluación
        std::vector<float> goodness_positive_vals;
        std::vector<float> goodness_negative_vals;

        // Solicita al usuario el tamaño de la capa completamente conectada
        size_t output_size;
        std::cout << "Ingrese el tamaño de la capa (número de neuronas): ";
        std::cin >> output_size;

        // Solicita al usuario la cantidad de inicializaciones y épocas por inicialización
        size_t num_initializations;
        size_t initial_epochs;
        std::cout << "Ingrese el número de inicializaciones iniciales: ";
        std::cin >> num_initializations;
        std::cout << "Ingrese el número de épocas por inicialización (1 a 4): ";
        std::cin >> initial_epochs;

        // Solicita al usuario el número de mejores modelos a recordar (ntop)
        size_t ntop;
        std::cout << "Ingrese el número de mejores modelos a recordar (ntop): ";
        std::cin >> ntop;

        // Configuración de la paciencia (tolerancia)
        size_t patience;
        std::cout << "Ingrese el número de épocas de tolerancia sin mejora (patience): ";
        std::cin >> patience;

        // Asegurarse de que la carpeta principal para los histogramas y ntop cache existan
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
                std::random_device rd;
                std::mt19937 g(rd());
                std::shuffle(combined_train_samples.begin(), combined_train_samples.end(), g);

                // Entrenamiento en la lista combinada barajada (sin OpenMP)
                for (size_t i = 0; i < combined_train_samples.size(); ++i) {
                    const Eigen::VectorXf& input = combined_train_samples[i].first.get();
                    bool is_positive = combined_train_samples[i].second;
                    Eigen::VectorXf output;
                    current_layer.forward(input, output, true, is_positive, threshold, activation, activation_derivative);
                }

                // Evaluación en conjunto de validación
                size_t correct_positive = 0;
                size_t correct_negative = 0;

                goodness_positive_vals.clear();
                goodness_negative_vals.clear();

                // Evaluación en muestras positivas (sin OpenMP)
                for (size_t i = 0; i < val_positive_samples.getNumSamples(); ++i) {
                    const Eigen::VectorXf& input = val_positive_samples.getSample(i);
                    Eigen::VectorXf output;
                    current_layer.forward(input, output, false, true, threshold, activation, activation_derivative);

                    float goodness = output.squaredNorm();
                    goodness_positive_vals.push_back(goodness);

                    if (goodness > threshold) {
                        ++correct_positive;
                    }
                }

                // Evaluación en muestras negativas (sin OpenMP)
                for (size_t i = 0; i < val_negative_samples.getNumSamples(); ++i) {
                    const Eigen::VectorXf& input = val_negative_samples.getSample(i);
                    Eigen::VectorXf output;
                    current_layer.forward(input, output, false, false, threshold, activation, activation_derivative);

                    float goodness = output.squaredNorm();
                    goodness_negative_vals.push_back(goodness);

                    if (goodness < threshold) {
                        ++correct_negative;
                    }
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
                if (dynamic_threshold) {
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

        // Evaluación en muestras positivas
        for (size_t i = 0; i < val_positive_samples.getNumSamples(); ++i) {
            const Eigen::VectorXf& input = val_positive_samples.getSample(i);
            Eigen::VectorXf output;
            best_individual_model.forward(input, output, false, true, threshold, activation, activation_derivative);

            float goodness = output.squaredNorm();

            if (goodness > threshold) {
                ++correct_positive_individual;
            }
        }

        // Evaluación en muestras negativas
        for (size_t i = 0; i < val_negative_samples.getNumSamples(); ++i) {
            const Eigen::VectorXf& input = val_negative_samples.getSample(i);
            Eigen::VectorXf output;
            best_individual_model.forward(input, output, false, false, threshold, activation, activation_derivative);

            float goodness = output.squaredNorm();

            if (goodness < threshold) {
                ++correct_negative_individual;
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

        // Evaluación en muestras positivas
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
                ++correct_positive_ensemble_vote;
            }
        }

        // Evaluación en muestras negativas
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
                ++correct_negative_ensemble_vote;
            }
        }

        double accuracy_ensemble_vote = (static_cast<double>(correct_positive_ensemble_vote + correct_negative_ensemble_vote) /
                                        (val_positive_samples.getNumSamples() + val_negative_samples.getNumSamples())) * 100.0;

        std::cout << "Precisión del ensemble mediante votación: " << accuracy_ensemble_vote << "%\n";

        // Crear el modelo promedio (promediando pesos y biases)
        std::cout << "\n--- Creando el modelo promedio (promedio de pesos y biases) ---\n";
        FullyConnectedLayer averaged_model = ensemble_models.front(); // Inicializar con el primer modelo

        // Promediar los pesos y biases de los modelos
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

        // Evaluación en muestras positivas
        for (size_t i = 0; i < val_positive_samples.getNumSamples(); ++i) {
            const Eigen::VectorXf& input = val_positive_samples.getSample(i);
            Eigen::VectorXf output;
            averaged_model.forward(input, output, false, true, threshold, activation, activation_derivative);

            float goodness = output.squaredNorm();

            if (goodness > threshold) {
                ++correct_positive_averaged;
            }
        }

        // Evaluación en muestras negativas
        for (size_t i = 0; i < val_negative_samples.getNumSamples(); ++i) {
            const Eigen::VectorXf& input = val_negative_samples.getSample(i);
            Eigen::VectorXf output;
            averaged_model.forward(input, output, false, false, threshold, activation, activation_derivative);

            float goodness = output.squaredNorm();

            if (goodness < threshold) {
                ++correct_negative_averaged;
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
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(combined_train_samples.begin(), combined_train_samples.end(), g);

            // Entrenamiento en la lista combinada barajada (sin OpenMP)
            for (size_t i = 0; i < combined_train_samples.size(); ++i) {
                const Eigen::VectorXf& input = combined_train_samples[i].first.get();
                bool is_positive = combined_train_samples[i].second;
                Eigen::VectorXf output;
                layer.forward(input, output, true, is_positive, threshold, activation, activation_derivative);
            }

            // Evaluación en conjunto de validación
            size_t correct_positive = 0;
            size_t correct_negative = 0;

            goodness_positive_vals.clear();
            goodness_negative_vals.clear();

            // Evaluación en muestras positivas (sin OpenMP)
            for (size_t i = 0; i < val_positive_size; ++i) {
                const Eigen::VectorXf& input = val_positive_samples.getSample(i);
                Eigen::VectorXf output;
                layer.forward(input, output, false, true, threshold, activation, activation_derivative);

                float goodness = output.squaredNorm();
                goodness_positive_vals.push_back(goodness);

                if (goodness > threshold) {
                    ++correct_positive;
                }
            }

            // Evaluación en muestras negativas (sin OpenMP)
            for (size_t i = 0; i < val_negative_size; ++i) {
                const Eigen::VectorXf& input = val_negative_samples.getSample(i);
                Eigen::VectorXf output;
                layer.forward(input, output, false, false, threshold, activation, activation_derivative);

                float goodness = output.squaredNorm();
                goodness_negative_vals.push_back(goodness);

                if (goodness < threshold) {
                    ++correct_negative;
                }
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
            if (dynamic_threshold) {
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

// Función principal del programa
int main() {
    trainModel();
    return 0;
}
