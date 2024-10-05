// forward_forward.cpp

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <fstream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <cmath>
#include <memory>
#include <stdexcept>

// Incluimos Eigen para operaciones matriciales
#include <Eigen/Dense>

// Incluimos OpenCV para visualización
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

// ================================
// Estructura para Datos del Gráfico de Dispersión
// ================================
struct ScatterPlotData {
    cv::Mat image;                       // Imagen asociada
    std::vector<cv::Point> points;       // Puntos de interés en la imagen
    std::vector<std::string> image_paths; // Rutas de las imágenes
};

// ================================
// Clase Dataset
// ================================

/**
 * @brief Clase para manejar conjuntos de datos de imágenes.
 */
class Dataset {
public:
    /**
     * @brief Constructor que carga imágenes desde un directorio.
     * @param directory_path Ruta al directorio que contiene las imágenes.
     */
    Dataset(const std::string& directory_path);

    // Constructores por defecto y de copia
    Dataset() = default;
    Dataset(const Dataset& other);

    /**
     * @brief Obtiene el tamaño de entrada de cada muestra.
     * @return Tamaño de entrada.
     */
    size_t getInputSize() const;

    /**
     * @brief Obtiene el número de muestras en el conjunto de datos.
     * @return Número de muestras.
     */
    size_t getNumSamples() const;

    /**
     * @brief Obtiene una muestra específica.
     * @param index Índice de la muestra.
     * @return Referencia constante a la muestra.
     */
    const Eigen::VectorXf& getSample(size_t index) const;

    /**
     * @brief Obtiene la ruta de la imagen correspondiente a una muestra.
     * @param index Índice de la muestra.
     * @return Referencia constante a la ruta de la imagen.
     */
    const std::string& getImagePath(size_t index) const;

    /**
     * @brief Mezcla aleatoriamente las muestras del conjunto de datos.
     */
    void shuffle();

    /**
     * @brief Añade una nueva muestra al conjunto de datos.
     * @param sample Vector de características de la muestra.
     * @param image_path Ruta de la imagen correspondiente.
     */
    void addSample(const Eigen::VectorXf& sample,
                   const std::string& image_path);

    // Getters para propiedades de la imagen
    size_t getImageHeight() const { return image_height; }
    size_t getImageWidth() const { return image_width; }
    size_t getNumChannels() const { return num_channels; }

    /**
     * @brief Establece las propiedades de las imágenes en el conjunto de datos.
     * @param height Altura de la imagen.
     * @param width Ancho de la imagen.
     * @param channels Número de canales de la imagen.
     */
    void setImageProperties(size_t height, size_t width, size_t channels);

private:
    size_t input_size = 0;             // Tamaño de entrada de cada muestra
    size_t image_height = 0;           // Altura de las imágenes
    size_t image_width = 0;            // Ancho de las imágenes
    size_t num_channels = 0;           // Número de canales de las imágenes
    std::vector<Eigen::VectorXf> samples;      // Vector de muestras
    std::vector<std::string> image_paths;      // Rutas de las imágenes

    /**
     * @brief Carga imágenes desde un directorio y las convierte en vectores de características.
     * @param directory_path Ruta al directorio que contiene las imágenes.
     */
    void loadImages(const std::string& directory_path);
};

// Constructor por defecto de copia
Dataset::Dataset(const Dataset& other) = default;

// Constructor que carga imágenes desde un directorio
Dataset::Dataset(const std::string& directory_path) {
    loadImages(directory_path);
}

// Implementación de la función loadImages
void Dataset::loadImages(const std::string& directory_path) {
    std::vector<std::string> image_files;
    cv::glob(directory_path + "/*.png", image_files); // Busca archivos .png

    if (image_files.empty()) {
        throw std::runtime_error("No se encontraron imágenes en el directorio: " + directory_path);
    }

    for (const auto& file : image_files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_COLOR); // Lee la imagen en color
        if (img.empty()) {
            throw std::runtime_error("Error al leer la imagen: " + file);
        }

        // Establece las propiedades de la imagen si es la primera
        if (image_height == 0 && image_width == 0) {
            image_height = img.rows;
            image_width = img.cols;
        } else if (img.rows != image_height || img.cols != image_width) {
            throw std::runtime_error("Las imágenes deben tener el mismo tamaño.");
        }

        img.convertTo(img, CV_32F, 1.0 / 255.0); // Normaliza los píxeles

        // Aplana la imagen en un VectorXf de Eigen
        Eigen::VectorXf sample(img.rows * img.cols * img.channels());
        int idx = 0;
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                cv::Vec3f pixel = img.at<cv::Vec3f>(i, j);
                for (int c = 0; c < img.channels(); ++c) {
                    sample(idx++) = pixel[c];
                }
            }
        }

        // Establece el tamaño de entrada si es la primera muestra
        if (samples.empty()) {
            input_size = sample.size();
        } else if (sample.size() != input_size) {
            throw std::runtime_error("Tamaños de muestra inconsistentes en el conjunto de datos.");
        }

        samples.push_back(std::move(sample));
        image_paths.push_back(file);
    }

    num_channels = 3; // Asumimos imágenes RGB
    std::cout << "Cargadas " << samples.size() << " muestras de " << directory_path << "\n";
}

// Obtiene la ruta de una imagen específica
const std::string& Dataset::getImagePath(size_t index) const {
    if (index >= image_paths.size()) {
        throw std::out_of_range("Índice fuera de rango en getImagePath.");
    }
    return image_paths.at(index);
}

// Obtiene el tamaño de entrada
size_t Dataset::getInputSize() const {
    return input_size;
}

// Obtiene el número de muestras
size_t Dataset::getNumSamples() const {
    return samples.size();
}

// Añade una nueva muestra al conjunto de datos
void Dataset::addSample(const Eigen::VectorXf& sample,
                        const std::string& image_path) {
    if (samples.empty()) {
        input_size = sample.size();
    } else if (sample.size() != input_size) {
        throw std::runtime_error("Tamaños de muestra inconsistentes en el conjunto de datos.");
    }
    samples.push_back(sample);
    image_paths.push_back(image_path);
}

// Obtiene una muestra específica
const Eigen::VectorXf& Dataset::getSample(size_t index) const {
    return samples.at(index);
}

// Mezcla aleatoriamente las muestras
void Dataset::shuffle() {
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<size_t> indices(samples.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<Eigen::VectorXf> shuffled_samples(samples.size());
    std::vector<std::string> shuffled_image_paths(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        shuffled_samples[i] = samples[indices[i]];
        shuffled_image_paths[i] = image_paths[indices[i]];
    }
    samples = std::move(shuffled_samples);
    image_paths = std::move(shuffled_image_paths);
}

// Establece las propiedades de las imágenes
void Dataset::setImageProperties(size_t height, size_t width, size_t channels) {
    image_height = height;
    image_width = width;
    num_channels = channels;
}

// ================================
// Clases de Optimización
// ================================

/**
 * @brief Clase base para optimizadores.
 */
class Optimizer {
public:
    /**
     * @brief Actualiza los pesos de una capa.
     * @param weights Matriz de pesos de la capa.
     * @param gradients Gradientes calculados.
     */
    virtual void updateWeights(Eigen::MatrixXf& weights,
                               const Eigen::MatrixXf& gradients) = 0;

    /**
     * @brief Actualiza los sesgos de una capa.
     * @param biases Vector de sesgos de la capa.
     * @param gradients Gradientes calculados.
     */
    virtual void updateBiases(Eigen::VectorXf& biases,
                              const Eigen::VectorXf& gradients) = 0;

    virtual ~Optimizer() = default;
};

/**
 * @brief Implementación del optimizador Adam.
 */
class AdamOptimizer : public Optimizer {
public:
    /**
     * @brief Constructor con parámetros personalizables.
     * @param learning_rate Tasa de aprendizaje.
     * @param beta1 Primer parámetro de momento.
     * @param beta2 Segundo parámetro de momento.
     * @param epsilon Término de estabilidad numérica.
     */
    AdamOptimizer(float learning_rate = 0.001f, float beta1 = 0.9f,
                  float beta2 = 0.999f, float epsilon = 1e-8f)
        : lr(learning_rate), beta1(beta1), beta2(beta2), eps(epsilon),
          t_weights(0), t_biases(0) {}

    /**
     * @brief Actualiza los pesos utilizando el algoritmo Adam.
     * @param weights Matriz de pesos a actualizar.
     * @param gradients Gradientes calculados.
     */
    void updateWeights(Eigen::MatrixXf& weights,
                       const Eigen::MatrixXf& gradients) override {
        if (m_weights.size() == 0) {
            m_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
            v_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        }

        ++t_weights;
        m_weights = beta1 * m_weights + (1.0f - beta1) * gradients;
        v_weights = beta2 * v_weights + (1.0f - beta2) * gradients.array().square().matrix();
        Eigen::MatrixXf m_hat = m_weights.array() / (1.0f - std::pow(beta1, t_weights));
        Eigen::MatrixXf v_hat = v_weights.array() / (1.0f - std::pow(beta2, t_weights));

        // Actualiza los pesos con corrección de bias
        weights.array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
    }

    /**
     * @brief Actualiza los sesgos utilizando el algoritmo Adam.
     * @param biases Vector de sesgos a actualizar.
     * @param gradients Gradientes calculados.
     */
    void updateBiases(Eigen::VectorXf& biases,
                      const Eigen::VectorXf& gradients) override {
        if (m_biases.size() == 0) {
            m_biases = Eigen::VectorXf::Zero(biases.size());
            v_biases = Eigen::VectorXf::Zero(biases.size());
        }

        ++t_biases;
        m_biases = beta1 * m_biases + (1.0f - beta1) * gradients;
        v_biases = beta2 * v_biases + (1.0f - beta2) * gradients.array().square().matrix();
        Eigen::VectorXf m_hat = m_biases.array() / (1.0f - std::pow(beta1, t_biases));
        Eigen::VectorXf v_hat = v_biases.array() / (1.0f - std::pow(beta2, t_biases));

        // Actualiza los sesgos con corrección de bias
        biases.array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
    }

private:
    float lr;        // Tasa de aprendizaje
    float beta1;     // Parámetro de momento 1
    float beta2;     // Parámetro de momento 2
    float eps;       // Término de estabilidad numérica
    int t_weights;   // Contador de pasos para pesos
    int t_biases;    // Contador de pasos para sesgos

    Eigen::MatrixXf m_weights;   // Momento para pesos
    Eigen::MatrixXf v_weights;   // Segundo momento para pesos
    Eigen::VectorXf m_biases;    // Momento para sesgos
    Eigen::VectorXf v_biases;    // Segundo momento para sesgos
};

// ================================
// Clase FullyConnectedLayer
// ================================

/**
 * @brief Clase que representa una capa completamente conectada en la red neuronal.
 */
class FullyConnectedLayer {
public:
    /**
     * @brief Constructor que inicializa pesos y sesgos.
     * @param input_size Tamaño de la entrada.
     * @param output_size Tamaño de la salida.
     * @param optimizer_ptr Puntero al optimizador a utilizar.
     */
    FullyConnectedLayer(size_t input_size, size_t output_size,
                        std::shared_ptr<Optimizer> optimizer_ptr);

        // Constructor de copia profunda
    FullyConnectedLayer(const FullyConnectedLayer& other) {
        input_size = other.input_size;
        output_size = other.output_size;
        optimizer = other.optimizer; // Compartir el mismo optimizador
        weights = other.weights;
        biases = other.biases;
        pre_activations = other.pre_activations;
    }

    // Operador de asignación
    FullyConnectedLayer& operator=(const FullyConnectedLayer& other) {
        if (this != &other) {
            input_size = other.input_size;
            output_size = other.output_size;
            optimizer = other.optimizer; // Compartir el mismo optimizador
            weights = other.weights;
            biases = other.biases;
            pre_activations = other.pre_activations;
        }
        return *this;
    }

    /**
     * @brief Realiza la pasada hacia adelante y opcionalmente actualiza los pesos.
     * @param inputs Vector de entrada.
     * @param outputs Vector de salida.
     * @param learn Indica si se debe aprender (actualizar pesos).
     * @param positive Indica si la muestra es positiva o negativa.
     * @param threshold Umbral para la bondad.
     * @param activation Función de activación.
     * @param activation_derivative Derivada de la función de activación.
     */
    void forward(const Eigen::VectorXf& inputs,
                Eigen::VectorXf& outputs, bool learn, bool positive,
                float& threshold, const std::function<float(float)>& activation,
                const std::function<float(float)>& activation_derivative);

    /**
     * @brief Guarda el modelo (pesos y sesgos) en un archivo.
     * @param filepath Ruta del archivo donde se guardará el modelo.
     */
    void saveModel(const std::string& filepath) const;

    /**
     * @brief Carga el modelo (pesos y sesgos) desde un archivo.
     * @param filepath Ruta del archivo desde donde se cargará el modelo.
     */
    void loadModel(const std::string& filepath);

    // Getters para tamaños de entrada y salida
    size_t getInputSize() const { return input_size; }
    size_t getOutputSize() const { return output_size; }

private:
    size_t input_size;                    // Tamaño de la entrada
    size_t output_size;                   // Tamaño de la salida
    std::shared_ptr<Optimizer> optimizer; // Puntero al optimizador
    Eigen::MatrixXf weights;              // Matriz de pesos
    Eigen::VectorXf biases;               // Vector de sesgos
    Eigen::VectorXf pre_activations;      // Activaciones antes de la función de activación

    /**
     * @brief Actualiza los pesos y sesgos basándose en los gradientes.
     * @param inputs Vector de entrada.
     * @param outputs Vector de salida.
     * @param is_positive Indica si la muestra es positiva o negativa.
     * @param threshold Umbral para la bondad.
     * @param activation_derivative Derivada de la función de activación.
     */
    void updateWeights(const Eigen::VectorXf& inputs,
                       const Eigen::VectorXf& outputs, bool is_positive,
                       float threshold,
                       const std::function<float(float)>& activation_derivative);
};

// Constructor que inicializa pesos y sesgos
FullyConnectedLayer::FullyConnectedLayer(size_t input_size,
                                         size_t output_size,
                                         std::shared_ptr<Optimizer> optimizer_ptr)
    : input_size(input_size), output_size(output_size),
      optimizer(std::move(optimizer_ptr)) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float std_dev = std::sqrt(2.0f / input_size); // Inicialización He
    std::normal_distribution<float> weight_dist(0.0f, std_dev);

    // Inicializa pesos con distribución normal
    weights = Eigen::MatrixXf::NullaryExpr(output_size, input_size, [&]() { return weight_dist(gen); });
    biases = Eigen::VectorXf::Constant(output_size, 0.01f); // Inicializa sesgos pequeños
    pre_activations.resize(output_size);
}

// Implementación de la función forward
void FullyConnectedLayer::forward(const Eigen::VectorXf& inputs,
                                  Eigen::VectorXf& outputs, bool learn, bool positive,
                                  float& threshold, const std::function<float(float)>& activation,
                                  const std::function<float(float)>& activation_derivative) {
    if (inputs.size() != input_size) {
        throw std::invalid_argument("El tamaño de entrada no coincide con input_size.");
    }

    // Calcula las pre-activaciones: W * X + b
    pre_activations.noalias() = weights * inputs + biases;

    // Aplica la función de activación
    outputs = pre_activations.unaryExpr(activation);

    // Si se está en modo de aprendizaje, actualiza los pesos y sesgos
    if (learn) {
        updateWeights(inputs, outputs, positive, threshold, activation_derivative);
    }
}

// Implementación de la función updateWeights
void FullyConnectedLayer::updateWeights(const Eigen::VectorXf& inputs,
                                        const Eigen::VectorXf& outputs, bool is_positive,
                                        float threshold, const std::function<float(float)>& activation_derivative) {
    // Calcula la bondad como la norma cuadrada de las salidas
    float goodness = outputs.squaredNorm();

    // Calcula la probabilidad usando la función sigmoide
    float p = 1.0f / (1.0f + std::exp(-(goodness - threshold)));

    // Etiqueta objetivo: 1 para positivo, 0 para negativo
    float y = is_positive ? 1.0f : 0.0f;

    // Derivada de la pérdida con respecto a la bondad
    float dL_dG = p - y;

    // Derivada de la bondad con respecto a las activaciones
    Eigen::VectorXf dG_da = 2.0f * outputs;

    // Derivada de la pérdida con respecto a las activaciones
    Eigen::VectorXf dL_da = dL_dG * dG_da;

    // Derivada de la pérdida con respecto a las pre-activaciones
    Eigen::VectorXf dL_dz = dL_da.array() * pre_activations.unaryExpr(activation_derivative).array();

    // Gradientes para pesos y sesgos
    Eigen::MatrixXf grad_weights = dL_dz * inputs.transpose();
    Eigen::VectorXf grad_biases = dL_dz;

    // Actualiza pesos y sesgos usando el optimizador
    optimizer->updateWeights(weights, grad_weights);
    optimizer->updateBiases(biases, grad_biases);
}


// Guarda el modelo en un archivo binario
void FullyConnectedLayer::saveModel(const std::string& filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo para guardar el modelo: " + filepath);
    }

    // Guarda tamaños de entrada y salida
    ofs.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
    ofs.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

    // Guarda pesos y sesgos
    ofs.write(reinterpret_cast<const char*>(weights.data()),
              weights.size() * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(biases.data()),
              biases.size() * sizeof(float));

    ofs.close();
}

// Carga el modelo desde un archivo binario
void FullyConnectedLayer::loadModel(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo para cargar el modelo: " + filepath);
    }

    // Carga tamaños de entrada y salida
    ifs.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
    ifs.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));

    // Redimensiona matrices de pesos y sesgos
    weights.resize(output_size, input_size);
    biases.resize(output_size);

    // Carga pesos y sesgos
    ifs.read(reinterpret_cast<char*>(weights.data()),
             weights.size() * sizeof(float));
    ifs.read(reinterpret_cast<char*>(biases.data()),
             biases.size() * sizeof(float));

    pre_activations.resize(output_size);

    ifs.close();
}

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
 * @brief Genera y guarda los histogramas de "goodness" para los conjuntos positivos y negativos.
 * @param goodness_positive_vals Vectores de "goodness" para muestras positivas.
 * @param goodness_negative_vals Vectores de "goodness" para muestras negativas.
 * @param threshold Umbral actual utilizado para la clasificación.
 * @param save_path Ruta donde se guardarán las imágenes de los histogramas.
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

    // Opcional: Mostrar las imágenes si se desea
    /*
    cv::imshow("Histograma de Bondad - Positivos", histImagePositive);
    cv::imshow("Histograma de Bondad - Negativos", histImageNegative);
    cv::waitKey(1); // Espera breve para mostrar la imagen
    */
}



/**
 * @brief Función principal que maneja el flujo de entrenamiento del modelo.
 */
void trainModel() {
    try {
        std::cout << "Cargando conjuntos de datos...\n";
        Dataset positive_samples("positive_images/"); // Directorio de imágenes positivas
        Dataset negative_samples("negative_images/"); // Directorio de imágenes negativas

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
