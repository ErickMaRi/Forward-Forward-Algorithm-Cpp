#include "neural_network.hpp"

#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

///////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////// // // CONJUNTO DE DATOS // // //
///////////////////////////////////////////////////////////////////////////////////////


/**
 * @brief Constructor de la clase Dataset que carga imágenes desde un directorio especificado.
 * @param directory_path Ruta del directorio que contiene las imágenes.
 */
Dataset::Dataset(const std::string& directory_path) {
    loadImages(directory_path);
}

/**
 * @brief Carga las imágenes desde el directorio especificado.
 * @param directory_path Ruta del directorio que contiene las imágenes a cargar.
 * @throws std::runtime_error si no se encuentran imágenes o si hay un error al leer alguna imagen.
 */
void Dataset::loadImages(const std::string& directory_path) {
    // Declaramos el vector de strings usando glob para buscar el patrón
    std::vector<std::string> image_files;
    cv::glob(directory_path + "/*.png", image_files);

    // Verificamos la existencia de la carpeta
    if (image_files.empty()) {
        throw std::runtime_error("No se encontraron imágenes en el directorio: " + directory_path);
    }

    // Preasignamos espacio para evitar realocaciones
    samples.reserve(image_files.size());
    image_paths.reserve(image_files.size());

    // Por cada archivo en el vector de strings
    for (const auto& file : image_files) {
        // Leemos la imágen
        cv::Mat img = cv::imread(file, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            throw std::runtime_error("Error al leer la imagen: " + file);
        }

        // Determinamos el número de canales, deberíamos hacerlo por fuera del for
        //      asumiendo que todas las imágenes tienen la misma dimensionalidad
        if (samples.empty()) {
            image_height = img.rows;
            image_width = img.cols;
            num_channels = img.channels();
            input_size = image_height * image_width * num_channels;
        } else {
            if (static_cast<size_t>(img.rows) != image_height || static_cast<size_t>(img.cols) != image_width) {
                throw std::runtime_error("Las imágenes deben tener el mismo tamaño.");
            }
            if (static_cast<size_t>(img.channels()) != num_channels) {
                throw std::runtime_error("Todas las imágenes deben tener el mismo número de canales.");
            }
        }

        img.convertTo(img, CV_32F, 1.0 / 255.0);

        // Asegurarse de que la matriz esté en un formato continuo
        if (!img.isContinuous()) {
            img = img.clone();
        }

        // Mapear directamente los datos de la imagen a un vector de Eigen
        Eigen::Map<Eigen::VectorXf> sample_map(reinterpret_cast<float*>(img.data),
                                              img.rows * img.cols * img.channels());

        // Crear una copia del mapeo para almacenar en el vector de muestras
        Eigen::VectorXf sample = sample_map;

        samples.emplace_back(std::move(sample));
        image_paths.emplace_back(file);
    }

    std::cout << "Cargadas " << samples.size() << " muestras de " << directory_path << "\n";
}

/**
 * @brief Retorna el tamaño de la entrada de la red.
 * @return Entero sin signo con el tamaño de la entrada.
 */
size_t Dataset::getInputSize() const {
    return input_size;
}

/**
 * @brief Retorna el número total de muestras en el conjunto de datos.
 * @return Entero sin signo con el número de muestras.
 */
size_t Dataset::getNumSamples() const {
    return samples.size();
}

/**
 * @brief Obtiene la muestra en el índice especificado.
 * @param index Índice de la muestra a obtener.
 * @return Referencia constante a un vector de Eigen que representa la muestra.
 * @throws std::out_of_range si el índice está fuera de rango.
 */
const Eigen::VectorXf& Dataset::getSample(size_t index) const {
    return samples.at(index);
}

/**
 * @brief Obtiene la ruta de la imagen en el índice especificado.
 * @param index Índice de la imagen cuya ruta se desea obtener.
 * @return Referencia constante a una cadena que representa la ruta de la imagen.
 * @throws std::out_of_range si el índice está fuera de rango.
 */
const std::string& Dataset::getImagePath(size_t index) const {
    if (index >= image_paths.size()) {
        throw std::out_of_range("Índice fuera de rango en getImagePath.");
    }
    return image_paths.at(index);
}

/**
 * @brief Mezcla aleatoriamente las muestras y las rutas de imágenes en el conjunto de datos.
 */
void Dataset::shuffle() {
    std::random_device rd;
    std::mt19937 g(rd());

    // Shuffling in-place usando el algoritmo Fisher-Yates
    for (size_t i = samples.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(g);
        std::swap(samples[i], samples[j]);
        std::swap(image_paths[i], image_paths[j]);
    }
}

/**
 * @brief Añade una nueva muestra y su ruta de imagen al conjunto de datos.
 * @param sample Vector de Eigen que representa la muestra a añadir.
 * @param image_path Ruta de la imagen correspondiente a la muestra.
 * @throws std::runtime_error si el tamaño de la muestra no coincide con el tamaño de entrada.
 */
void Dataset::addSample(const Eigen::VectorXf& sample, const std::string& image_path) {
    if (samples.empty()) {
        input_size = sample.size();
    } else if (sample.size() != input_size) {
        throw std::runtime_error("Tamaños de muestra inconsistentes en el conjunto de datos.");
    }
    samples.emplace_back(sample);
    image_paths.emplace_back(image_path);
}

/**
 * @brief Establece las propiedades de las imágenes en el conjunto de datos.
 * @param height Altura de las imágenes.
 * @param width Anchura de las imágenes.
 * @param channels Número de canales de las imágenes.
 */
void Dataset::setImageProperties(size_t height, size_t width, size_t channels) {
    image_height = height;
    image_width = width;
    num_channels = channels;
}

/**
 * @brief Retorna la altura de las imágenes en el conjunto de datos.
 * @return Entero sin signo con la altura de las imágenes.
 */
size_t Dataset::getImageHeight() const {
    return image_height;
}

/**
 * @brief Retorna la anchura de las imágenes en el conjunto de datos.
 * @return Entero sin signo con la anchura de las imágenes.
 */
size_t Dataset::getImageWidth() const {
    return image_width;
}

/**
 * @brief Retorna el número de canales de las imágenes en el conjunto de datos.
 * @return Entero sin signo con el número de canales.
 */
size_t Dataset::getNumChannels() const {
    return num_channels;
}

///////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////// // // CAPA RED NEURONAL // // //
///////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Constructor de la clase FullyConnectedLayer que inicializa la capa con pesos y biases.
 * @param input_size Tamaño de la entrada de la capa.
 * @param output_size Tamaño de la salida de la capa.
 * @param optimizer_ptr Puntero compartido al optimizador utilizado para actualizar pesos y biases.
 */
FullyConnectedLayer::FullyConnectedLayer(size_t input_size, size_t output_size,
                                         std::shared_ptr<Optimizer> optimizer_ptr)
    : input_size(input_size), output_size(output_size),
      optimizer(std::move(optimizer_ptr)),
      weights(output_size, input_size),
      biases(output_size),
      pre_activations(output_size) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Inicialización con distribución de He (He et al.)
    float std_dev = std::sqrt(2.0f / input_size);
    std::normal_distribution<float> weight_dist(0.0f, std_dev);

    // Asignar pesos utilizando Eigen's generar aleatorio
    for (int i = 0; i < weights.size(); ++i) {
        weights.data()[i] = weight_dist(gen);
    }

    biases.setConstant(0.01f);
}

/**
 * @brief Realiza el pase hacia adelante a través de la capa completamente conectada.
 * @param inputs Vector de Eigen que representa las entradas a la capa.
 * @param outputs Vector de Eigen donde se almacenarán las salidas de la capa.
 * @param learn Indica si se deben actualizar los pesos durante este pase.
 * @param positive Indica si la muestra actual es positiva para el aprendizaje.
 * @param threshold Umbral utilizado en el cálculo de la pérdida.
 * @param activation Función de activación a aplicar a las preactivaciones.
 * @param activation_derivative Derivada de la función de activación utilizada.
 * @throws std::invalid_argument si el tamaño de las entradas no coincide con input_size.
 */
void FullyConnectedLayer::forward(const Eigen::VectorXf& inputs,
                                  Eigen::VectorXf& outputs, bool learn, bool positive,
                                  float& threshold, const std::function<float(float)>& activation,
                                  const std::function<float(float)>& activation_derivative) {
    if (inputs.size() != input_size) {
        throw std::invalid_argument("El tamaño de entrada no coincide con input_size.");
    }

    // Calcula preactivaciones: weights * inputs + biases
    pre_activations.noalias() = weights * inputs;
    pre_activations += biases;

    // Aplica la función de activación
    outputs = pre_activations.unaryExpr(activation);

    if (learn) {
        updateWeights(inputs, outputs, positive, threshold, activation_derivative);
    }
}

/**
 * @brief Actualiza los pesos y biases de la capa utilizando el optimizador.
 * @param inputs Vector de Eigen que representa las entradas a la capa.
 * @param outputs Vector de Eigen que representa las salidas de la capa.
 * @param is_positive Indica si la muestra actual es positiva para el aprendizaje.
 * @param threshold Umbral utilizado en el cálculo de la pérdida.
 * @param activation_derivative Derivada de la función de activación utilizada.
 */
void FullyConnectedLayer::updateWeights(const Eigen::VectorXf& inputs,
                                        const Eigen::VectorXf& outputs, bool is_positive,
                                        float threshold, const std::function<float(float)>& activation_derivative) {
    float goodness = outputs.squaredNorm();
    float p = 1.0f / (1.0f + std::exp(-(goodness - threshold)));
    float y = is_positive ? 1.0f : 0.0f;

    // Cálculo de dL/dp sin simplificar
    // float dL_dp = - ( (y / p) - ((1.0f - y) / (1.0f - p)) );
    // dL_dp multiplicado por dp_dG (p(1-p)) da como resultado p - y

    // Simplificamos el cálculo, evitamos inestabilidad para p cercano a cero.
    float dL_dG = p - y;

    // Cálculo de dG/da
    Eigen::VectorXf dG_da = 2.0f * outputs;

    // Cálculo de dL/da
    Eigen::VectorXf dL_da = dL_dG * dG_da;

    // Derivada de la función de activación (Leaky ReLU)
    Eigen::VectorXf activation_derivatives = pre_activations.unaryExpr(activation_derivative);

    // Cálculo de dL/dz
    Eigen::VectorXf dL_dz = dL_da.array() * activation_derivatives.array();

    // Gradientes respecto a los pesos y biases
    Eigen::MatrixXf grad_weights = dL_dz * inputs.transpose();
    Eigen::VectorXf grad_biases = dL_dz;

    // Actualización de pesos y biases utilizando el optimizador
    optimizer->updateWeights(weights, grad_weights);
    optimizer->updateBiases(biases, grad_biases);
}

/**
 * @brief Guarda el modelo de la capa en un archivo especificado.
 * @param filepath Ruta del archivo donde se guardará el modelo.
 * @throws std::runtime_error si no se puede abrir el archivo para guardar el modelo.
 */
void FullyConnectedLayer::saveModel(const std::string& filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo para guardar el modelo: " + filepath);
    }

    ofs.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
    ofs.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

    ofs.write(reinterpret_cast<const char*>(weights.data()),
              weights.size() * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(biases.data()),
              biases.size() * sizeof(float));

    ofs.close();
}

/**
 * @brief Carga el modelo de la capa desde un archivo especificado.
 * @param filepath Ruta del archivo desde donde se cargará el modelo.
 * @throws std::runtime_error si no se puede abrir el archivo para cargar el modelo.
 */
void FullyConnectedLayer::loadModel(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo para cargar el modelo: " + filepath);
    }

    ifs.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
    ifs.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));

    weights.resize(output_size, input_size);
    biases.resize(output_size);

    ifs.read(reinterpret_cast<char*>(weights.data()),
             weights.size() * sizeof(float));
    ifs.read(reinterpret_cast<char*>(biases.data()),
             biases.size() * sizeof(float));

    pre_activations.resize(output_size);

    ifs.close();
}

/**
 * @brief Retorna el tamaño de la entrada de la capa
 * @return Entero sin signo con el tamaño de la entrada.
 */
size_t FullyConnectedLayer::getInputSize() const {
    return input_size;
}

/**
 * @brief Retorna el tamaño de la salida de la capa
 * @return Entero sin signo con el tamaño de la salida.
 */
size_t FullyConnectedLayer::getOutputSize() const {
    return output_size;
}
