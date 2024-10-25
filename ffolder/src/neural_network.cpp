#include "neural_network.hpp"

#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

// Implementación de la clase Dataset

Dataset::Dataset(const std::string& directory_path) {
    loadImages(directory_path);
}

void Dataset::loadImages(const std::string& directory_path) {
    std::vector<std::string> image_files;
    cv::glob(directory_path + "/*.png", image_files);

    if (image_files.empty()) {
        throw std::runtime_error("No se encontraron imágenes en el directorio: " + directory_path);
    }

    // Preasigna espacio para evitar realocaciones
    samples.reserve(image_files.size());
    image_paths.reserve(image_files.size());

    for (const auto& file : image_files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            throw std::runtime_error("Error al leer la imagen: " + file);
        }

        // Determinar el número de canales
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

size_t Dataset::getInputSize() const {
    return input_size;
}

size_t Dataset::getNumSamples() const {
    return samples.size();
}

const Eigen::VectorXf& Dataset::getSample(size_t index) const {
    return samples.at(index);
}

const std::string& Dataset::getImagePath(size_t index) const {
    if (index >= image_paths.size()) {
        throw std::out_of_range("Índice fuera de rango en getImagePath.");
    }
    return image_paths.at(index);
}

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

void Dataset::addSample(const Eigen::VectorXf& sample, const std::string& image_path) {
    if (samples.empty()) {
        input_size = sample.size();
    } else if (sample.size() != input_size) {
        throw std::runtime_error("Tamaños de muestra inconsistentes en el conjunto de datos.");
    }
    samples.emplace_back(sample);
    image_paths.emplace_back(image_path);
}

void Dataset::setImageProperties(size_t height, size_t width, size_t channels) {
    image_height = height;
    image_width = width;
    num_channels = channels;
}

size_t Dataset::getImageHeight() const {
    return image_height;
}

size_t Dataset::getImageWidth() const {
    return image_width;
}

size_t Dataset::getNumChannels() const {
    return num_channels;
}

// Implementación de la clase FullyConnectedLayer

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

void FullyConnectedLayer::updateWeights(const Eigen::VectorXf& inputs,
                                        const Eigen::VectorXf& outputs, bool is_positive,
                                        float threshold, const std::function<float(float)>& activation_derivative) {
    float goodness = outputs.squaredNorm();
    float p = 1.0f / (1.0f + std::exp(-(goodness - threshold)));
    float y = is_positive ? 1.0f : 0.0f;

    // dL_dG = p - y
    float dL_dG = p - y;

    // dG_da = 2 * outputs
    Eigen::VectorXf dG_da = 2.0f * outputs;

    // dL_da = dL_dG * dG_da
    Eigen::VectorXf dL_da = dL_dG * dG_da;

    // Derivada de la función de activación (Leaky ReLU)
    Eigen::VectorXf activation_derivatives = pre_activations.unaryExpr(activation_derivative);

    // dL_dz = dL_da * activation_derivatives
    Eigen::VectorXf dL_dz = dL_da.array() * activation_derivatives.array();

    // Gradientes respecto a los pesos y biases
    Eigen::MatrixXf grad_weights = dL_dz * inputs.transpose();
    Eigen::VectorXf grad_biases = dL_dz;

    // Actualización de pesos y biases utilizando el optimizador
    optimizer->updateWeights(weights, grad_weights);
    optimizer->updateBiases(biases, grad_biases);
}

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

size_t FullyConnectedLayer::getInputSize() const {
    return input_size;
}

size_t FullyConnectedLayer::getOutputSize() const {
    return output_size;
}
