#include "neural_network.hpp"

#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

///////////////////////////////////////////////////////////////////////////////////////
//                           CONJUNTO DE DATOS (Dataset)                              //
///////////////////////////////////////////////////////////////////////////////////////

Dataset::Dataset(const std::string& directory_path) {
    loadImages(directory_path);
}

void Dataset::loadImages(const std::string& directory_path) {
    std::vector<std::string> image_files;
    cv::glob(directory_path + "/*.png", image_files);

    if (image_files.empty()) {
        throw std::runtime_error("No se encontraron imágenes en el directorio: " + directory_path);
    }

    samples.reserve(image_files.size());
    image_paths.reserve(image_files.size());

    for (const auto& file : image_files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            throw std::runtime_error("Error al leer la imagen: " + file);
        }

        if (samples.empty()) {
            image_height = img.rows;
            image_width = img.cols;
            num_channels = img.channels();
            input_size = image_height * image_width * num_channels;
        } else {
            if (static_cast<size_t>(img.rows) != image_height ||
                static_cast<size_t>(img.cols) != image_width) {
                throw std::runtime_error("Las imágenes deben tener el mismo tamaño.");
            }
            if (static_cast<size_t>(img.channels()) != num_channels) {
                throw std::runtime_error("Todas las imágenes deben tener el mismo número de canales.");
            }
        }

        // Normalizamos la imagen a flotantes entre 0 y 1
        img.convertTo(img, CV_32F, 1.0f / 255.0f);

        // Asegurarnos de que sea continua en memoria
        if (!img.isContinuous()) {
            img = img.clone();
        }

        // Mapear a Eigen
        Eigen::Map<Eigen::VectorXf> sample_map(
            reinterpret_cast<float*>(img.data),
            img.rows * img.cols * img.channels()
        );

        // Copiamos el mapeo a nuestro vector
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
    static std::mt19937 g(std::random_device{}());
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
    } else if (static_cast<size_t>(sample.size()) != input_size) {
        throw std::runtime_error("Tamaños de muestra inconsistentes en el Dataset.");
    }
    samples.emplace_back(sample);
    image_paths.emplace_back(image_path);
}

void Dataset::setImageProperties(size_t height, size_t width, size_t channels) {
    image_height = height;
    image_width  = width;
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

///////////////////////////////////////////////////////////////////////////////////////
//                 CAPA COMPLETAMENTE CONECTADA (FullyConnectedLayer)                //
///////////////////////////////////////////////////////////////////////////////////////

FullyConnectedLayer::FullyConnectedLayer(size_t input_size, size_t output_size,
                                         std::shared_ptr<Optimizer> optimizer_ptr)
    : input_size(input_size),
      output_size(output_size),
      optimizer(std::move(optimizer_ptr)),
      weights(output_size, input_size),
      biases(output_size),
      pre_activations(output_size),
      // Inicializamos los acumuladores de gradientes en cero:
      grad_weights_accum(Eigen::MatrixXf::Zero(output_size, input_size)),
      grad_biases_accum(Eigen::VectorXf::Zero(output_size)),
      batch_count(0)
{
    static std::mt19937 gen(std::random_device{}());

    // Inicialización de He
    float std_dev = std::sqrt(2.0f / static_cast<float>(input_size));
    std::normal_distribution<float> weight_dist(0.0f, std_dev);

    // Rellenamos la matriz de pesos con la distribución normal
    for (int i = 0; i < weights.size(); ++i) {
        weights.data()[i] = weight_dist(gen);
    }

    // Sesgos en un valor pequeño inicial
    biases.setConstant(0.01f);
}

void FullyConnectedLayer::forward(const Eigen::VectorXf& inputs,
                                  Eigen::VectorXf& outputs,
                                  bool learn, bool positive,
                                  float& threshold,
                                  const std::function<float(float)>& activation,
                                  const std::function<float(float)>& activation_derivative)
{
    if (static_cast<size_t>(inputs.size()) != input_size) {
        throw std::invalid_argument("El tamaño de entrada no coincide con input_size.");
    }

    // pre_activations = W*x + b
    pre_activations.noalias() = weights * inputs;
    pre_activations += biases;

    // Aplicar función de activación, guardamos en 'outputs'
    outputs = pre_activations.unaryExpr(activation);

    // Si estamos en fase de entrenamiento (learn = true), acumulamos gradientes
    if (learn) {
        // 1) Acumular gradientes (no actualizamos de inmediato)
        accumulateGradients(inputs, outputs, positive, threshold, activation_derivative);

        // 2) Aumentamos el contador de muestras del minibatch
        ++batch_count;

        // 3) Si hemos llegado al tamaño de minibatch, actualizamos y reseteamos
        if (batch_count >= mini_batch_size) {
            // Escalamos los gradientes acumulados y actualizamos
            optimizer->updateWeights(weights, grad_weights_accum / float(mini_batch_size));
            optimizer->updateBiases(biases, grad_biases_accum / float(mini_batch_size));

            // Reseteamos
            grad_weights_accum.setZero();
            grad_biases_accum.setZero();
            batch_count = 0;
        }
    }
}

void FullyConnectedLayer::accumulateGradients(const Eigen::VectorXf& inputs,
                                              const Eigen::VectorXf& outputs,
                                              bool is_positive,
                                              float threshold,
                                              const std::function<float(float)>& activation_derivative)
{
    // Cálculo de goodness (G) = ||outputs||^2
    float goodness = outputs.squaredNorm();

    // Probabilidad p = 1 / [1 + e^{-(G - threshold)}]
    float p = 1.0f / (1.0f + std::exp(-(goodness - threshold)));
    float y = is_positive ? 1.0f : 0.0f;

    // dL/dG simplificado (con la idea de cross-entropy y sigma)
    float dL_dG = p - y;

    // dG/da = 2*outputs
    Eigen::VectorXf dG_da = 2.0f * outputs;

    // dL/da = dL/dG * dG/da
    Eigen::VectorXf dL_da = dL_dG * dG_da;

    // Derivada de la activación (Leaky ReLU) en pre_activations
    Eigen::VectorXf act_derivs = pre_activations.unaryExpr(activation_derivative);

    // dL/dz = dL/da * d(a)/d(z)
    Eigen::VectorXf dL_dz = dL_da.array() * act_derivs.array();

    // grad_weights es outer product de dL_dz con inputs
    Eigen::MatrixXf grad_w = dL_dz * inputs.transpose();
    Eigen::VectorXf grad_b = dL_dz;

    // ---- Acumular ----
    grad_weights_accum += grad_w;
    grad_biases_accum  += grad_b;
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

///////////////////////////////////////////////////////////////////////////////////////
//                 FUNCIÓN DE ACTIVACIÓN (Leaky ReLU y su derivada)                  //
///////////////////////////////////////////////////////////////////////////////////////

std::function<float(float)> activation = [](float x) -> float {
    const float alpha = 1.0f / 64.0f; // 2^-6
    return (x >= 0.0f) ? x : alpha * x;
};

std::function<float(float)> activation_derivative = [](float x) -> float {
    const float alpha = 1.0f / 64.0f;
    return (x >= 0.0f) ? 1.0f : alpha;
};
