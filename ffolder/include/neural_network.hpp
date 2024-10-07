#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "optimizer.hpp"

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <memory>
#include <functional>

/**
 * @brief Clase para manejar conjuntos de datos de imágenes.
 */
class Dataset {
public:
    Dataset() = default;
    Dataset(const std::string& directory_path);
    Dataset(const Dataset& other) = default;

    size_t getInputSize() const;
    size_t getNumSamples() const;
    const Eigen::VectorXf& getSample(size_t index) const;
    const std::string& getImagePath(size_t index) const;
    void shuffle();
    void addSample(const Eigen::VectorXf& sample, const std::string& image_path);
    void setImageProperties(size_t height, size_t width, size_t channels);
    size_t getImageHeight() const;
    size_t getImageWidth() const;
    size_t getNumChannels() const;

private:
    size_t input_size = 0;
    size_t image_height = 0;
    size_t image_width = 0;
    size_t num_channels = 0;
    std::vector<Eigen::VectorXf> samples;
    std::vector<std::string> image_paths;

    void loadImages(const std::string& directory_path);
};

/**
 * @brief Clase que representa una capa completamente conectada en la red neuronal.
 */
class FullyConnectedLayer {
public:
    FullyConnectedLayer(size_t input_size, size_t output_size,
                        std::shared_ptr<Optimizer> optimizer_ptr);
    FullyConnectedLayer(const FullyConnectedLayer& other) = default;
    FullyConnectedLayer& operator=(const FullyConnectedLayer& other) = default;

    void forward(const Eigen::VectorXf& inputs,
                Eigen::VectorXf& outputs, bool learn, bool positive,
                float& threshold, const std::function<float(float)>& activation,
                const std::function<float(float)>& activation_derivative);

    void saveModel(const std::string& filepath) const;
    void loadModel(const std::string& filepath);

    size_t getInputSize() const;
    size_t getOutputSize() const;

    // Métodos para obtener y establecer pesos y sesgos
    Eigen::MatrixXf getWeights() const { return weights; }
    Eigen::VectorXf getBiases() const { return biases; }
    void setWeights(const Eigen::MatrixXf& new_weights) { weights = new_weights; }
    void setBiases(const Eigen::VectorXf& new_biases) { biases = new_biases; }

private:
    size_t input_size;
    size_t output_size;
    std::shared_ptr<Optimizer> optimizer;
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    Eigen::VectorXf pre_activations;

    void updateWeights(const Eigen::VectorXf& inputs,
                       const Eigen::VectorXf& outputs, bool is_positive,
                       float threshold,
                       const std::function<float(float)>& activation_derivative);
};

#endif // NEURAL_NETWORK_HPP
