#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "optimizer.hpp"

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>

/**
 * @brief Clase para manejar conjuntos de datos de imágenes.
 */
class Dataset {
public:
    // Constructores
    Dataset() = default;
    explicit Dataset(const std::string& directory_path);
    Dataset(const Dataset& other) = default;
    Dataset& operator=(const Dataset& other) = default;
    Dataset(Dataset&& other) noexcept = default;
    Dataset& operator=(Dataset&& other) noexcept = default;

    // Métodos de acceso
    size_t getInputSize() const;
    size_t getNumSamples() const;
    const Eigen::VectorXf& getSample(size_t index) const;
    const std::string& getImagePath(size_t index) const;
    
    // Métodos de manipulación
    void shuffle();
    void addSample(const Eigen::VectorXf& sample, const std::string& image_path);
    void setImageProperties(size_t height, size_t width, size_t channels);
    
    // Métodos para obtener propiedades de las imágenes
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
    // Constructores
    FullyConnectedLayer(size_t input_size, size_t output_size,
                        std::shared_ptr<Optimizer> optimizer_ptr); // Constructor por parámetros
    FullyConnectedLayer(const FullyConnectedLayer& other) = default; // Constructor de copia
    FullyConnectedLayer& operator=(const FullyConnectedLayer& other) = default; // Constructor asignador
    FullyConnectedLayer(FullyConnectedLayer&& other) noexcept = default; // Constructor de movimiento
    FullyConnectedLayer& operator=(FullyConnectedLayer&& other) noexcept = default; // Asignación por movimiento

    /**
     * @brief Realiza la propagación hacia adelante en la capa completamente conectada.
     * 
     * @param inputs Vector de entrada.
     * @param outputs Vector de salida.
     * @param learn Indica si se debe actualizar los pesos.
     * @param positive Indica si la muestra es positiva.
     * @param threshold Umbral para la bondad.
     * @param activation Función de activación.
     * @param activation_derivative Derivada de la función de activación.
     */
    void forward(const Eigen::VectorXf& inputs,
                Eigen::VectorXf& outputs, bool learn, bool positive,
                float& threshold, const std::function<float(float)>& activation,
                const std::function<float(float)>& activation_derivative);

    /**
     * @brief Guarda el modelo en un archivo binario.
     * 
     * @param filepath Ruta del archivo donde se guardará el modelo.
     */
    void saveModel(const std::string& filepath) const;

    /**
     * @brief Carga el modelo desde un archivo binario.
     * 
     * @param filepath Ruta del archivo desde donde se cargará el modelo.
     */
    void loadModel(const std::string& filepath);

    // Métodos de acceso
    size_t getInputSize() const;
    size_t getOutputSize() const;

    // Métodos para obtener y establecer pesos y sesgos
    const Eigen::MatrixXf& getWeights() const { return weights; }
    const Eigen::VectorXf& getBiases() const { return biases; }
    void setWeights(const Eigen::MatrixXf& new_weights) { weights = new_weights; }
    void setBiases(const Eigen::VectorXf& new_biases) { biases = new_biases; }

private:
    size_t input_size;
    size_t output_size;
    std::shared_ptr<Optimizer> optimizer;
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    Eigen::VectorXf pre_activations;

    /**
     * @brief Actualiza los pesos y sesgos de la capa.
     * 
     * @param inputs Vector de entrada.
     * @param outputs Vector de salida.
     * @param is_positive Indica si la muestra es positiva.
     * @param threshold Umbral para la bondad.
     * @param activation_derivative Derivada de la función de activación.
     */
    void updateWeights(const Eigen::VectorXf& inputs,
                       const Eigen::VectorXf& outputs, bool is_positive,
                       float threshold,
                       const std::function<float(float)>& activation_derivative);
};

#endif // NEURAL_NETWORK_HPP
