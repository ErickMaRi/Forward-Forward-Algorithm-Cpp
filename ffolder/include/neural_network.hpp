#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "optimizer.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <functional>

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
 * @brief Clase que representa una capa completamente conectada en la red neuronal,
 *        ahora con soporte para acumulación de gradientes en minibatches.
 */
class FullyConnectedLayer {
public:
    // Constructores
    FullyConnectedLayer(size_t input_size, size_t output_size,
                        std::shared_ptr<Optimizer> optimizer_ptr); // Constructor por parámetros
    FullyConnectedLayer(const FullyConnectedLayer& other) = default; // Constructor de copia
    FullyConnectedLayer& operator=(const FullyConnectedLayer& other) = default; // Asignación de copia
    FullyConnectedLayer(FullyConnectedLayer&& other) noexcept = default; // Constructor de movimiento
    FullyConnectedLayer& operator=(FullyConnectedLayer&& other) noexcept = default; // Asignación por movimiento

    /**
     * @brief Realiza la propagación hacia adelante en la capa completamente conectada.
     * 
     * @param inputs   Vector de entrada (dim: input_size).
     * @param outputs  Vector de salida (dim: output_size).
     * @param learn    Indica si se deben acumular/actualizar gradientes.
     * @param positive Indica si la muestra es “positiva” (y=1) o “negativa” (y=0).
     * @param threshold Umbral para la función de bondad.
     * @param activation Función de activación.
     * @param activation_derivative Derivada de la función de activación.
     */
    void forward(const Eigen::VectorXf& inputs,
                 Eigen::VectorXf& outputs, bool learn, bool positive,
                 float& threshold, const std::function<float(float)>& activation,
                 const std::function<float(float)>& activation_derivative);

    /**
     * @brief Guarda el modelo (pesos y bias) en un archivo binario.
     * @param filepath Ruta del archivo donde se guardará el modelo.
     */
    void saveModel(const std::string& filepath) const;

    /**
     * @brief Carga el modelo (pesos y bias) desde un archivo binario.
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
    void setBiases(const Eigen::VectorXf& new_biases)   { biases = new_biases; }

private:
    // Tamaños y optimizador
    size_t input_size;
    size_t output_size;
    std::shared_ptr<Optimizer> optimizer;

    // Parámetros entrenables
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;

    // Preactivaciones temporales
    Eigen::VectorXf pre_activations;

    // ---------- NUEVO: Soporte para minibatch ----------
    /**
     * @brief Acumuladores de gradientes para pesos y sesgos.
     */
    Eigen::MatrixXf grad_weights_accum;
    Eigen::VectorXf grad_biases_accum;

    /**
     * @brief Contador de muestras acumuladas en el minibatch.
     */
    size_t batch_count = 0;

    /**
     * @brief Tamaño de minibatch (ajustable según necesidades).
     */
    static constexpr size_t mini_batch_size = 16;

    /**
     * @brief Método interno que acumula los gradientes en buffers locales
     *        sin actualizar los pesos todavía.
     * 
     * @param inputs  Vector de entrada.
     * @param outputs Vector de salida de la capa.
     * @param is_positive Indica si la muestra es positiva.
     * @param threshold Umbral para la bondad.
     * @param activation_derivative Derivada de la función de activación.
     */
    void accumulateGradients(const Eigen::VectorXf& inputs,
                             const Eigen::VectorXf& outputs,
                             bool is_positive,
                             float threshold,
                             const std::function<float(float)>& activation_derivative);

};

extern std::function<float(float)> activation;
extern std::function<float(float)> activation_derivative;

#endif // NEURAL_NETWORK_HPP
