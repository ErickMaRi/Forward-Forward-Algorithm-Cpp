// include/optimizer.hpp

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <Eigen/Dense>

/**
 * @brief Clase base para optimizadores.
 */
class Optimizer {
public:
    virtual void updateWeights(Eigen::MatrixXf& weights,
                               const Eigen::MatrixXf& gradients) = 0;

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
                  float beta2 = 0.999f, float epsilon = 1e-8f);

    void updateWeights(Eigen::MatrixXf& weights,
                       const Eigen::MatrixXf& gradients) override;

    void updateBiases(Eigen::VectorXf& biases,
                      const Eigen::VectorXf& gradients) override;

private:
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t_weights;
    int t_biases;

    Eigen::MatrixXf m_weights;
    Eigen::MatrixXf v_weights;
    Eigen::VectorXf m_biases;
    Eigen::VectorXf v_biases;
};

/**
 * @brief Implementación del optimizador Low Pass Filter.
 */
class LowPassFilterOptimizer : public Optimizer {
public:
    /**
     * @brief Constructor con parámetros personalizables.
     * @param learning_rate Tasa de aprendizaje.
     * @param alpha Factor de suavizado para el filtro.
     */
    LowPassFilterOptimizer(float learning_rate = 0.01f, float alpha = 0.1f);

    void updateWeights(Eigen::MatrixXf& weights,
                       const Eigen::MatrixXf& gradients) override;

    void updateBiases(Eigen::VectorXf& biases,
                      const Eigen::VectorXf& gradients) override;

private:
    float lr;
    float alpha;

    Eigen::MatrixXf ema_weights;
    Eigen::VectorXf ema_biases;
    bool initialized_weights;
    bool initialized_biases;
};

#endif // OPTIMIZER_HPP
