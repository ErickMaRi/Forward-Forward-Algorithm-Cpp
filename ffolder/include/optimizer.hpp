// include/optimizer.hpp

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <Eigen/Dense>

/**
 * @brief Clase base para optimizadores.
 */
class Optimizer {
public:
    /**
     * @brief Actualiza los pesos del modelo.
     * @param weights Matriz de pesos a actualizar.
     * @param gradients Gradientes correspondientes a los pesos.
     */
    virtual void updateWeights(Eigen::MatrixXf& weights,
                               const Eigen::MatrixXf& gradients) = 0;

    /**
     * @brief Actualiza los sesgos (biases) del modelo.
     * @param biases Vector de sesgos a actualizar.
     * @param gradients Gradientes correspondientes a los sesgos.
     */
    virtual void updateBiases(Eigen::VectorXf& biases,
                              const Eigen::VectorXf& gradients) = 0;

    virtual ~Optimizer() = default;
};

/**
 * @brief Implementación del optimizador SGD (Stochastic Gradient Descent) con momento opcional.
 */
class SGDOptimizer : public Optimizer {
public:
    /**
     * @brief Constructor del optimizador SGD.
     * @param learning_rate Tasa de aprendizaje.
     * @param momentum Factor de momento (0 para sin momento).
     */
    SGDOptimizer(float learning_rate = 0.01f, float momentum = 0.0f);

    void updateWeights(Eigen::MatrixXf& weights,
                       const Eigen::MatrixXf& gradients) override;

    void updateBiases(Eigen::VectorXf& biases,
                      const Eigen::VectorXf& gradients) override;

private:
    float lr;
    float momentum;

    Eigen::MatrixXf velocity_weights;
    Eigen::VectorXf velocity_biases;
};

/**
 * @brief Implementación del optimizador Adam.
 */
class AdamOptimizer : public Optimizer {
public:
    /**
     * @brief Constructor con parámetros personalizables.
     * @param learning_rate Tasa de aprendizaje.
     * @param beta1 Parámetro beta1 de momento.
     * @param beta2 Parámetro beta2 de momento.
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
 * @brief Implementación del optimizador RMSProp.
 */
class RMSPropOptimizer : public Optimizer {
public:
    /**
     * @brief Constructor del optimizador RMSProp.
     * @param learning_rate Tasa de aprendizaje.
     * @param beta Parámetro de decaimiento exponencial.
     * @param epsilon Término de estabilidad numérica.
     */
    RMSPropOptimizer(float learning_rate = 0.001f, float beta = 0.9f, float epsilon = 1e-8f);

    void updateWeights(Eigen::MatrixXf& weights,
                       const Eigen::MatrixXf& gradients) override;

    void updateBiases(Eigen::VectorXf& biases,
                      const Eigen::VectorXf& gradients) override;

private:
    float lr;
    float beta;
    float eps;

    Eigen::MatrixXf s_weights;
    Eigen::VectorXf s_biases;
};

/**
 * @brief Implementación del optimizador Adagrad.
 */
class AdagradOptimizer : public Optimizer {
public:
    /**
     * @brief Constructor del optimizador Adagrad.
     * @param learning_rate Tasa de aprendizaje.
     * @param epsilon Término de estabilidad numérica.
     */
    AdagradOptimizer(float learning_rate = 0.01f, float epsilon = 1e-8f);

    void updateWeights(Eigen::MatrixXf& weights,
                       const Eigen::MatrixXf& gradients) override;

    void updateBiases(Eigen::VectorXf& biases,
                      const Eigen::VectorXf& gradients) override;

private:
    float lr;
    float eps;

    Eigen::MatrixXf accumulated_grad_squared_weights;
    Eigen::VectorXf accumulated_grad_squared_biases;
};

/**
 * @brief Implementación del optimizador Adadelta.
 */
class AdadeltaOptimizer : public Optimizer {
public:
    /**
     * @brief Constructor del optimizador Adadelta.
     * @param rho Parámetro de decaimiento exponencial.
     * @param epsilon Término de estabilidad numérica.
     */
    AdadeltaOptimizer(float rho = 0.95f, float epsilon = 1e-6f);

    void updateWeights(Eigen::MatrixXf& weights,
                       const Eigen::MatrixXf& gradients) override;

    void updateBiases(Eigen::VectorXf& biases,
                      const Eigen::VectorXf& gradients) override;

private:
    float rho;
    float eps;

    Eigen::MatrixXf accumulated_grad_squared_weights;
    Eigen::MatrixXf accumulated_update_squared_weights;

    Eigen::VectorXf accumulated_grad_squared_biases;
    Eigen::VectorXf accumulated_update_squared_biases;
};

/**
 * @brief Implementación del optimizador AdamW (Adam con decaimiento de pesos).
 */
class AdamWOptimizer : public Optimizer {
public:
    /**
     * @brief Constructor del optimizador AdamW.
     * @param learning_rate Tasa de aprendizaje.
     * @param beta1 Parámetro beta1 de momento.
     * @param beta2 Parámetro beta2 de momento.
     * @param epsilon Término de estabilidad numérica.
     * @param weight_decay Factor de decaimiento de pesos.
     */
    AdamWOptimizer(float learning_rate = 0.001f, float beta1 = 0.9f,
                   float beta2 = 0.999f, float epsilon = 1e-8f, float weight_decay = 0.01f);

    void updateWeights(Eigen::MatrixXf& weights,
                       const Eigen::MatrixXf& gradients) override;

    void updateBiases(Eigen::VectorXf& biases,
                      const Eigen::VectorXf& gradients) override;

private:
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
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
