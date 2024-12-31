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

/* ------------------------------------------------------------------
 *  1) SGDOptimizer (con momento opcional)
 * ------------------------------------------------------------------ */
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

    Eigen::MatrixXf velocity_weights;  // Acumulado de momento para los pesos
    Eigen::VectorXf velocity_biases;   // Acumulado de momento para los biases
};

/* ------------------------------------------------------------------
 *  2) AdamOptimizer (clásico)
 * ------------------------------------------------------------------ */
class AdamOptimizer : public Optimizer {
public:
    /**
     * @brief Constructor con parámetros personalizables.
     * @param learning_rate Tasa de aprendizaje.
     * @param beta1 Parámetro beta1 de momento.
     * @param beta2 Parámetro beta2 de momento.
     * @param epsilon Término de estabilidad numérica.
     */
    AdamOptimizer(float learning_rate = 0.001f,
                  float beta1 = 0.9f,
                  float beta2 = 0.999f,
                  float epsilon = 1e-8f);

    void updateWeights(Eigen::MatrixXf& weights,
                       const Eigen::MatrixXf& gradients) override;

    void updateBiases(Eigen::VectorXf& biases,
                      const Eigen::VectorXf& gradients) override;

private:
    float lr;
    float beta1;
    float beta2;
    float eps;

    int t_weights;  // Contador de actualizaciones para los pesos
    int t_biases;   // Contador de actualizaciones para los biases

    Eigen::MatrixXf m_weights;  // Promedio de primer momento para pesos
    Eigen::MatrixXf v_weights;  // Promedio de segundo momento para pesos
    Eigen::VectorXf m_biases;   // Promedio de primer momento para biases
    Eigen::VectorXf v_biases;   // Promedio de segundo momento para biases
};

/* ------------------------------------------------------------------
 *  3) LowPassFilterOptimizer (out of the box)
 * ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------
 *  4) AdaBeliefOptimizer (otro distinto, basado en Adam)
 * ------------------------------------------------------------------ */
class AdaBeliefOptimizer : public Optimizer {
public:
    /**
     * @brief Constructor del optimizador AdaBelief.
     * @param learning_rate Tasa de aprendizaje.
     * @param beta1 Parámetro beta1.
     * @param beta2 Parámetro beta2.
     * @param epsilon Término de estabilidad numérica.
     */
    AdaBeliefOptimizer(float learning_rate = 0.001f,
                       float beta1 = 0.9f,
                       float beta2 = 0.999f,
                       float epsilon = 1e-8f);

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

    // Acumulados para primer y segundo momento
    Eigen::MatrixXf m_weights;
    Eigen::MatrixXf v_weights;
    Eigen::VectorXf m_biases;
    Eigen::VectorXf v_biases;
};

#endif // OPTIMIZER_HPP
