#include "optimizer.hpp"
#include <cmath>
#include <stdexcept>

/* ===============================================================
 *        1) SGDOptimizer (con momento opcional)
 * ===============================================================*/
SGDOptimizer::SGDOptimizer(float learning_rate, float momentum)
    : lr(learning_rate), momentum(momentum) {
    // Se inicializarán velocity_weights y velocity_biases en la primera llamada
}

void SGDOptimizer::updateWeights(Eigen::MatrixXf& weights,
                                 const Eigen::MatrixXf& gradients) {
    if (momentum > 0.0f) {
        // Inicializar si es la primera vez
        if (velocity_weights.size() == 0) {
            velocity_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        }
        // Actualizar la velocidad
        velocity_weights = momentum * velocity_weights - lr * gradients;
        // Actualizar los pesos
        weights += velocity_weights;
    } else {
        // Actualizar los pesos directamente
        weights -= lr * gradients;
    }
}

void SGDOptimizer::updateBiases(Eigen::VectorXf& biases,
                                const Eigen::VectorXf& gradients) {
    if (momentum > 0.0f) {
        // Inicializar si es la primera vez
        if (velocity_biases.size() == 0) {
            velocity_biases = Eigen::VectorXf::Zero(biases.size());
        }
        // Actualizar la velocidad
        velocity_biases = momentum * velocity_biases - lr * gradients;
        // Actualizar los biases
        biases += velocity_biases;
    } else {
        // Actualizar los biases directamente
        biases -= lr * gradients;
    }
}

/* ===============================================================
 *        2) AdamOptimizer (clásico)
 * ===============================================================*/
AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float epsilon)
    : lr(learning_rate),
      beta1(beta1),
      beta2(beta2),
      eps(epsilon),
      t_weights(0),
      t_biases(0) {
}

void AdamOptimizer::updateWeights(Eigen::MatrixXf& weights,
                                  const Eigen::MatrixXf& gradients) {
    // Inicialización diferida
    if (m_weights.size() == 0) {
        m_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        v_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
    }

    // Aumentar contador
    ++t_weights;

    // Actualizar m y v
    m_weights = beta1 * m_weights + (1.0f - beta1) * gradients;
    v_weights = beta2 * v_weights + (1.0f - beta2) * gradients.array().square().matrix();

    // Corrección de sesgo (biased-correction)
    Eigen::MatrixXf m_hat = m_weights.array() / (1.0f - std::pow(beta1, t_weights));
    Eigen::MatrixXf v_hat = v_weights.array() / (1.0f - std::pow(beta2, t_weights));

    // Actualizar los pesos
    weights.array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
}

void AdamOptimizer::updateBiases(Eigen::VectorXf& biases,
                                 const Eigen::VectorXf& gradients) {
    // Inicialización diferida
    if (m_biases.size() == 0) {
        m_biases = Eigen::VectorXf::Zero(biases.size());
        v_biases = Eigen::VectorXf::Zero(biases.size());
    }

    // Aumentar contador
    ++t_biases;

    // Actualizar m y v
    m_biases = beta1 * m_biases + (1.0f - beta1) * gradients;
    v_biases = beta2 * v_biases + (1.0f - beta2) * gradients.array().square().matrix();

    // Corrección de sesgo (biased-correction)
    Eigen::VectorXf m_hat = m_biases.array() / (1.0f - std::pow(beta1, t_biases));
    Eigen::VectorXf v_hat = v_biases.array() / (1.0f - std::pow(beta2, t_biases));

    // Actualizar biases
    biases.array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
}

/* ===============================================================
 *        3) LowPassFilterOptimizer (out of the box)
 * ===============================================================*/
LowPassFilterOptimizer::LowPassFilterOptimizer(float learning_rate, float alpha_val)
    : lr(learning_rate),
      alpha(alpha_val),
      initialized_weights(false),
      initialized_biases(false) {
}

void LowPassFilterOptimizer::updateWeights(Eigen::MatrixXf& weights,
                                           const Eigen::MatrixXf& gradients) {
    // Inicializar la primera vez
    if (!initialized_weights) {
        ema_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        initialized_weights = true;
    }

    // Calcular EMA (Exponential Moving Average) de los gradientes
    ema_weights = alpha * gradients + (1.0f - alpha) * ema_weights;

    // Actualizar los pesos usando el EMA de gradientes
    weights.array() -= lr * ema_weights.array();
}

void LowPassFilterOptimizer::updateBiases(Eigen::VectorXf& biases,
                                          const Eigen::VectorXf& gradients) {
    // Inicializar la primera vez
    if (!initialized_biases) {
        ema_biases = Eigen::VectorXf::Zero(biases.size());
        initialized_biases = true;
    }

    // Calcular EMA de los gradientes
    ema_biases = alpha * gradients + (1.0f - alpha) * ema_biases;

    // Actualizar los biases
    biases.array() -= lr * ema_biases.array();
}

/* ===============================================================
 *        4) AdaBeliefOptimizer (basado en Adam)
 * ===============================================================*/
AdaBeliefOptimizer::AdaBeliefOptimizer(float learning_rate,
                                       float beta1,
                                       float beta2,
                                       float epsilon)
    : lr(learning_rate),
      beta1(beta1),
      beta2(beta2),
      eps(epsilon),
      t_weights(0),
      t_biases(0) {
}

void AdaBeliefOptimizer::updateWeights(Eigen::MatrixXf& weights,
                                       const Eigen::MatrixXf& gradients) {
    if (m_weights.size() == 0) {
        m_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        v_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
    }
    ++t_weights;

    // Actualizamos m
    m_weights = beta1 * m_weights + (1.0f - beta1) * gradients;

    // En AdaBelief, se calcula la diferencia entre gradiente y su promedio m_weights
    // y se acumula esa diferencia al cuadrado en v_weights
    Eigen::MatrixXf diff = gradients - m_weights;
    v_weights = beta2 * v_weights + (1.0f - beta2) * diff.array().square().matrix();

    // Corrección de sesgo
    Eigen::MatrixXf m_hat = m_weights.array() / (1.0f - std::pow(beta1, t_weights));
    Eigen::MatrixXf v_hat = v_weights.array() / (1.0f - std::pow(beta2, t_weights));

    // Actualizar pesos
    weights.array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
}

void AdaBeliefOptimizer::updateBiases(Eigen::VectorXf& biases,
                                      const Eigen::VectorXf& gradients) {
    if (m_biases.size() == 0) {
        m_biases = Eigen::VectorXf::Zero(biases.size());
        v_biases = Eigen::VectorXf::Zero(biases.size());
    }
    ++t_biases;

    // Actualizamos m
    m_biases = beta1 * m_biases + (1.0f - beta1) * gradients;

    // Diferencia para las biases
    Eigen::VectorXf diff = gradients - m_biases;
    v_biases = beta2 * v_biases + (1.0f - beta2) * diff.array().square().matrix();

    // Corrección de sesgo
    Eigen::VectorXf m_hat = m_biases.array() / (1.0f - std::pow(beta1, t_biases));
    Eigen::VectorXf v_hat = v_biases.array() / (1.0f - std::pow(beta2, t_biases));

    // Actualizar biases
    biases.array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
}
