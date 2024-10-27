// src/optimizer.cpp

#include "optimizer.hpp"

#include <cmath>
#include <stdexcept>

/* Implementación del constructor de SGDOptimizer */
SGDOptimizer::SGDOptimizer(float learning_rate, float momentum)
    : lr(learning_rate), momentum(momentum) {
    // Las matrices de velocidad se inicializarán en el primer uso
}

/* Implementación de updateWeights para SGDOptimizer */
void SGDOptimizer::updateWeights(Eigen::MatrixXf& weights,
                                 const Eigen::MatrixXf& gradients) {
    if (momentum > 0.0f) {
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

/* Implementación de updateBiases para SGDOptimizer */
void SGDOptimizer::updateBiases(Eigen::VectorXf& biases,
                                const Eigen::VectorXf& gradients) {
    if (momentum > 0.0f) {
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

/* Implementación del constructor de AdamOptimizer */
AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float epsilon)
    : lr(learning_rate), beta1(beta1), beta2(beta2), eps(epsilon),
      t_weights(0), t_biases(0),
      m_weights(), v_weights(),
      m_biases(), v_biases() {}

/* Implementación de updateWeights para AdamOptimizer */
void AdamOptimizer::updateWeights(Eigen::MatrixXf& weights,
                                  const Eigen::MatrixXf& gradients) {
    if (m_weights.size() == 0) {
        m_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        v_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
    }

    ++t_weights;
    m_weights = beta1 * m_weights + (1.0f - beta1) * gradients;
    v_weights = beta2 * v_weights + (1.0f - beta2) * gradients.array().square().matrix();
    Eigen::MatrixXf m_hat = m_weights.array() / (1.0f - std::pow(beta1, t_weights));
    Eigen::MatrixXf v_hat = v_weights.array() / (1.0f - std::pow(beta2, t_weights));

    weights.array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
}

/* Implementación de updateBiases para AdamOptimizer */
void AdamOptimizer::updateBiases(Eigen::VectorXf& biases,
                                 const Eigen::VectorXf& gradients) {
    if (m_biases.size() == 0) {
        m_biases = Eigen::VectorXf::Zero(biases.size());
        v_biases = Eigen::VectorXf::Zero(biases.size());
    }

    ++t_biases;
    m_biases = beta1 * m_biases + (1.0f - beta1) * gradients;
    v_biases = beta2 * v_biases + (1.0f - beta2) * gradients.array().square().matrix();
    Eigen::VectorXf m_hat = m_biases.array() / (1.0f - std::pow(beta1, t_biases));
    Eigen::VectorXf v_hat = v_biases.array() / (1.0f - std::pow(beta2, t_biases));

    biases.array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
}

/* Implementación del constructor de RMSPropOptimizer */
RMSPropOptimizer::RMSPropOptimizer(float learning_rate, float beta, float epsilon)
    : lr(learning_rate), beta(beta), eps(epsilon),
      s_weights(), s_biases() {}

/* Implementación de updateWeights para RMSPropOptimizer */
void RMSPropOptimizer::updateWeights(Eigen::MatrixXf& weights,
                                     const Eigen::MatrixXf& gradients) {
    if (s_weights.size() == 0) {
        s_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
    }

    s_weights = beta * s_weights + (1.0f - beta) * gradients.array().square().matrix();
    weights.array() -= lr * gradients.array() / (s_weights.array().sqrt() + eps);
}

/* Implementación de updateBiases para RMSPropOptimizer */
void RMSPropOptimizer::updateBiases(Eigen::VectorXf& biases,
                                    const Eigen::VectorXf& gradients) {
    if (s_biases.size() == 0) {
        s_biases = Eigen::VectorXf::Zero(biases.size());
    }

    s_biases = beta * s_biases + (1.0f - beta) * gradients.array().square().matrix();
    biases.array() -= lr * gradients.array() / (s_biases.array().sqrt() + eps);
}

/* Implementación del constructor de AdagradOptimizer */
AdagradOptimizer::AdagradOptimizer(float learning_rate, float epsilon)
    : lr(learning_rate), eps(epsilon),
      accumulated_grad_squared_weights(), accumulated_grad_squared_biases() {}

/* Implementación de updateWeights para AdagradOptimizer */
void AdagradOptimizer::updateWeights(Eigen::MatrixXf& weights,
                                     const Eigen::MatrixXf& gradients) {
    if (accumulated_grad_squared_weights.size() == 0) {
        accumulated_grad_squared_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
    }

    accumulated_grad_squared_weights += gradients.array().square().matrix();
    weights.array() -= (lr * gradients.array()) / (accumulated_grad_squared_weights.array().sqrt() + eps);
}

/* Implementación de updateBiases para AdagradOptimizer */
void AdagradOptimizer::updateBiases(Eigen::VectorXf& biases,
                                    const Eigen::VectorXf& gradients) {
    if (accumulated_grad_squared_biases.size() == 0) {
        accumulated_grad_squared_biases = Eigen::VectorXf::Zero(biases.size());
    }

    accumulated_grad_squared_biases += gradients.array().square().matrix();
    biases.array() -= (lr * gradients.array()) / (accumulated_grad_squared_biases.array().sqrt() + eps);
}

/* Implementación del constructor de AdadeltaOptimizer */
AdadeltaOptimizer::AdadeltaOptimizer(float rho, float epsilon)
    : rho(rho), eps(epsilon),
      accumulated_grad_squared_weights(), accumulated_update_squared_weights(),
      accumulated_grad_squared_biases(), accumulated_update_squared_biases() {}

/* Implementación de updateWeights para AdadeltaOptimizer */
void AdadeltaOptimizer::updateWeights(Eigen::MatrixXf& weights,
                                      const Eigen::MatrixXf& gradients) {
    if (accumulated_grad_squared_weights.size() == 0) {
        accumulated_grad_squared_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        accumulated_update_squared_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
    }

    accumulated_grad_squared_weights = rho * accumulated_grad_squared_weights
                                       + (1.0f - rho) * gradients.array().square().matrix();

    Eigen::MatrixXf update = (accumulated_update_squared_weights.array().sqrt() + eps)
                             / (accumulated_grad_squared_weights.array().sqrt() + eps)
                             * gradients.array();

    weights.array() -= update.array();

    accumulated_update_squared_weights = rho * accumulated_update_squared_weights
                                         + (1.0f - rho) * update.array().square().matrix();
}

/* Implementación de updateBiases para AdadeltaOptimizer */
void AdadeltaOptimizer::updateBiases(Eigen::VectorXf& biases,
                                     const Eigen::VectorXf& gradients) {
    if (accumulated_grad_squared_biases.size() == 0) {
        accumulated_grad_squared_biases = Eigen::VectorXf::Zero(biases.size());
        accumulated_update_squared_biases = Eigen::VectorXf::Zero(biases.size());
    }

    accumulated_grad_squared_biases = rho * accumulated_grad_squared_biases
                                      + (1.0f - rho) * gradients.array().square().matrix();

    Eigen::VectorXf update = (accumulated_update_squared_biases.array().sqrt() + eps)
                             / (accumulated_grad_squared_biases.array().sqrt() + eps)
                             * gradients.array();

    biases.array() -= update.array();

    accumulated_update_squared_biases = rho * accumulated_update_squared_biases
                                        + (1.0f - rho) * update.array().square().matrix();
}

/* Implementación del constructor de AdamWOptimizer */
AdamWOptimizer::AdamWOptimizer(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay)
    : lr(learning_rate), beta1(beta1), beta2(beta2), eps(epsilon), weight_decay(weight_decay),
      t_weights(0), t_biases(0),
      m_weights(), v_weights(),
      m_biases(), v_biases() {}

/* Implementación de updateWeights para AdamWOptimizer */
void AdamWOptimizer::updateWeights(Eigen::MatrixXf& weights,
                                   const Eigen::MatrixXf& gradients) {
    if (m_weights.size() == 0) {
        m_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        v_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
    }

    ++t_weights;
    m_weights = beta1 * m_weights + (1.0f - beta1) * gradients;
    v_weights = beta2 * v_weights + (1.0f - beta2) * gradients.array().square().matrix();
    Eigen::MatrixXf m_hat = m_weights.array() / (1.0f - std::pow(beta1, t_weights));
    Eigen::MatrixXf v_hat = v_weights.array() / (1.0f - std::pow(beta2, t_weights));

    // Aplicar decaimiento de peso
    weights.array() -= lr * weight_decay * weights.array();

    weights.array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
}

/* Implementación de updateBiases para AdamWOptimizer */
void AdamWOptimizer::updateBiases(Eigen::VectorXf& biases,
                                  const Eigen::VectorXf& gradients) {
    if (m_biases.size() == 0) {
        m_biases = Eigen::VectorXf::Zero(biases.size());
        v_biases = Eigen::VectorXf::Zero(biases.size());
    }

    ++t_biases;
    m_biases = beta1 * m_biases + (1.0f - beta1) * gradients;
    v_biases = beta2 * v_biases + (1.0f - beta2) * gradients.array().square().matrix();
    Eigen::VectorXf m_hat = m_biases.array() / (1.0f - std::pow(beta1, t_biases));
    Eigen::VectorXf v_hat = v_biases.array() / (1.0f - std::pow(beta2, t_biases));

    biases.array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
}

/* Implementación del constructor de LowPassFilterOptimizer */
LowPassFilterOptimizer::LowPassFilterOptimizer(float learning_rate, float alpha_val)
    : lr(learning_rate), alpha(alpha_val),
      ema_weights(), ema_biases(),
      initialized_weights(false), initialized_biases(false) {}

/* Implementación de updateWeights para LowPassFilterOptimizer */
void LowPassFilterOptimizer::updateWeights(Eigen::MatrixXf& weights,
                                           const Eigen::MatrixXf& gradients) {
    if (!initialized_weights) {
        ema_weights = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        initialized_weights = true;
    }

    // Actualizar el promedio móvil exponencial de los gradientes
    ema_weights = alpha * gradients + (1.0f - alpha) * ema_weights;

    // Actualizar los pesos usando el EMA de los gradientes
    weights.array() -= lr * ema_weights.array();
}

/* Implementación de updateBiases para LowPassFilterOptimizer */
void LowPassFilterOptimizer::updateBiases(Eigen::VectorXf& biases,
                                          const Eigen::VectorXf& gradients) {
    if (!initialized_biases) {
        ema_biases = Eigen::VectorXf::Zero(biases.size());
        initialized_biases = true;
    }

    // Actualizar el promedio móvil exponencial de los gradientes
    ema_biases = alpha * gradients + (1.0f - alpha) * ema_biases;

    // Actualizar los biases usando el EMA de los gradientes
    biases.array() -= lr * ema_biases.array();
}
