#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>

class Layer {
private:
  std::vector<double> weights;

public:
  Layer(int input_size) {
    weights.resize(input_size, 0.0);
    randomize_weights();
  }

  void randomize_weights(double lower_bound = -0.1, double upper_bound = 0.1) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(lower_bound, upper_bound);
    for (double& weight : weights) {
      weight = distribution(gen);
    }
  }

  std::vector<double> forward(const std::vector<double>& x) {
    std::vector<double> output(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
      for (size_t j = 0; j < weights.size(); ++j) {
        output[i] += weights[j] * x[j];
      }
      // Función de activación sigmoide
      output[i] = 1.0 / (1.0 + std::exp(-output[i]));
    }
    return output;
  }

  double calculate_goodness(const std::vector<double>& output, double desired_value) {
    double sum_squared_errors = 0.0;
    for (double activity : output) {
      double error = activity - desired_value;
      sum_squared_errors += error * error;
    }
    return sum_squared_errors / output.size();
  }

  void train(const std::vector<double>& x_pos,
             const std::vector<double>& x_neg,
             int epochs, double lr,
             double threshold = 0.5) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
      // Positive forward pass
      std::vector<double> g_pos = forward(x_pos);
      double goodness_pos = calculate_goodness(g_pos, 1.0);

      // Negative forward pass
      std::vector<double> g_neg = forward(x_neg);
      double goodness_neg = calculate_goodness(g_neg, 0.0);
  
      // Loss calculation with separate MSE and penalty
      double loss_pos = calculate_goodness(g_pos, 1.0);
      double loss_neg = calculate_goodness(g_neg, 0.0);
      double loss = loss_pos + loss_neg + std::max(0.0, goodness_neg - goodness_pos);  // Penalty for negative goodness being higher

      // Imprimir información relevante en cada iteración
      std::cout << "mean_g_neg: " << goodness_neg << " / " << "mean_g_pos: " << goodness_pos << " ." << std::endl;
      std::cout << "Loss: " << loss << std::endl;

      // Calculate gradients for each weight
      std::vector<double> gradients(weights.size(), 0.0);
      for (size_t i = 0; i < weights.size(); ++i) {
        double sum_gradient_pos = 0.0;
        for (double x : x_pos) {
          // Consider the derivative of ReLU for backpropagation
          if (g_pos[i] > 0) {
            sum_gradient_pos += x * (g_pos[i] - threshold);  // Gradient based on positive goodness
          }
        }

        double sum_gradient_neg = 0.0;
        for (double x : x_neg) {
          if (g_neg[i] > 0) {
            sum_gradient_neg += x * (threshold - g_neg[i]);  // Gradient based on negative goodness
          }
        }

        gradients[i] = sum_gradient_pos - sum_gradient_neg;
      }

      // Update weights using gradients
      for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] += lr * gradients[i];
      }
    }
  }
};

class NeuralNetwork {
private:
  std::vector<Layer> layers;

public:
  void addLayer(int input_size) {
    layers.push_back(Layer(input_size));
  }

  std::vector<double> forward(const std::vector<double>& input) {
    std::vector<double> current_output = input;
    for (Layer& layer : layers) {
      current_output = layer.forward(current_output);
      // Normalizar la salida de la capa
      double min_value = *std::min_element(current_output.begin(), current_output.end());
      double max_value = *std::max_element(current_output.begin(), current_output.end());
      for (auto& value : current_output) {
        value = (value - min_value) / (max_value - min_value);  // Normalización al rango [0, 1]
      }
    }
    return current_output;
  }

  void train(const std::vector<double>& x_pos,
             const std::vector<double>& x_neg,
             int epochs, double lr) {
    for (Layer& layer : layers) {
      layer.train(x_pos, x_neg, epochs, lr);
    }
  }
};



// Función para agregar ruido a los datos
void add_noise(std::vector<double>& data, double noiseLevel) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> distribution(0.0, noiseLevel);

  for (auto& value : data) {
    value += distribution(gen);
  }
}

  double calculate_goodness(const std::vector<double>& output) {
    double sum_squared_activities = 0.0;
    for (double activity : output) {
      sum_squared_activities += activity * activity;
    }
    return sum_squared_activities;
  }

int main() {
  std::cout << "Inicio del entrenamiento:" << std::endl;
  // Parámetros de generación de datos
  const int input_size = 256; // Tamaño de la señal periódica en cada punto de datos
  int num_data_points = 1024; // Número de puntos de datos

  // Generador de números aleatorios
  std::random_device rd;
  std::mt19937 gen(rd());

  // Distribuciones de probabilidad para generar datos
  std::uniform_real_distribution<double> dist(0.95, 1.05); // Rango para frecuencia y amplitud
  std::uniform_real_distribution<double> phase_dist(0.0, 0.1* M_PI); // Rango para fase

  // Contenedores para datos de entrenamiento
  std::vector<std::vector<double>> x_sine(num_data_points, std::vector<double>(input_size, 0.0));
  std::vector<std::vector<double>> x_sawtooth(num_data_points, std::vector<double>(input_size, 0.0));

  // Generación de datos para la señal sinusoidal
  for (int i = 0; i < num_data_points; ++i) {
    double sine_frequency = dist(gen);
    double sine_amplitude = dist(gen);
    double sine_phase = phase_dist(gen);

    for (int j = 0; j < input_size; ++j) {
      double x = j / static_cast<double>(input_size - 1); // Normalizar la posición dentro de la señal
      x_sine[i][j] = sine_amplitude * sin(4 * sine_frequency * x + sine_phase); // Valor sinusoidal en la posición j
    }
  }

  // Generación de datos para la señal de diente de sierra
  for (int i = 0; i < num_data_points; ++i) {
    double sawtooth_frequency = dist(gen);
    double sawtooth_amplitude = dist(gen);
    double sawtooth_phase = phase_dist(gen);

    for (int j = 0; j < input_size; ++j) {
      double x = j / static_cast<double>(input_size - 1); // Normalizar la posición dentro de la señal
      x_sawtooth[i][j] = sawtooth_amplitude * (2 * (x + sawtooth_phase - std::floor(x + sawtooth_phase + 0.5))) - sawtooth_amplitude; // Valor de diente de sierra en la posición j
    }
  }

  // Parámetros de entrenamiento
  int epochs = 32;
  double learning_rate = 0.00005;
  double noiseLevel = 0.01;
  int num_iterations = 1;

  // Red neuronal
  NeuralNetwork nn;
  nn.addLayer(input_size);

  // Entrenamiento con ruido
  for (int i = 0; i < num_iterations; ++i) {
    // Copias de los datos con ruido
    std::vector<std::vector<double>> x_pos_copy = x_sine;
    std::vector<std::vector<double>> x_neg_copy = x_sawtooth;
    add_noise(x_pos_copy[i], noiseLevel);
    add_noise(x_neg_copy[i], noiseLevel);

    // Entrenamiento de la red
    nn.train(x_pos_copy[i], x_neg_copy[i], epochs, learning_rate);
    
    // Imprimir solo en la última iteración
    if (i == (num_iterations - 1)) {
      std::cout << "Finalización del entrenamiento." << std::endl;
    }
  }

  // Generación de datos de prueba
  std::vector<std::vector<double>> test_sine(num_data_points, std::vector<double>(input_size, 0.0));
  std::vector<std::vector<double>> test_sawtooth(num_data_points, std::vector<double>(input_size, 0.0));

  // Generación de datos de prueba para la señal sinusoidal
for (int i = 0; i < num_data_points; ++i) {
  double sine_frequency = dist(gen);
  double sine_amplitude = dist(gen);
  double sine_phase = phase_dist(gen);

  for (int j = 0; j < input_size; ++j) {
    double x = j / static_cast<double>(input_size - 1); // Normalizar la posición dentro de la señal
    test_sine[i][j] = sine_amplitude * sin(4 * sine_frequency * x + sine_phase); // Valor sinusoidal en la posición j
  }
}

// Generación de datos de prueba para la señal de diente de sierra
for (int i = 0; i < num_data_points; ++i) {
  double sawtooth_frequency = dist(gen);
  double sawtooth_amplitude = dist(gen);
  double sawtooth_phase = phase_dist(gen);

  for (int j = 0; j < input_size; ++j) {
    double x = j / static_cast<double>(input_size - 1); // Normalizar la posición dentro de la señal
    test_sawtooth[i][j] = sawtooth_amplitude * (2 * (x + sawtooth_phase - std::floor(x + sawtooth_phase + 0.5))) - sawtooth_amplitude; // Valor de diente de sierra en la posición j
  }
}

// Evaluación del rendimiento en datos de prueba
std::vector<double> output_sine = nn.forward(test_sine[0]);
std::vector<double> output_sawtooth = nn.forward(test_sawtooth[0]);

double goodness_sine = calculate_goodness(output_sine);
double goodness_sawtooth = calculate_goodness(output_sawtooth);

// Imprimir resultados
std::cout << "Bondad de la señal sinusoidal: " << goodness_sine << std::endl;
std::cout << "Bondad de la señal de diente de sierra: " << goodness_sawtooth << std::endl;

return 0;
}