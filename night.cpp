#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>

/*
Red neuronal Forward Forward

Escrita en c++ por:
    Erick Marin Rojas
    B94544

*/

class Layer {
private:
    std::vector<double> weights; // pesos de la capa
public:
    // Constructor
    Layer(int input_size) {
        weights.resize(input_size, 0);
        randomize_weights();
    }

    std::vector<double> getWeights() const {
    return weights;
    }

    void randomize_weights() {
        for (double& weight : weights) {
            weight = static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
        }
    }

    double mean(const std::vector<std::vector<double>>& vec) {
        double sum = 0.0;
        int total_elements = 0;
        for (const auto& sub_vec : vec) {
            for (double val : sub_vec) {
                sum += val;
                ++total_elements;
            }
        }
        return sum / total_elements;
    }

    // Pase hacia adelante
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& x) {
        std::vector<std::vector<double>> output(x.size(), std::vector<double>(weights.size(), 0.0));
        for (size_t i = 0; i < x.size(); ++i) {
            for (size_t j = 0; j < weights.size(); ++j) {
                output[i][j] = weights[j] * x[i][j];
                if (output[i][j] < 0.0){
                output[i][j] = 0.0;
                }
            }
        }
        return output;
    }

    // Entrenamiento de la capa
    std::vector<double> train(const std::vector<std::vector<double>>& x_pos,
                          const std::vector<std::vector<double>>& x_neg,
                          int epochs, double lr, bool debug = true) {
    std::vector<double> best_weights = weights;
    double best_loss = std::numeric_limits<double>::infinity();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Positive forward pass
        std::vector<std::vector<double>> g_pos = forward(x_pos);
        for (auto& sub_vec : g_pos) {
            for (double& val : sub_vec) {
                val = std::pow(val, 2);
            }
        }
        double mean_g_pos = mean(g_pos);

        // Negative forward pass
        std::vector<std::vector<double>> g_neg = forward(x_neg);
        for (auto& sub_vec : g_neg) {
            for (double& val : sub_vec) {
                val = std::pow(val, 2);
            }
        }
        double mean_g_neg = mean(g_neg);

        double loss = std::log(1 + std::exp(mean_g_neg - mean_g_pos + 2));

        if (!std::isnan(loss)) {
            // Calculate gradients for each weight
            std::vector<double> gradients(weights.size(), 0.0);
            for (size_t i = 0; i < weights.size(); ++i) {
                double sum_gradient_pos = 0.0;
                for (size_t j = 0; j < x_pos.size(); ++j) {
                    sum_gradient_pos += x_pos[j][i] * loss;
                }

                double sum_gradient_neg = 0.0;
                for (size_t j = 0; j < x_neg.size(); ++j) {
                    sum_gradient_neg += x_neg[j][i] * loss;
                }

                gradients[i] = sum_gradient_pos - sum_gradient_neg;
            }

            // Update weights using gradients
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] += lr * gradients[i];
            }

            if (loss < best_loss) {
                best_loss = loss;
                best_weights = weights;
            }

            if (debug && epoch == epochs - 1) {
                std::cout << "Epoch: " << epoch << ", Loss: " << loss << ", Mean_g_pos: " << mean_g_pos << ", Mean_g_neg: " << mean_g_neg << std::endl;
                for (size_t i = 0; i < weights.size(); ++i) {
                    std::cout << "Weight[" << i << "]: " << weights[i] << ", Gradient[" << i << "]: " << gradients[i] << std::endl;
                }
            }
        } else {
            std::cout << "Stopped execution due to NaN loss" << std::endl;
            std::cout << "-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-" << std::endl;
            std::cout << "Stopped execution due to NaN loss" << std::endl;
            std::cout << "Stopped at epoch: " << epoch << std::endl;
            break;
        }
    }

    std::cout << "Best loss: " << best_loss << std::endl;
    weights = best_weights;
    return weights;
}

};

class NeuralNetwork {
private:
    std::vector<Layer> layers;  // Contenedor para las capas

    std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>>& data) {
        std::vector<std::vector<double>> normalized_data = data;  // Copia inicial
        for (std::vector<double>& vec : normalized_data) {  // Recorre cada vector
            double length = 0.0;
            for (double x : vec) {  // Calcula la longitud del vector
                length += x * x;
            }
            length = std::sqrt(length);  // Raíz cuadrada para obtener la longitud real
            if (length > 0) {
                for (double& x : vec) {  // Normaliza cada elemento
                    x /= length;
                }
            }
        }
        return normalized_data;
    }

public:
    // Añade una nueva capa a la red
    void addLayer(int input_size) {
        Layer new_layer(input_size);
        layers.push_back(new_layer);
    }

    std::vector<double> getLayerWeights(int layer_index) const {
    if (layer_index >= 0 && layer_index < layers.size()) {
      return layers[layer_index].getWeights();
    } else {
      // Handle invalid layer index (e.g., throw an exception)
    }
    }

    // Propagación hacia adelante
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input) {
        std::vector<std::vector<double>> current_output = input;
        for (Layer& layer : layers) {
            current_output = layer.forward(current_output);
            current_output = normalize(current_output);
        }
        return current_output;
    }

    // Entrenamiento de la red
    void train(const std::vector<std::vector<double>>& x_pos,
               const std::vector<std::vector<double>>& x_neg,
               int epochs, double lr, bool debug = false) {
        std::vector<std::vector<double>> current_x_pos = x_pos;
        std::vector<std::vector<double>> current_x_neg = x_neg;

        for (Layer& layer : layers) {
            layer.train(current_x_pos, current_x_neg, epochs, lr, debug);
            current_x_pos = normalize(layer.forward(current_x_pos));
            current_x_neg = normalize(layer.forward(current_x_neg));
        }
    }

    std::vector<std::vector<double>> test(const std::vector<std::vector<double>>& input) {
        std::vector<std::vector<double>> current_output = input;
        for (Layer& layer : layers) {
            current_output = layer.forward(current_output);
            current_output = normalize(current_output);
        }
        return current_output;
    }

    
};

double calculategood(const std::vector<std::vector<double>>& output, NeuralNetwork NN) {
    // Create a temporary copy of output to modify values without affecting the original
    std::vector<std::vector<double>> output_copy = output;  // Create a modifiable copy

    // Square elements in the copy
    for (auto& sub_vec : output_copy) {
        for (double& val : sub_vec) {
            val = std::pow(val, 2);  // Modify values in the copy
        }
    }

    // Calculate mean on the squared copy
    double sum = 0.0;
    int total_elements = 0;
    for (const std::vector<double>& sub_vec : output_copy) {  // Use the copy here too
        for (double valor : sub_vec) {
            sum += valor;
            ++total_elements;
        }
    }

    return sum / total_elements;
}


// Función para agregar ruido a los datos
void addNoise(std::vector<std::vector<double>>& data, double noiseLevel) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> distribution(0.0, noiseLevel);

    for (auto& row : data) {
        for (auto& value : row) {
            value += distribution(generator);
        }
    }
}

int main() {
    // Step 1: Initialize random seed for reproducibility
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Data Generation Parameters
    const int input_size = 512; // Size of the periodic signal in each data point
    int num_data_points = 1024; // Number of data points

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.95, 1.05); // Range for frequency and amplitude
    std::uniform_real_distribution<double> phase_dist(0.0, 0.1* M_PI); // Range for phase

    std::vector<std::vector<double>> x_sine(num_data_points, std::vector<double>(input_size, 0.0));
    std::vector<std::vector<double>> x_sawtooth(num_data_points, std::vector<double>(input_size, 0.0));

    // Generate data for sine wave with randomized properties
    for (int i = 0; i < num_data_points; ++i) {
        double sine_frequency = dist(gen);
        double sine_amplitude = dist(gen);
        double sine_phase = phase_dist(gen);

        for (int j = 0; j < input_size; ++j) {
            double x = j / static_cast<double>(input_size - 1); // Normalize position within the signal
            x_sine[i][j] = sine_amplitude * sin(4 * sine_frequency * x + sine_phase); // Sine value at position j
        }
    }

    // Generate data for sawtooth wave with randomized properties
    for (int i = 0; i < num_data_points; ++i) {
        double sawtooth_frequency = dist(gen);
        double sawtooth_amplitude = dist(gen);
        double sawtooth_phase = phase_dist(gen);

        for (int j = 0; j < input_size; ++j) {
            double x = j / static_cast<double>(input_size - 1); // Normalize position within the signal
            x_sawtooth[i][j] = sawtooth_amplitude * (2 * (x + sawtooth_phase - std::floor(x + sawtooth_phase + 0.5))) - sawtooth_amplitude; // Sawtooth value at position j
        }
    }
// Step 3: Initialize NeuralNetwork and add layers
    NeuralNetwork nn;
    nn.addLayer(input_size);
    nn.addLayer(input_size);
    nn.addLayer(input_size);
    int epochs = 64;
    double learning_rate = 0.001;
    double noiseLevel = 0.1;
    int num_iterations = 16;

    for (int i = 0; i < num_iterations; ++i) {
        // Train with noise-added data
        std::vector<std::vector<double>> x_pos_copy = x_sine; // Use x_sine for positive data
        std::vector<std::vector<double>> x_neg_copy = x_sawtooth; // Use x_sawtooth for negative data
        addNoise(x_pos_copy, noiseLevel);
        addNoise(x_neg_copy, noiseLevel);
        nn.train(x_pos_copy, x_neg_copy, epochs, learning_rate);

        // Print only on the last iteration
        if (i == (num_iterations - 1)) {
            std::cout << "Final training complete." << std::endl;
        }
    }

    // Create test data for sine and sawtooth waves
    std::vector<std::vector<double>> test_sine(num_data_points, std::vector<double>(input_size, 0.0));
    std::vector<std::vector<double>> test_sawtooth(num_data_points, std::vector<double>(input_size, 0.0));

    // Generate test data for sine wave
    for (int i = 0; i < num_data_points; ++i) {
        double sine_frequency = dist(gen);
        double sine_amplitude = dist(gen);
        double sine_phase = phase_dist(gen);

        for (int j = 0; j < input_size; ++j) {
            double x = j / static_cast<double>(input_size - 1); // Normalize position within the signal
            test_sine[i][j] = sine_amplitude * sin(2 * sine_frequency * x + sine_phase); // Sine value at position j
        }
    }

    // Generate test data for sawtooth wave
    for (int i = 0; i < num_data_points; ++i) {
        double sawtooth_frequency = dist(gen);
        double sawtooth_amplitude = dist(gen);
        double sawtooth_phase = phase_dist(gen);

        for (int j = 0; j < input_size; ++j) {
            double x = j / static_cast<double>(input_size - 1); // Normalize position within the signal
            test_sawtooth[i][j] = sawtooth_amplitude * (2 * (x + sawtooth_phase - std::floor(x + sawtooth_phase + 0.5))) - sawtooth_amplitude; // Sawtooth value at position j
        }
    }

    std::vector<std::vector<double>> test_output_sine = nn.test(test_sine);
    std::vector<std::vector<double>> test_output_sawtooth = nn.test(test_sawtooth);

    // Calculate MSE
    double mse_sine = calculategood(test_output_sine, nn);
    double mse_sawtooth = calculategood(test_output_sawtooth, nn);

    std::cout << "AvGood for sine wave: " << mse_sine << std::endl;
    std::cout << "AvGood for sawtooth wave: " << mse_sawtooth << std::endl;

    std::cout << "Layer 1 Weights:" << std::endl;
    for (double weight : nn.getLayerWeights(0)) {
    std::cout << std::round(weight * 100.0) / 100.0 << " ";
    }
    std::cout << std::endl << std::endl;


    std::cout << "Layer 2 Weights:" << std::endl;
    for (double weight : nn.getLayerWeights(1)) {
    std::cout << std::round(weight * 100.0) / 100.0 << " ";
    }

    std::cout << std::endl;

    std::cout << "Layer 3 Weights:" << std::endl;
    for (double weight : nn.getLayerWeights(2)) {
    std::cout << std::round(weight * 100.0) / 100.0 << " ";
    }

    std::cout << std::endl;

    return 0;
}