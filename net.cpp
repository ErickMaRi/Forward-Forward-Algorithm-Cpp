#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include "mnist.h"

#define USE_MNIST_LOADER

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

    std::vector<double> getWeights(){
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
               int epochs, double lr, bool debug = false) {
        std::vector<double> bestweighs;
        double bestloss = 1000;

        for (int epoch = 0; epoch < epochs; ++epoch) {

            // Pase hacia adelante positivo
            std::vector<std::vector<double>> g_pos = forward(x_pos);
            for (auto& sub_vec : g_pos) {
                for (double& val : sub_vec) {
                    val = std::pow(val, 2);
                }
            }
            double mean_g_pos = mean(g_pos);

            // Pase hacia adelante negativo
            std::vector<std::vector<double>> g_neg = forward(x_neg);
            for (auto& sub_vec : g_neg) {
                for (double& val : sub_vec) {
                    val = std::pow(val, 2);
                }
            }
            double mean_g_neg = mean(g_neg);

            double loss = std::log(1 + std::exp(mean_g_neg - mean_g_pos));

            if(std::isnan(loss)){
            std::cout << "Se detuvo la ejecucion por loss NaN" << std::endl;
            std::cout << "-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-" << std::endl;
            std::cout << "Se detuvo la ejecucion por loss NaN" << std::endl;
            std::cout << "Nos detuvimos en el epoch:" << epoch << std::endl;
            break;
            }
            // Calculamos los gradientes individuales para cada peso
            std::vector<double> gradients(weights.size(), 0.0);
            for (size_t i = 0; i < weights.size(); ++i) {
                double sum_gradient_pos = 0;
                for (size_t j = 0; j < x_pos.size(); ++j) {
                    sum_gradient_pos += x_pos[j][i] * loss;  // derivada parcial con respecto a weights[i] para datos positivos
                }

                double sum_gradient_neg = 0;
                for (size_t j = 0; j < x_neg.size(); ++j) {
                    sum_gradient_neg += x_neg[j][i] * loss;  // derivada parcial con respecto a weights[i] para datos negativos
                }

                gradients[i] = sum_gradient_pos - sum_gradient_neg;
            }

            // Ahora, actualizamos cada peso usando su gradiente individual
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] += lr * gradients[i];
            }

            if(loss<bestloss){
                bestloss = loss;
                bestweighs = weights;
            }

            if(debug and (epoch == epochs - 1)){
                std::cout << "Epoch: " << epoch << ", Loss: " << loss << ", Mean_g_pos: " << mean_g_pos << ", Mean_g_neg: " << mean_g_neg << std::endl;
                for (size_t i = 0; i < weights.size(); ++i) {
                    std::cout << "Weight[" << i << "]: " << weights[i] << ", Gradient[" << i << "]: " << gradients[i] << std::endl;
                }
            }
        }
        std::cout << "Best loss: " << bestloss << std::endl;
        weights = bestweighs;
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
};

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

    // Load MNIST Data
    load_mnist();

    std::vector<std::vector<double>> x_pos_original;
    for(int i = 0; i < NUM_TRAIN; ++i) {
        std::vector<double> temp(train_image[i], train_image[i] + SIZE);
        x_pos_original.push_back(temp);
    }


    std::vector<std::vector<double>> x_neg_original;

    for(int i = 0; i < NUM_TRAIN; ++i) { // NUM_TRAIN is the size of your positive dataset
        // Step 1: Select an example
        auto base_image = getRandomMNISTImage();

        // Step 2: Blurring and Thresholding
        auto mask = createMaskWithBlurAndThreshold(base_image);

        // Step 3: Select two more data points
        auto image1 = getRandomMNISTImage();
        auto image2 = getRandomMNISTImage();

        // Step 4: Data Mixing
        auto negative_data_point = mixImagesUsingMask(image1, image2, mask);

        // Step 5: Add to negative dataset
        x_neg_original.push_back(negative_data_point);
    }

    // Step 3: Initialize a NeuralNetwork object and add layers
    NeuralNetwork nn;
    nn.addLayer(120);  // Assuming all data vectors have the same size
    nn.addLayer(120);


    int epochs = 16;  // Or whatever number of epochs you prefer
    double learning_rate = 0.001;  // Or your preferred learning rate
    // Agregar ruido a los datos
    double noiseLevel = 0.01; // Ajusta el nivel de ruido según tus necesidades

    int num_iterations = 256;
    bool debug = false;

    for (int i = 0; i < num_iterations; ++i) {
        // Copiamos los datos originales
        std::vector<std::vector<double>> x_pos = x_pos_original;
        std::vector<std::vector<double>> x_neg = x_neg_original;

        // Aplicamos ruido y entrenamos
        addNoise(x_pos, noiseLevel);
        addNoise(x_neg, noiseLevel);
        if (i == (num_iterations - 1)){
            debug = true;
        }
        nn.train(x_pos, x_neg, epochs, learning_rate, debug);

        // Limpiamos la memoria para esta iteración
        x_pos.clear();
        x_neg.clear();

        std::cout << "-------------------------------------------------" << std::endl;
    }

    return 0;
}
