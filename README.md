# Forward-Forward en C++

Este documento explica la implementación del **Algoritmo Forward-Forward** introducido por Geoffrey Hinton aquí presente. El objetivo es ofrecer una comprensión matemática y conceptual completa de cada componente del código, facilitando así su estudio y aplicación.

## Árbol de carpetas y archivos sin los datos

```
F-F_Cpp$ tree
.
├── ffolder
│   ├── debugtest
│   ├── debugtest.cpp
│   ├── FastNoiseLite.h
│   ├── ff
│   ├── forward_forward.cpp
│   ├── gifs
│   │   ├── 3neur2reset.gif
│   │   ├── 3neurnoresetdyn.gif
│   │   ├── 3neurnoreset.gif
│   │   ├── outputsingleneur2reset.gif
│   │   ├── outputsingleneuronnoreset.gif
│   │   └── singlenoresetdyn.gif
│   ├── histograms
│   │   └── hacer gif con los png de la carpeta.txt
│   ├── Imagenet64_train_part1
│   │   ├── extraer datos.py
│   │   └── train_data_batch_1
│   ├── justnoise
│   ├── justnoise.cpp
│   ├── Makefile
│   ├── negative_images
│   ├── positive_images
│   ├── README.md
│   └── singleneuron2resetdyn.gif
└── README.md
```

1. `gifs` contiene algunos resultados para redes pequeñas.
2. `histograms` contiene las dos distribuciones de bondades para los conjuntos de datos, por época.
3. `Imagenet64_train_part1` contiene un segmento de los conjuntos de datos para imagenet de 64 píxeles.
4. `negative_images` contiene los PNGs de 64x64 píxeles a color del conjunto de datos sintetizado.
5. `positive_images` contiene los PNGs de 64x64 píxeles a color del conjunto de datos a estudiar.

## Tabla de Contenidos

1. [Algoritmo Forward-Forward](#algoritmo-forward-forward)
   - [Makefile](#makefile)
   - [Archivos Principales](#archivos-principales)
2. [Detalles del Código](#detalles-del-código)
   - [Bibliotecas y Dependencias](#bibliotecas-y-dependencias)
   - [Clase `Dataset`](#clase-dataset)
   - [Clase `Optimizer`](#clase-optimizer)
     - [Clase `AdamOptimizer`](#clase-adamoptimizer)
   - [Clase `FullyConnectedLayer`](#clase-fullyconnectedlayer)
   - [Funciones Auxiliares](#funciones-auxiliares)
     - [División del Conjunto de Datos](#división-del-conjunto-de-datos)
     - [Selección del Optimizador](#selección-del-optimizador)
   - [Funciones de Visualización](#funciones-de-visualización)
     - [Visualización PCA](#visualización-pca)
     - [Histogramas de Bondad](#histogramas-de-bondad)
   - [Funciones de Entrenamiento y Evaluación](#funciones-de-entrenamiento-y-evaluación)
   - [Función Principal (`main`)](#función-principal-main)
3. [Explicación Matemática y Conceptual](#explicación-matemática-y-conceptual)
   - [Conceptos Básicos de Redes Neuronales](#conceptos-básicos-de-redes-neuronales)
   - [Forward-Forward vs Backpropagation](#forward-forward-vs-backpropagation)
   - [Funciones de Activación y Derivadas](#funciones-de-activación-y-derivadas)
   - [Optimización con Adam](#optimización-con-adam)
   - [Métricas de Evaluación](#métricas-de-evaluación)
4. [Referencias](#referencias)

---

## Algoritmo Forward-Forward

El **Algoritmo Forward-Forward** es una alternativa al método de **backpropagation** tradicionalmente utilizado para entrenar redes neuronales profundas. Introducido por Geoffrey Hinton, este algoritmo reemplaza las pasadas hacia adelante y hacia atrás de backpropagation por dos pasadas hacia adelante: una con datos positivos (reales) y otra con datos negativos (generados por la red misma o suministrados externamente).

Cada capa de la red tiene su propia función objetivo que simplemente busca maximizar una medida de "bondad" para los datos positivos y minimizarla para los datos negativos. Esta aproximación tiene el potencial de ser más biológicamente plausible y de funcionar de manera eficiente en hardware analógico de bajo consumo. Además de crear una asimetría entre la magnitud de las salidas para un conjunto vs otro útil para aplicar otras técnicas de análisis de datos.

### Makefile

El `Makefile` facilita la compilación de los diferentes ejecutables (3):

```makefile
CXX = g++

# Detectar automáticamente si se necesita -lstdc++fs basado en la versión de GCC
GCC_VERSION := $(shell $(CXX) -dumpversion | cut -d. -f1)
ifeq ($(shell [ $(GCC_VERSION) -ge 9 ] && echo yes),yes)
    STD = -std=c++17
    FS_LIB =
else
    STD = -std=c++17
    FS_LIB = -lstdc++fs
endif

CXXFLAGS = $(STD) -Ofast -march=native -flto -fopenmp -ffast-math -funroll-loops -fno-math-errno `pkg-config --cflags opencv4`
LDFLAGS = -flto -fopenmp `pkg-config --libs opencv4` $(FS_LIB)

# Definir los ejecutables
TARGETS = ff debugtest justnoise

all: $(TARGETS)

# Regla para compilar 'ff'
ff: forward_forward.cpp
	$(CXX) $(CXXFLAGS) -o ff forward_forward.cpp $(LDFLAGS) -I /usr/include/eigen3

# Regla para compilar 'debugtest'
debugtest: debugtest.cpp
	$(CXX) $(CXXFLAGS) -o debugtest debugtest.cpp $(LDFLAGS) -I /usr/include/eigen3

# Regla para compilar 'justnoise'
justnoise: justnoise.cpp
	$(CXX) $(CXXFLAGS) -o justnoise justnoise.cpp $(LDFLAGS) -I /usr/include/eigen3

clean:
	rm -f $(TARGETS)

.PHONY: all clean
```

Si ya tenemos los datos en las carpetas positivas y negativas podemos proceder directamente a ejecutar `./ff` para entrenar el modelo deseado, de otra forma podemos usar `justnoise` o `debugtest` para generar los datos positivos y/o negativos.

#### Explicación de Componentes Clave:

- **Variables de Compilación:**
  - `CXX`: Define el compilador a utilizar (`g++`).
  - `GCC_VERSION`: Obtiene la versión de GCC instalada para determinar si se requiere la biblioteca `-lstdc++fs`.
  - `STD`: Establece el estándar de C++ (`-std=c++17`).
  - `FS_LIB`: Biblioteca del sistema de archivos, necesaria para versiones de GCC menores a 9.
  - `CXXFLAGS`: Flags de compilación optimizadas para rendimiento, incluyendo optimizaciones de velocidad y soporte para OpenMP y OpenCV.
  - `LDFLAGS`: Flags de enlace, incluyendo optimizaciones y bibliotecas necesarias.

- **Reglas de Compilación:**
  - `all`: Compila todos los ejecutables definidos en `TARGETS`.
  - `ff`, `debugtest`, `justnoise`: Las tres reglas sirven para construir los datos y entrenar la red.

- **Regla de Limpieza:**
  - `clean`: Elimina los ejecutables compilados.

### Archivos Principales

El proyecto contiene principalmente tres ejecutables:

1. **`ff` (`forward_forward.cpp`):** Construye el ejecutable para entrenar, evaluar y visualizar una red.
2. **`debugtest` (`debugtest.cpp`):** Usa Gaussian Splatting para producir un conjunto de datos complejo de juguete de imágenes de 64x64 píxeles de un color aleatorio con manchones de colores, también produce el conjunto negativo a partir del positivo, mezclando dos datos por canal con tres máscaras aleatorias.
3. **`justnoise` (`justnoise.cpp`):** Únicamente produce el conjunto de datos negativo a partir de los contenidos de la carpeta de datos negativos.

## `forward_forward.cpp`

### Clase `Dataset`

#### Carga y Preprocesamiento de Imágenes

La clase `Dataset` se encarga de cargar imágenes desde un directorio, normalizarlas y convertirlas en vectores de características para su posterior uso en el entrenamiento de la red neuronal.

1. **Normalización de Píxeles:**

   ```cpp
   img.convertTo(img, CV_32F, 1.0 / 255.0); // Normaliza los píxeles
   ```

   - **Es decir:** Cada píxel de la imagen, originalmente representado por un valor entero en el rango [0, 255], se escala al rango [0, 1] dividiendo por 255.

     $$
     \text{normalizado} = \frac{\text{original}}{255.0}
     $$

2. **Aplanamiento de la Imagen:**

   ```cpp
   Eigen::VectorXf sample(img.rows * img.cols * img.channels());
   int idx = 0;
   for (int i = 0; i < img.rows; ++i) {
       for (int j = 0; j < img.cols; ++j) {
           cv::Vec3f pixel = img.at<cv::Vec3f>(i, j);
           for (int c = 0; c < img.channels(); ++c) {
               sample(idx++) = pixel[c];
           }
       }
   }
   ```

   - **Es decir:** La imagen 2D con múltiples canales (por ejemplo, RGB) se convierte en un vector de una sola dimensión concatenando los valores de los píxeles.

3. **Validación de Tamaños Consistentes:**

   ```cpp
   if (img.rows != image_height || img.cols != image_width) {
       throw std::runtime_error("Las imágenes deben tener el mismo tamaño.");
   }
   ```

   - **Es decir:** Todas las muestras deben tener el mismo tamaño $ d = \text{rows} \times \text{cols} \times \text{channels} $ para que la red neuronal pueda procesarlas de manera consistente.

### Clase `Optimizer` y `AdamOptimizer`

#### Optimización de Parámetros

El optimizador es responsable de actualizar los pesos y sesgos de las capas de la red neuronal en función de los gradientes calculados durante el entrenamiento.

1. **Algoritmo Adam:**

   ```cpp
   m_weights = beta1 * m_weights + (1.0f - beta1) * gradients;
   v_weights = beta2 * v_weights + (1.0f - beta2) * gradients.array().square().matrix();
   Eigen::MatrixXf m_hat = m_weights.array() / (1.0f - std::pow(beta1, t_weights));
   Eigen::MatrixXf v_hat = v_weights.array() / (1.0f - std::pow(beta2, t_weights));

   weights.array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
   ```

   - **Es decir:**

     El optimizador Adam actualiza los parámetros $ \theta $ (pesos o sesgos) utilizando las siguientes ecuaciones:

     $$
     m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
     $$
     $$
     v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
     $$
     $$
     \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
     $$
     $$
     \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
     $$
     $$
     \theta = \theta - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
     $$

     Donde:
     - $ g_t $ es el gradiente en el tiempo $ t $.
     - $ \beta_1 $ y $ \beta_2 $ son los coeficientes de decaimiento para los momentos.
     - $ \alpha $ es la tasa de aprendizaje.
     - $ \epsilon $ es un término de estabilidad para evitar divisiones por cero.
     - $ \hat{m}_t $ y $ \hat{v}_t $ son las estimaciones corregidas de sesgo para el primer y segundo momento, respectivamente.

2. **Actualización de Pesos y Sesgos:**

   El optimizador utiliza las estimaciones $ \hat{m}_t $ y $ \hat{v}_t $ para actualizar los pesos y sesgos mediante el optimizador seleccionado (por ejemplo, Adam).

   ```cpp
   optimizer->updateWeights(weights, grad_weights);
   optimizer->updateBiases(biases, grad_biases);
   ```

   - **Es decir:** La actualización sigue las reglas definidas por el optimizador Adam, como se explicó anteriormente.

### Clase `FullyConnectedLayer`

#### Inicialización de Pesos y Sesgos

1. **Inicialización de He:**

   ```cpp
   float std_dev = std::sqrt(2.0f / input_size); // Inicialización He
   std::normal_distribution<float> weight_dist(0.0f, std_dev);
   weights = Eigen::MatrixXf::NullaryExpr(output_size, input_size, [&]() { return weight_dist(gen); });
   biases = Eigen::VectorXf::Constant(output_size, 0.01f); // Inicializa sesgos pequeños
   ```

   - **Es decir:** Los pesos se inicializan con una distribución normal centrada en 0 con una desviación estándar de $ \sqrt{\frac{2}{n_{\text{entrada}}}} $, donde $ n_{\text{entrada}} $ es el número de entradas a la capa. Esta inicialización, conocida como **Inicialización de He**, ayuda a mantener la varianza de las activaciones a través de las capas, facilitando el entrenamiento de redes profundas. No se ha experimentado si debemos buscar distribuciones probabilísticas específicas para la inicialización de redes F-F, pues aprenden capa a capa.

     $$
     W_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{entrada}}}}\right)
     $$
     $$
     b_i = 0.01
     $$

#### Pasada Hacia Adelante (`forward`)

1. **Cálculo de Pre-Activaciones:**

   ```cpp
   pre_activations.noalias() = weights * inputs + biases;
   ```

   - **Es decir:**

     $$
     \mathbf{z} = \mathbf{W} \cdot \mathbf{x} + \mathbf{b}
     $$

     Donde:
     - $ \mathbf{W} $ es la matriz de pesos.
     - $ \mathbf{x} $ es el vector de entrada.
     - $ \mathbf{b} $ es el vector de sesgos.
     - $ \mathbf{z} $ son las pre-activaciones.

2. **Aplicación de la Función de Activación:**

   ```cpp
   outputs = pre_activations.unaryExpr(activation);
   ```

   - **Es decir:**

     $$
     \mathbf{a} = \sigma(\mathbf{z})
     $$

     Donde $ \sigma $ es la función de activación (por ejemplo, Leaky ReLU):

     $$
     \sigma(z_i) = \begin{cases}
     z_i & \text{si } z_i > 0 \\
     0.01 z_i & \text{si } z_i \leq 0
     \end{cases}
     $$

3. **Cálculo de la Bondad:**

   ```cpp
   float goodness = outputs.squaredNorm();
   ```

   - **Es decir:**

     La **bondad** se define como la norma euclidiana al cuadrado de las activaciones de salida:

     $$
     \text{bondad} = \| \mathbf{a} \|^2 = \sum_{i=1}^{m} a_i^2
     $$

     Donde $ m $ es el número de neuronas en la capa.

#### Cálculo de la Pérdida y Gradientes

1. **Pérdida Basada en la Bondad:**

   ```cpp
   float p = 1.0f / (1.0f + std::exp(-(goodness - threshold)));
   float y = is_positive ? 1.0f : 0.0f;
   float dL_dG = p - y;
   ```

   - **Es decir:**

     Se utiliza una función sigmoide para mapear la bondad a una probabilidad $ p $:

     $$
     p = \sigma(\text{bondad} - \theta) = \frac{1}{1 + e^{-(G - \theta)}}
     $$

     Donde $ \theta $ es el umbral.

     La pérdida se define como la diferencia entre la probabilidad predicha y la etiqueta objetivo $ y $:

     $$
     \frac{\partial L}{\partial G} = p - y
     $$

     Donde:
     - $ y = 1 $ si la muestra es positiva.
     - $ y = 0 $ si la muestra es negativa.

2. **Cálculo de Gradientes:**

   ```cpp
   Eigen::VectorXf dG_da = 2.0f * outputs;
   Eigen::VectorXf dL_da = dL_dG * dG_da;
   Eigen::VectorXf dL_dz = dL_da.array() * pre_activations.unaryExpr(activation_derivative).array();
   Eigen::MatrixXf grad_weights = dL_dz * inputs.transpose();
   Eigen::VectorXf grad_biases = dL_dz;
   ```

   - **Es decir:**

     - **Derivada de la Bondad respecto a las Activaciones:**

       $$
       \frac{\partial G}{\partial \mathbf{a}} = 2 \mathbf{a}
       $$

     - **Derivada de la Pérdida respecto a las Activaciones:**

       $$
       \frac{\partial L}{\partial \mathbf{a}} = \frac{\partial L}{\partial G} \cdot \frac{\partial G}{\partial \mathbf{a}} = (p - y) \cdot 2 \mathbf{a}
       $$

     - **Derivada de la Pérdida respecto a las Pre-Activaciones:**

       Aplicando la regla de la cadena con la función de activación:

       $$
       \frac{\partial L}{\partial \mathbf{z}} = \frac{\partial L}{\partial \mathbf{a}} \cdot \frac{\partial \mathbf{a}}{\partial \mathbf{z}} = \frac{\partial L}{\partial \mathbf{a}} \cdot \sigma'(\mathbf{z})
       $$

       Donde $ \sigma'(\mathbf{z}) $ es la derivada de la función de activación.

     - **Gradientes para Pesos y Sesgos:**

       $$
       \frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{z}} \cdot \mathbf{x}^T
       $$
       $$
       \frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}}
       $$

3. **Actualización de Pesos y Sesgos:**

   Los gradientes calculados se utilizan para actualizar los pesos y sesgos mediante el optimizador seleccionado (por ejemplo, Adam).

   ```cpp
   optimizer->updateWeights(weights, grad_weights);
   optimizer->updateBiases(biases, grad_biases);
   ```

   - **Es decir:** La actualización sigue las reglas definidas por el optimizador Adam, como se explicó anteriormente.

### Funciones Auxiliares

#### División del Conjunto de Datos (`splitDataset`)

Esta función divide el conjunto de datos completo en subconjuntos de entrenamiento y validación basándose en una fracción dada.

- **Es decir:**

  Dado un conjunto de datos $ D $, se divide en $ D_{\text{entrenamiento}} $ y $ D_{\text{validación}} $ tal que:

  $$
  D_{\text{entrenamiento}} = \{ x_i \}_{i=1}^{\lfloor |D| \cdot f \rfloor}
  $$
  $$
  D_{\text{validación}} = \{ x_i \}_{i=\lfloor |D| \cdot f \rfloor + 1}^{|D|}
  $$

  Donde $ f $ es la fracción de datos destinados al entrenamiento (por ejemplo, 0.8 para 80%).

#### Selección del Optimizador (`selectOptimizer`)

Permite al usuario seleccionar el optimizador a utilizar durante el entrenamiento. Actualmente, ofrece la opción de `AdamOptimizer`.

- **Es decir:** No involucra directamente operaciones matemáticas, pero la elección del optimizador afecta cómo se actualizan los parámetros $ \theta $ durante el entrenamiento.

### Funciones de Visualización

#### Visualización PCA (`visualizePCA`)

Realiza un Análisis de Componentes Principales (PCA) sobre las activaciones de una capa de la red para reducir la dimensionalidad y visualizar los datos en 2D o 3D.

1. **Análisis de Componentes Principales:**

   ```cpp
   cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);
   cv::Mat projected_data = pca.project(data);
   ```

   - **Es decir:**

     PCA busca encontrar una base ortonormal de $ k $ dimensiones que captura la mayor varianza posible de los datos.

     $$
     \mathbf{Y} = \mathbf{X} \mathbf{W}
     $$

     Donde:
     - $ \mathbf{X} $ es la matriz de datos original.
     - $ \mathbf{W} $ es la matriz de pesos de PCA.
     - $ \mathbf{Y} $ es la matriz de datos proyectados en el espacio PCA.

2. **Mapeo de Coordenadas PCA a Píxeles:**

   Las coordenadas resultantes de PCA se escalan para ajustarse a la imagen de salida para el scatter plot.

   ```cpp
   auto mapToPixel = [&](float val, float min_val, float max_val) {
       return static_cast<int>((val - min_val) / (max_val - min_val) *
                               (img_size - 40) + 20);
   };
   ```

   - **Es decir:** Normaliza las coordenadas PCA al rango [20, img_size - 20] para visualización.

     $$
     \text{pixel\_coord} = \left( \frac{\text{val} - \text{min\_val}}{\text{max\_val} - \text{min\_val}} \right) \times (\text{img\_size} - 40) + 20
     $$

3. **Interacción con el Scatter Plot:**

   Permite al usuario hacer clic en puntos específicos del scatter plot para visualizar la imagen correspondiente.

### Funciones de Entrenamiento y Evaluación (`trainAndEvaluate`)

Esta función maneja el ciclo de entrenamiento del modelo, incluyendo la pasada hacia adelante, actualización de pesos, evaluación en el conjunto de validación y ajuste dinámico del umbral si está habilitado.

1. **Ciclo de Entrenamiento por Época:**

   Para cada época:
   
   - **Mezcla Aleatoria de Muestras:**
     
     ```cpp
     train_positive_samples.shuffle();
     train_negative_samples.shuffle();
     ```

     - **Es decir:** Se asegura de que el orden de las muestras no influya en el entrenamiento, mejorando la generalización del modelo.

   - **Entrenamiento en Muestras Positivas y Negativas:**

     ```cpp
     #pragma omp parallel for schedule(static)
     for (size_t i = 0; i < train_positive_size; ++i) {
         const Eigen::VectorXf& input = train_positive_samples.getSample(i);
         Eigen::VectorXf output;
         layer.forward(input, output, true, true, threshold, activation, activation_derivative);
     }
     ```

     - **Es decir:** Para cada muestra positiva $ x $, se realiza una pasada hacia adelante y se actualizan los parámetros para **maximizar** la bondad $ G(x) $.

     ```cpp
     #pragma omp parallel for schedule(static)
     for (size_t i = 0; i < train_negative_size; ++i) {
         const Eigen::VectorXf& input = train_negative_samples.getSample(i);
         Eigen::VectorXf output;
         layer.forward(input, output, true, false, threshold, activation, activation_derivative);
     }
     ```

     - **Es decir:** Para cada muestra negativa $ x $, se realiza una pasada hacia adelante y se actualizan los parámetros para **minimizar** la bondad $ G(x) $.

2. **Evaluación en Conjunto de Validación:**

   ```cpp
   size_t correct_positive = 0;
   size_t correct_negative = 0;
   
   // Evaluación en muestras positivas
   #pragma omp parallel for reduction(+:correct_positive)
   for (size_t i = 0; i < val_positive_size; ++i) {
       const Eigen::VectorXf& input = val_positive_samples.getSample(i);
       Eigen::VectorXf output;
       layer.forward(input, output, false, true, threshold, activation, activation_derivative);
   
       float goodness = output.squaredNorm();
   
       if (goodness > threshold) {
           ++correct_positive;
       }
   }
   
   // Evaluación en muestras negativas
   #pragma omp parallel for reduction(+:correct_negative)
   for (size_t i = 0; i < val_negative_size; ++i) {
       const Eigen::VectorXf& input = val_negative_samples.getSample(i);
       Eigen::VectorXf output;
       layer.forward(input, output, false, false, threshold, activation, activation_derivative);
   
       float goodness = output.squaredNorm();
   
       if (goodness < threshold) {
           ++correct_negative;
       }
   }
   ```

   - **Es decir:**

     - **Para Muestras Positivas:**

       Se considera correcta una predicción si la bondad $ G(x) $ excede el umbral $ \theta $.

       $$
       \text{correct\_positive} += \mathbb{I}(G(x) > \theta)
       $$

     - **Para Muestras Negativas:**

       Se considera correcta una predicción si la bondad $ G(x) $ es inferior al umbral $ \theta $.

       $$
       \text{correct\_negative} += \mathbb{I}(G(x) < \theta)
       $$

       Donde $ \mathbb{I} $ es la función indicadora.

3. **Cálculo de la Precisión:**

   ```cpp
   double accuracy = (static_cast<double>(correct_positive + correct_negative) /
                     (val_positive_size + val_negative_size)) * 100.0;
   ```

   - **Es decir:**

     $$
     \text{Precisión} = \left( \frac{\text{correct\_positive} + \text{correct\_negative}}{N_{\text{validación}}} \right) \times 100\%
     $$

     Donde $ N_{\text{validación}} $ es el número total de muestras en el conjunto de validación.

4. **Selección y Restauración del Mejor Modelo:**

   ```cpp
   if (accuracy > best_score) {
       best_score = accuracy;
       epochs_without_improvement = 0;
   
       // Guarda el mejor modelo
       best_layer = layer;
       best_threshold = threshold;
   } else {
       epochs_without_improvement++;
   }
   
   if (epochs_without_improvement >= patience) {
       layer = best_layer;
       threshold = best_threshold;
       epochs_without_improvement = 0;
   }
   ```

   - **Es decir:**

     Se mantiene y restaura el mejor modelo encontrado hasta el momento basado en la precisión. Si no hay mejora durante un número determinado de épocas (`patience`), se revierte al mejor modelo.

5. **Ajuste Dinámico del Umbral:**

   ```cpp
   if (dynamic_threshold) {
       float avg_goodness_positive = std::accumulate(goodness_positive_vals.begin(),
                                                     goodness_positive_vals.end(), 0.0f) / val_positive_size;
       float avg_goodness_negative = std::accumulate(goodness_negative_vals.begin(),
                                                     goodness_negative_vals.end(), 0.0f) / val_negative_size;
   
       threshold = (avg_goodness_positive + avg_goodness_negative) / 2.0f;
   }
   ```

   - **Es decir:**

     Si se habilita, el umbral $ \theta $ se ajusta dinámicamente calculando la media de la bondad de las muestras positivas y negativas:

     $$
     \theta = \frac{\overline{G}_{\text{positivo}} + \overline{G}_{\text{negativo}}}{2}
     $$

     Donde:
     - $ \overline{G}_{\text{positivo}} $ es la bondad promedio de las muestras positivas.
     - $ \overline{G}_{\text{negativo}} $ es la bondad promedio de las muestras negativas.

### Funciones de Visualización de Histogramas

#### Histograma Combinado de Bondades (`plotGoodnessHistogramsCombined`)

Esta función crea un histograma combinado que muestra la distribución de la bondad para muestras positivas y negativas, facilitando la visualización de la separación entre ambas clases.

1. **Cálculo de Histogramas:**

   ```cpp
   cv::calcHist(&goodness_positive, 1, 0, cv::Mat(), hist_positive, 1,
               &histSize, &histRange, uniform, accumulate);
   cv::calcHist(&goodness_negative, 1, 0, cv::Mat(), hist_negative, 1,
               &histSize, &histRange, uniform, accumulate);
   ```

   - **Es decir:**

     Se calcula la frecuencia de valores de bondad en diferentes intervalos (bins) para ambas clases.

2. **Normalización y Visualización:**

   ```cpp
   cv::normalize(hist_positive, hist_positive, 0, 400, cv::NORM_MINMAX);
   cv::normalize(hist_negative, hist_negative, 0, 400, cv::NORM_MINMAX);
   
   // Dibujar los histogramas
   for (int i = 1; i < histSize; i++) {
       cv::line(histImageCombined,
           cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_positive.at<float>(i - 1))),
           cv::Point(bin_w * i, hist_h - cvRound(hist_positive.at<float>(i))),
           cv::Scalar(255, 0, 0), 2); // Azul para positivos
       
       cv::line(histImageCombined,
           cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_negative.at<float>(i - 1))),
           cv::Point(bin_w * i, hist_h - cvRound(hist_negative.at<float>(i))),
           cv::Scalar(0, 255, 0), 2); // Verde para negativos
   }
   ```

   - **Es decir:**

     Los histogramas se normalizan para ajustarse al tamaño de la imagen y se dibujan utilizando líneas continuas para cada bin, coloreando positivamente y negativamente para distinguir las clases.

3. **Visualización del Umbral:**

   ```cpp
   cv::line(histImageCombined,
            cv::Point(threshold_x, 0),
            cv::Point(threshold_x, hist_h),
            cv::Scalar(0, 0, 0), 2); // Línea negra para el umbral
   ```

   - **Es decir:**

     Se dibuja una línea vertical en la posición correspondiente al umbral $ \theta $, permitiendo visualizar cómo separa las distribuciones de bondad de las clases positiva y negativa.

### Función Principal (`trainModel`)

Esta función orquesta todo el flujo de entrenamiento, desde la carga de datos hasta la visualización de resultados.

1. **Carga de Conjuntos de Datos:**

   ```cpp
   Dataset positive_samples("positive_images/"); // Directorio de imágenes positivas
   Dataset negative_samples("negative_images/"); // Directorio de imágenes negativas
   ```

   - **Es decir:** Se cargan y preprocesan las imágenes positivas y negativas para crear conjuntos de datos adecuados para el entrenamiento y la validación.

2. **División en Entrenamiento y Validación:**

   ```cpp
   splitDataset(positive_samples, 0.8f, train_positive_samples,
                val_positive_samples);
   splitDataset(negative_samples, 0.8f, train_negative_samples,
                val_negative_samples);
   ```

   - **Es decir:** Se divide cada conjunto (positivo y negativo) en un 80% para entrenamiento y un 20% para validación, asegurando que ambas clases estén representadas adecuadamente en ambos subconjuntos.

3. **Selección del Optimizador y Configuración Inicial:**

   ```cpp
   std::shared_ptr<Optimizer> optimizer = selectOptimizer();
   
   bool dynamic_threshold = false;
   std::cout << "¿Desea utilizar un umbral dinámico? (1 = Sí, 0 = No): ";
   int threshold_choice;
   std::cin >> threshold_choice;
   dynamic_threshold = (threshold_choice == 1);
   
   float threshold;
   std::cout << "Ingrese el umbral inicial para determinar la bondad: ";
   std::cin >> threshold;
   
   size_t epochs;
   std::cout << "Ingrese el número de épocas de entrenamiento: ";
   std::cin >> epochs;
   ```

   - **Es decir:** Se configura el optimizador y se establece el umbral $ \theta $ inicial, que es crucial para la clasificación de muestras positivas y negativas.

4. **Definición de la Función de Activación y su Derivada:**

   ```cpp
   auto activation = [](float x) -> float {
       return x > 0.0f ? x : 0.01f * x; // Leaky ReLU
   };
   
   auto activation_derivative = [](float x) -> float {
       return x > 0.0f ? 1.0f : 0.01f;
   };
   ```

   - **Es decir:**

     Se define la función de activación **Leaky ReLU** y su derivada:

     $$
     \sigma(z) = \begin{cases}
     z & \text{si } z > 0 \\
     0.01 z & \text{si } z \leq 0
     \end{cases}
     $$
     
     $$
     \sigma'(z) = \begin{cases}
     1 & \text{si } z > 0 \\
     0.01 & \text{si } z \leq 0
     \end{cases}
     $$

5. **Inicialización de la Capa Completamente Conectada:**

   ```cpp
   FullyConnectedLayer layer(input_size, output_size, optimizer);
   ```

   - **Es decir:** Se crea una capa completamente conectada con dimensiones de entrada y salida específicas, y se asocia con el optimizador seleccionado.

6. **Seguimiento del Mejor Modelo:**

   ```cpp
   double best_score = -std::numeric_limits<double>::infinity();
   FullyConnectedLayer best_layer = layer; // Copia inicial del modelo
   float best_threshold = threshold;
   ```

   - **Es decir:** Se inicializan variables para mantener el mejor modelo encontrado durante el entrenamiento, basándose en la precisión en el conjunto de validación.

7. **Entrenamiento y Evaluación:**

   ```cpp
   trainAndEvaluate(train_positive_samples, train_negative_samples,
                    val_positive_samples, val_negative_samples,
                    layer, threshold, epochs,
                    activation, activation_derivative,
                    true, best_score, dynamic_threshold,
                    goodness_positive_vals, goodness_negative_vals,
                    patience, best_layer, best_threshold);
   ```

   - **Es decir:** Se ejecuta el ciclo de entrenamiento, que incluye la actualización de parámetros y la evaluación continua para identificar y mantener el mejor modelo.

8. **Guardado del Mejor Modelo:**

   ```cpp
   layer.saveModel(model_path);
   ```

   - **Es decir:** Se guarda el conjunto de parámetros $ \theta $ (pesos y sesgos) que corresponden al mejor desempeño del modelo.

9. **Visualización de Resultados:**

   ```cpp
   plotGoodnessHistogramsCombined(goodness_positive_vals,
                                  goodness_negative_vals,
                                  threshold,
                                  final_hist_filename);
   visualizePCA(layer, val_positive_samples, val_negative_samples, num_components, threshold);
   ```

   - **Es decir:** Se visualizan las distribuciones de bondad y las proyecciones PCA para analizar la separación entre clases y la efectividad del modelo.

## Referencias

1. **Eigen:** [Eigen C++ Library](https://eigen.tuxfamily.org/)  
   Biblioteca de C++ para álgebra lineal, matrices y vectores, optimizada para operaciones matemáticas de alto rendimiento.

2. **OpenCV:** [Open Source Computer Vision Library](https://opencv.org/)  
   Biblioteca de visión por computadora que facilita la manipulación, procesamiento y visualización de imágenes y videos.

3. **FastNoiseLite:** [FastNoiseLite GitHub Repository](https://github.com/Auburn/FastNoiseLite)  
   Biblioteca de generación de ruido rápida y eficiente utilizada para crear máscaras y datos sintéticos en la generación de conjuntos de datos.

4. **Forward-Forward Algorithm:**  
   Hinton, G. (2022). *The Forward-Forward Algorithm: Some Preliminary Investigations*. Disponible en: [https://www.cs.toronto.edu/~hinton/FF.pdf](https://www.cs.toronto.edu/~hinton/FF.pdf)  
   Documento que introduce el algoritmo Forward-Forward como una alternativa al backpropagation tradicional para el entrenamiento de redes neuronales.

5. **C++ Standard Library:**  
   Documentación oficial de la [Standard Template Library (STL)](https://en.cppreference.com/w/cpp) utilizada para estructuras de datos, algoritmos y utilidades fundamentales en C++.

6. **OpenMP:** [OpenMP Official Website](https://www.openmp.org/)  
   API para programación paralela en sistemas de memoria compartida, utilizada en el código para acelerar el entrenamiento y evaluación de la red neuronal.

7. **Filesystem Library:**  
   Documentación de la [Biblioteca `<filesystem>` de C++](https://en.cppreference.com/w/cpp/filesystem) utilizada para operaciones de manejo de archivos y directorios.

8. **C++ Compiler (GCC):**  
   GNU Compiler Collection, utilizado para compilar el código con optimizaciones de rendimiento y soporte para C++17.
