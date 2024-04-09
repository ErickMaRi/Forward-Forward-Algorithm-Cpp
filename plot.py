import pandas as pd
import matplotlib.pyplot as plt

layer1_weights = "3.28 4.73 4.16 4.84 3.73 3.61 4.42 3.66 4.41 4.64 4.34 5.18 5.21 4.46 4.45 5.35 4.48 5.85 5.43 5.35 5.04 6.2 4.66 4.82 4.53 5.47 5.83 5.87 4.56 5.57 6.18 5.29 5.67 5.53 5.2 6.4 6.07 6.47 6.77 5.11 5.61 5.53 6.62 5.1 6.14 5.15 6.43 6.55 4.9 5.72 5.7 5.65 5.58 5.97 6.01 5.6 4.89 5.25 4.82 4.76 6.11 6.26 5.26 4.92 4.89 5.5 4.29 3.88 4.87 3.93 3.87 3.34 4.3 3.35 3.3 3.33 3.38 2.65 2.81 3.2 3.3 3.44 3.8 1.86 2.42 2.87 2.55 2.42 1.26 2.56 2.41 2.66 2.23 1.18 1.23 0.83 0.48 1.43 0.73 1.49 1.63 1.02 1.39 0.57 1.13 1.56 0.92 1.63 1.48 1.08 0.35 0.4 0.32 0.11 0.37 0.96 1.28 1.37 -0.02 1.26 0.77 -0.64 1.04 0.21 -0.17 -0.41 0.44 -0.22"
layer2_weights = "5.52 10.17 10.05 12.04 8.54 9.34 13.06 11.48 13.68 15.32 15.55 18.88 20.64 16.37 18.25 22.39 19.08 25.98 24.01 24.97 23.2 30.45 22.04 24.43 22.39 27.85 30.66 31.81 23.54 30.97 35.08 30.03 31.94 31.28 28.3 37.51 34.17 36 39.48 28.78 30.86 30.72 38.55 27.14 34.21 28.03 35.03 36.56 25.72 29.67 28.9 29.34 28.15 29.55 29.61 27.6 22.6 24.48 21.21 21.11 26.71 26.44 21.85 18.8 18.03 19.7 14.3 12.99 15.23 12.81 10.89 8.71 11.2 7.78 7.41 8.05 7.73 5.36 5.51 5.5 4.17 3.88 4.72 2.4 2.4 3.56 2.51 2.87 1.14 1.02 1.97 0.86 1.28 0.73 0.74 -0.79 0.84 1.17 0.47 -0.61 1.06 -0.75 0.07 -0.2 -0.07 0.93 -0.3 -0.21 0.45 -0.67 -0.97 0.61 -0.61 -0.25 -0.01 0.22 -0.58 -0.79 -15.87 0.77 -0.34 -153.56 -0.62 -0.88 -45.28 -96.15 -1.14 -55.37"
layer3_weights = "-0.5 2.04 0.9 4.31 1.26 1.42 3.39 2.34 4.47 6.75 6.7 10.4 12.82 7.69 8.72 14.24 10.07 21.58 18 19.06 17.27 28.56 14.17 17.1 15.57 24.28 29.19 32.11 16.84 28.64 38.73 27.35 30.7 28.78 23.98 42.94 36.8 41.34 48.24 24.92 29 28.39 46.48 22.27 37.03 24.44 37.39 42.18 19.89 27.13 25.37 25.7 24.57 26.87 27.3 22.88 14.44 18.4 13.13 12.98 21.33 21.68 13.87 11.14 10.65 12.68 6.53 4.84 6.85 5.57 4.08 2.17 4.72 1.66 1.6 2.19 1.27 0.82 0.98 1.62 -0.01 1.18 0.26 -0.52 -0.08 1.04 0.64 0.04 0.3 -0.76 -1.81 -3.29 -1.22 -1.99 -1.63 -0.01 -1.25 -1.59 -1.51 0.95 -1.85 -0.21 -2.5 0.09 0.48 -2.39 -0.65 0.97 -2.89 0.43 0.64 -3.56 0.07 -0.85 0.16 -1.84 0.92 -0.12 -0.07 -2.57 0.77 0.37 0.61 0.64 -0.91 -0.41 -0.37 0.81"

# Split weight strings into lists of floats
layer1_data = [float(x) for x in layer1_weights.split()]
layer2_data = [float(x) for x in layer2_weights.split()]
layer3_data = [float(x) for x in layer3_weights.split()]

# Create DataFrames for each layer
layer1_df = pd.DataFrame(layer1_data)
layer2_df = pd.DataFrame(layer2_data)
layer3_df = pd.DataFrame(layer3_data)

# Analyze each layer DataFrame
for layer_df, layer_name in zip([layer1_df, layer2_df, layer3_df], ["Layer 1", "Layer 2", "Layer 3"]):
  print(f"\n**{layer_name} Analysis:**")
  
  # Descriptive statistics
  print(layer_df.describe())
  
  # Distribution Visualization (Histogram)
  plt.figure(figsize=(8, 6))
  plt.hist(layer_df.values.ravel(), bins=50, edgecolor='black')
  plt.xlabel('Weight Values')
  plt.ylabel('Frequency')
  plt.title(f"{layer_name} Weight Distribution")
  plt.grid(True)
  plt.show()
  
  # Outlier Analysis (Box Plot)
  plt.figure(figsize=(6, 4))
  plt.boxplot(layer_df.values, vert=False)
  plt.xlabel('Weight Values')
  plt.ylabel('Layer Weights')
  plt.title(f"{layer_name} Weight Box Plot")
  plt.grid(True)
  plt.show()

# Additional Analysis (Optional)
# - Calculate absolute/relative difference between weights in different layers
weight_diffs = []
for i in range(1, len(layer1_df.columns)):
  weight_diffs.append(abs(layer1_df.iloc[:, i] - layer2_df.iloc[:, i]))
  weight_diffs.append(abs(layer2_df.iloc[:, i] - layer3_df.iloc[:, i]))

weight_diff_df = pd.DataFrame(weight_diffs).transpose()
weight_diff_df.columns = [f"Layer {i+1} - Layer {i+2}" for i in range(len(layer1_df.columns) - 1)]
print("\n**Weight Difference Analysis (Absolute Difference):**")
print(weight_diff_df.describe())

# You can similarly calculate relative differences and perform further analysis