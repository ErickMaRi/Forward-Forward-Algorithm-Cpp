# Makefile

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

CXXFLAGS = $(STD) -Ofast -march=native -flto -fopenmp -ffast-math -funroll-loops -fno-math-errno `pkg-config --cflags opencv4` $(INCLUDE_DIR)
LDFLAGS = -flto -fopenmp `pkg-config --libs opencv4` $(FS_LIB)

# Directorio de headers
INCLUDE_DIR = -I include -I /usr/include/eigen3

# Ejecutables
TARGETS = noise train

all: $(TARGETS)

# Regla para compilar 'noise'
noise: src/image_utils.cpp src/main_noise.cpp
	$(CXX) $(CXXFLAGS) -o noise src/image_utils.cpp src/main_noise.cpp $(LDFLAGS)

# Regla para compilar 'train'
train: src/neural_network.cpp src/optimizer.cpp src/main_train.cpp
	$(CXX) $(CXXFLAGS) -o train src/neural_network.cpp src/optimizer.cpp src/main_train.cpp $(LDFLAGS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
