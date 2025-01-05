CXX = g++

GCC_VERSION := $(shell $(CXX) -dumpversion | cut -d. -f1)

# Definimos el estándar de C++ y la libstdc++fs si hace falta
ifeq ($(shell [ $(GCC_VERSION) -ge 9 ] && echo yes),yes)
    STD = -std=c++17
    FS_LIB =
else
    STD = -std=c++17
    FS_LIB = -lstdc++fs
endif

# Paths de include (Eigen, headers propios, etc.)
INCLUDE_DIR = -I include -isystem /usr/include/eigen3

# Flags de compilación y link según modo (debug o release)
ifeq ($(BUILD),debug)
    # ---- COMPILACIÓN DEBUG ----
    CXXFLAGS = $(STD) -Og -g -Wall -Wextra -Wpedantic -Werror -Wunused \
               -fopenmp -fsanitize=address -fno-omit-frame-pointer \
               `pkg-config --cflags opencv4` $(INCLUDE_DIR)
    LDFLAGS  = -fopenmp -fsanitize=address \
               `pkg-config --libs opencv4` $(FS_LIB)

else
    # ---- COMPILACIÓN RELEASE ----
    CXXFLAGS = $(STD) -Ofast -march=native -flto -fopenmp -ffast-math \
               -funroll-loops -fno-math-errno -Wall -Wextra -Wpedantic \
               -Werror -Wunused \
               `pkg-config --cflags opencv4` $(INCLUDE_DIR)
    LDFLAGS  = -flto -fopenmp \
               `pkg-config --libs opencv4` $(FS_LIB)
endif

# Ejecutables que queremos generar
TARGETS = noise train

all: $(TARGETS)

# Reglas de compilación
noise: src/image_utils.cpp src/main_noise.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

train: src/neural_network.cpp src/optimizer.cpp src/plotting.cpp src/main_train.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
