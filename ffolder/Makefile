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
