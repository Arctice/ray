flags = -std=c++17 -ffast-math -march=native -fopenmp=libiomp5 \
	-g -fno-omit-frame-pointer -Wall -Wpedantic -Wno-c++20-designator
libs = -pthread -latomic -lomp5 -lsfml-window -lsfml-graphics -lsfml-system

rayt: rayt.cpp
	clang++ ${flags} rayt.cpp -o rayt ${libs}

core: flags += -O2
core: rayt

debug: flags += -fsanitize=undefined,address
debug: rayt
