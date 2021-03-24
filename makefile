rayt: rayt.cpp
	clang++ -std=c++2a -O2 -ffast-math -march=native \
	-g -fno-omit-frame-pointer \
	-Wall -Wpedantic \
	rayt.cpp -o rayt \
	-pthread -latomic -fopenmp=libiomp5 -lomp5 \
	-lsfml-window -lsfml-graphics -lsfml-system
core: rayt
