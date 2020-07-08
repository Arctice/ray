core:
	clang++ -std=c++17 -Ofast -march=native \
	-Wall -Wpedantic \
	rt.cpp -o rt \
	-pthread -latomic \
	-lsfml-window -lsfml-graphics -lsfml-system
debug:
	clang++ -std=c++17 -O1 rt.cpp -o rt \
	-fsanitize=undefined,address -g -fno-omit-frame-pointer \
	-lpthread -march=native  \
	-lsfml-window -lsfml-graphics -lsfml-system
run:
	./rt
