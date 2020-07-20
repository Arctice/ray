tobjl:
	cd tinyobjloader && mkdir -p build && cd build && cmake .. && make
rt: rt.cpp
	clang++ -std=c++17 -Ofast -march=native \
	-Wall -Wpedantic \
	rt.cpp -o rt \
	-pthread -latomic \
	-lsfml-window -lsfml-graphics -lsfml-system \
	-L./tinyobjloader/build/ -ltinyobjloader
core: rt
debug:
	clang++ -std=c++17 -O1 -march=native \
	-fsanitize=undefined,address -g -fno-omit-frame-pointer \
	-Wall -Wpedantic \
	rt.cpp -o rt \
	-pthread -latomic \
	-lsfml-window -lsfml-graphics -lsfml-system \
	-L./tinyobjloader/build/ -ltinyobjloader
run:
	./rt
