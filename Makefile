NVCC = nvcc
FLAGS = --compiler-options '-Wall -Wextra -std=c++11 -O3'

# Create build and results directory if they don't exist
$(shell mkdir -p build)
$(shell mkdir -p results)

all: build/sort.exe

build/sort.exe: src/sorting.cu
	$(NVCC) -o $@ $< $(FLAGS)

run: build/sort.exe
	./build/sort.exe

clean:
	rm -rf build

.PHONY: clean