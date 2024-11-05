NVCC = nvcc

all: matmult

matmult: slow.cu
	# clang++ -Wall -Wextra -Werror -O2 -o slow slow.cpp
	$(NVCC) -o vectorAdd -std=c++20 $^


clean:
	rm -f *.o *~ core
