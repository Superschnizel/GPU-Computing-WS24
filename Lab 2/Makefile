NVCC = nvcc

all: matmult

matmult: slow.cu
	# clang++ -Wall -Wextra -Werror -O2 -o slow slow.cpp
	$(NVCC) -o lab2 $^


clean:
	rm -f *.o *~ core
