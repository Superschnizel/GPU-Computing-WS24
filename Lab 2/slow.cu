#include <iostream>
#include <random>
#include <iostream>
#include <chrono>

#define GRIDSIZE 16.0

void check(cudaError_t err, std::string msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << "(error code:" << cudaGetErrorString(err) << ")";
        exit(EXIT_FAILURE);
    }
}

void init(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat) {
    // std::random_device dev;
    std::mt19937 prng(2024);
    std::uniform_int_distribution <int32_t> distrib(-16, 16);

    for (auto i = 0; i < size; i++) {
        vec_a[i] = distrib(prng);
        vec_b[i] = distrib(prng);
    }

    for (auto i = 0; i < size * size; i++)
        mat[i] = distrib(prng);
}

void compute(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat, int32_t *out) {
    auto tmp = (int32_t *) malloc(sizeof(int32_t) * size);
    for (auto i = 0; i < size; i++)
        tmp[i] = vec_a[i] + vec_b[i];

    for (auto i = 0; i < size; i++) {
        out[i] = 0;
        for (auto j = 0; j < size; j++)
            out[i] += tmp[j] * mat[i * size + j];
    }
    free(tmp);
}

__global__ void vectorAdd(const int32_t *A, const int32_t *B,
                          int32_t *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

__global__ void setZero(int32_t *A) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    A[i] = 0;
}

__global__ void matrixMult(const int32_t size, const int32_t *V, const int32_t *M, int32_t *out) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    out[i] += V[j] * M[i * size + j];
}

void pretty_print(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat) {
    std::cout << "Vec A:" << std::endl;
    for (auto i = 0; i < size; i++)
        std::cout << vec_a[i] << std::endl;

    std::cout << "Vec B:" << std::endl;
    for (auto i = 0; i < size; i++)
        std::cout << vec_b[i] << std::endl;

    std::cout << "Matrix:" << std::endl;
    for (auto i = 0; i < size; i++) {
        for (auto j = 0; j < size; j++)
            std::cout << mat[i * size + j] << " ";

        std::cout << std::endl;
    }
}

int main() {
    // int32_t size = 3;
    int32_t size = 32768;

    cudaError_t err = cudaSuccess;

    auto h_vec_a = (int32_t *) malloc(sizeof(int32_t) * size);
    auto h_vec_b = (int32_t *) malloc(sizeof(int32_t) * size);
    // Flat Buffer for matrix
    auto h_mat = (int32_t *) malloc(sizeof(int32_t * ) * size * size);
    auto h_out = (int32_t *) malloc(sizeof(int32_t) * size);

    init(size, h_vec_a, h_vec_b, h_mat);

    int32_t *d_vec_a = NULL;
    err = cudaMalloc((void **) &d_vec_a, size);
    check(err, "Failed to allocate device vector A");

    int32_t *d_vec_b = NULL;
    err = cudaMalloc((void **) &d_vec_b, size);
    check(err, "Failed to allocate device vector B");

    int32_t *d_out = NULL;
    err = cudaMalloc((void **) &d_out, size);
    check(err, "Failed to allocate device vector OUT");

    int32_t *d_mat = NULL;
    err = cudaMalloc((void **) &d_mat, size * size);
    check(err, "Failed to allocate device Matrix");

    std::cout << "Copy input data from the host memory to the CUDA device\n";
    err = cudaMemcpy(d_vec_a, h_vec_a, size, cudaMemcpyHostToDevice);
    check(err, "Failed to copy vector A from host to device");

    err = cudaMemcpy(d_vec_b, h_vec_b, size, cudaMemcpyHostToDevice);
    check(err, "Failed to copy vector B from host to device");

    err = cudaMemcpy(d_mat, h_mat, size * size, cudaMemcpyHostToDevice);
    check(err, "Failed to copy Matrix from host to device");
    // pretty_print(size, vec_a, vec_b, mat);


    int numberOfThreadsPerBlock = (int) GRIDSIZE * GRIDSIZE;
    int oneDimBlockCount = (int) ceil(size / (double) numberOfThreadsPerBlock);

    vectorAdd << oneDimBlockCount, numberOfThreadsPerBlock >> (d_vec_a, d_vec_b, d_out, size);

    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess) {
        fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror)); // if error, output error
    }

    setZero << OneDimBlockCount, NumberOfThreadsPerBlock >> (d_vec_a);

    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess) {
        fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror)); // if error, output error

    }

    dim3 threadsPerBlock(GRIDSIZE, GRIDSIZE, 1);
    const int32_t twoDimBlockCount = ceil(size / GRIDSIZE);
    dim3 numBlocks(twoDimBlockCount, twoDimBlockCount, 1);

    matrixMult << numBlocks, threadsPerBlock >> (size, d_out, d_mat, d_vec_a);

    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess)
        fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror)); // if error, output error

    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();

    std::cout << "First 3 entries of Out Vec:" << std::endl;
    for (int32_t i = 0; i < 3; i++)
        std::cout << out[i] << std::endl;

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    free(vec_a);
    free(vec_b);
    free(mat);
    free(out);

    return 0;
}
