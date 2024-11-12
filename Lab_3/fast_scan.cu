#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>

#include "helper.cu"

#define BLOCK_DIM 512

void sequential_scan(size_t size, float *in_h, float *out_h)
{
  out_h[0] = in_h[0];
  out_h[1] = in_h[1];
  for (auto i = 2; i < size; i += 2)
  {
    float real_prev = out_h[i - 2];
    float real_cur = in_h[i];
    float im_prev = out_h[i - 1];
    float im_cur = in_h[i + 1];

    out_h[i] = real_prev * real_cur - im_prev * im_cur;
    out_h[i + 1] = real_prev * im_cur + real_cur * im_prev;
  }
}

__global__ void parallell_scan(size_t size, float *input, float *output)
{
  // create shared memory for block to compute local product
  __shared__ float block_result[BLOCK_DIM * 4];

  // compute the starting point of this segment.
  auto segment = 2 * blockDim.x * blockIdx.x;
  // compute the global and local id of the current thread
  auto global_id = segment + threadIdx.x * 2;
  auto local_id = threadIdx.x * 2;
  float a, b, c, d;

  // need to make sure we are inside bounds.
  if (global_id < size)
  {
    // calculate the first product.
    //                    | >  < |
    // [real a][imag a]...[real b][imag b]
    // (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    if (threadIdx.x > 0)
    {
      // calculate first iteration
      a = input[global_id - 2];
      b = input[global_id - 1];
      c = input[global_id];
      d = input[global_id + 1];
      block_result[local_id] = a * c - b * d;
      block_result[local_id + 1] = a * d + b * c;
    }
    else
    {
      // if index 0, copy input to shared
      block_result[local_id] = input[global_id];
      block_result[local_id + 1] = input[global_id + 1];
    }
  }

  // Stride is the distance to look back.
  for (auto stride = 2; stride < blockDim.x; stride *= 2)
  {
    // synchronize threads here
    __syncthreads();

    if (global_id >= size)
    {
      continue;
    }

    int previous_id = local_id - (stride * 2);
    // only run if previous id is not < 0
    if (previous_id >= 0)
    {
      // [real a][imag a]...[real b][imag b]
      // (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
      a = block_result[previous_id];
      b = block_result[previous_id + 1];
      c = block_result[local_id];
      d = block_result[local_id + 1];

      block_result[local_id] = a * c - b * d;
      block_result[local_id + 1] = a * d + b * c;

      continue;
    }
  }

  if (global_id >= size)
  {
    return;
  }

  // write result of block to output and finish computation on the HOST.
  output[global_id] = block_result[local_id];
  output[global_id + 1] = block_result[local_id + 1];
}

__global__ void finish_scan(size_t size, float *output)
{
  // compute the starting point of this segment.
  auto segment = 4 * blockDim.x * blockIdx.x;
  // compute the global and local id of the current thread
  auto global_id = segment + threadIdx.x * 2;

  if (global_id >= size)
  {
    return;
  }

  float a = output[global_id];
  float b = output[global_id + 1];

  for (int previous_id = segment - 2; previous_id > 0; previous_id -= 2 * blockDim.x)
  {
    float c = output[previous_id];
    float d = output[previous_id + 1];

    a = a * c - b * d;
    b = a * d + b * c;
  }
}

int main()
{
  size_t size = 33554432 * 2;
  float *in_d, *in_h, *out_d, *out_h, *out_h2;

  // Allocate on host
  in_h = (float *)calloc(size, sizeof(float));
  CHECK_ALLOC(in_h);
  out_h = (float *)calloc(size, sizeof(float));
  CHECK_ALLOC(out_h);
  out_h2 = (float *)calloc(size, sizeof(float));
  CHECK_ALLOC(out_h2);
  // Allocate on device
  CUDA_CALL(cudaMalloc((void **)&in_d, size * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&out_d, size * sizeof(float)));

  // Initialize
  int e = random_init(size, in_d, in_h);
  if (e == EXIT_FAILURE)
    return EXIT_FAILURE;

  std::cout << "Running sequential scan" << std::endl;

  auto start = std::chrono::system_clock::now();
  sequential_scan(size, in_h, out_h);
  auto end = std::chrono::system_clock::now();

  std::cout << "First 3 entries of In Vec:" << std::endl;
  for (int32_t i = 0; i < 5 * 2; i += 2)
    std::cout << in_h[i] << "," << in_h[i + 1] << std::endl;
  std::cout << "First 3 entries of Out Vec:" << std::endl;
  for (int32_t i = 0; i < 5 * 2; i += 2)
    std::cout << out_h[i] << " + " << out_h[i + 1] << std::endl;

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Running parralel scan" << std::endl;

  // (numElements + threadsPerBlock - 1) / threadsPerBlock
  int numberOfBlocks = (size / 2 + BLOCK_DIM - 1) / BLOCK_DIM;
  int numberOfThreads = BLOCK_DIM;

  std::cout << "With " << numberOfBlocks << " Blocks a " << numberOfThreads << " Threads." << std::endl;

  start = std::chrono::system_clock::now();
  parallell_scan<<<numberOfBlocks, numberOfThreads>>>(size, in_d, out_d);
  CUDA_CALL(cudaGetLastError());
  finish_scan<<<numberOfBlocks, numberOfThreads>>>(size, out_d);
  end = std::chrono::system_clock::now();

  CUDA_CALL(cudaMemcpy(out_h2, out_d, size, cudaMemcpyDeviceToHost));

  elapsed_seconds = end - start;
  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
  std::cout << "First 3 entries of Out Vec:" << std::endl;
  for (int32_t i = 0; i < 5 * 2; i += 2)
    std::cout << out_h2[i] << " + " << out_h2[i + 1] << std::endl;

  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Comparing results" << std::endl;

  for (size_t i = 0; i < size; i++)
  {
    if (std::abs(out_h[i] - out_h2[i]) > 10)
    {
      std::cout << "An error occured at position " << i << ":\nWith values " << out_h[i] << " and " << out_h2[i] << std::endl;
      break;
    }
  }

  CUDA_CALL(cudaFree(in_d));
  CUDA_CALL(cudaFree(out_d));
  free(in_h);
  free(out_h);
  return EXIT_SUCCESS;

  return 0;
}
