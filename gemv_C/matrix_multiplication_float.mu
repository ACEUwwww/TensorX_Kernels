#include "musa_runtime.h"
#include <iostream>
#include <chrono>
#include "musa_fp16.h"
#include <vector>
#include <cstdlib>
#include <ctime>
#include <math.h>
#define TILE_WIDTH 256
#define M 1
#define K 16
#define N 16
#define EPSILON 1e-2
// @@ GPU Kernel 
// @@ Performing Matrix multiplication
__global__ void matrix_multiply(float *C, float *A, float *B, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (col < numCColumns){
      float sum = 0.0;
      for (int k = 0; k < numAColumns; k++) {
        sum += (float)(A[k] * B[k * numBColumns + col]);
      }
      C[col] = (float)sum;
    }
}

// @@ CPU Kernel for test 
void matrix_multiply_cpu(float *C, float *A, float *B, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    for(int m = 0; m < numARows; m++) {
      for(int n = 0; n < numBColumns; n++) {
        float sum = 0;
        for (int k = 0; k < numAColumns; k++) {
          sum += (A[m * numAColumns + k] * B[k * numBColumns + n]);
        }
        C[m * numCColumns + n] = sum;
      }
    }
    return;
}

// @@ Generate a float number from 0 to 4
float generateRandomFloat() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 4.0f;
}



int main(int argc, char **argv){
  float *hostA;
  float *hostB;
  float *hostC;
  float *hostCompareC;
  float *deviceA;
  float *deviceB;
  float *deviceC;
  float *deviceWarmupA;
  float *deviceWarmupB;
  float *deviceWarmupC;
  int numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns;
  numARows = M;
  numAColumns = K;
  numBRows = K;
  numBColumns = N;
  numCRows = numARows;
  numCColumns = numBColumns;
  size_t sizeA = numARows * numAColumns * sizeof(float);
  size_t sizeB = numBRows * numBColumns * sizeof(float);
  size_t sizeC = numCRows * numCColumns * sizeof(float);

  //@@ Allocate CPU memory here
  hostA = (float *)malloc(sizeA);
  hostB = (float *)malloc(sizeB);
  hostC = (float *)malloc(sizeC);
  hostCompareC = (float *)malloc(sizeC);

  //@@ Initialize matrix
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
        hostA[m * numAColumns + k] = generateRandomFloat();
    }
  }

  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
        hostB[k * numBColumns + n] = generateRandomFloat();
    }
  }

  //@@ Allocate GPU memory here
  musaMalloc((void **)&deviceA, sizeA);
  musaMalloc((void **)&deviceB, sizeB);
  musaMalloc((void **)&deviceC, sizeC);
  musaMalloc((void **)&deviceWarmupA, sizeA);
  musaMalloc((void **)&deviceWarmupB, sizeB);
  musaMalloc((void **)&deviceWarmupC, sizeC);

  // CPU gloden func
  matrix_multiply_cpu(hostCompareC, hostA, hostB, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  //@@ Device Warmup here
  musaMemcpy(deviceWarmupA, hostA, sizeA, musaMemcpyHostToDevice);
  musaMemcpy(deviceWarmupA, hostB, sizeB, musaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid((numCColumns + TILE_WIDTH -1) / TILE_WIDTH, 1, 1);
  dim3 dimBlock(TILE_WIDTH, 1, 1);  //1D


  //@@ Kernel warm up
  matrix_multiply<<<dimGrid, dimBlock>>>(deviceWarmupC, deviceWarmupA, deviceWarmupB, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  musaDeviceSynchronize();

  musaFree(deviceWarmupA);
  musaFree(deviceWarmupB); 
  musaFree(deviceWarmupC);

  //@@ Copy memory to the GPU here
  musaMemcpy(deviceA, hostA, sizeA, musaMemcpyHostToDevice);
  musaMemcpy(deviceB, hostB, sizeB, musaMemcpyHostToDevice);
  musaMemset(deviceC, 0xFFFF, sizeC);
  musaDeviceSynchronize();
  
  std::cout << "========================================" << std::endl;
  std::cout << "Performing musa computation" << std::endl;

  //@@ Launch the GPU Kernel here
  auto start_kernel = std::chrono::high_resolution_clock::now();
  matrix_multiply<<<dimGrid, dimBlock>>>(deviceC, deviceA, deviceB, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  musaDeviceSynchronize();

  //@@ Time measuring
  auto end_kernel = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_kernel = end_kernel - start_kernel;
  std::cout << "Kernel execution time: " << elapsed_kernel.count()*1000000 << " microseconds" << std::endl;

  std::cout << "Performing musa computation" << std::endl;
  std::cout << "========================================" << std::endl;

  //@@ Copy the GPU memory back to the CPU here
  musaMemcpy(hostC, deviceC, sizeC, musaMemcpyDeviceToHost);

  double timeInSeconds = elapsed_kernel.count();
  size_t totalBytes = sizeA + sizeB + sizeC;

  //@@ Caculate bandwidth in GB/s
  std::cout << "totalBytes here " << totalBytes / (1000.0 * 1000.0 * 1000.0) << " GB" << std::endl;
  std::cout << "TimeInSeconds here " << timeInSeconds << " seconds" << std::endl; 
  double bandwidth = (totalBytes/timeInSeconds) / (1000.0 * 1000.0 * 1000.0);
  std::cout << "Real bandwidth: " << bandwidth << " GB/s" << std::endl;


  //@@ Free the GPU memory here
  musaFree(deviceA); musaFree(deviceB); musaFree(deviceC);
  
  //@@ Verification
  for (int i = 0; i < numCColumns * numCRows; i++) {
        if (fabs(hostC[i] - hostCompareC[i]) > EPSILON || isnan(hostC[i])) {
            fprintf(stderr, "Result not matched at element number %d: your output = %f :: actual output = %f\n", i, hostC[i], hostCompareC[i]);
            exit(EXIT_FAILURE);
        }
    }
  std::cout << "All matched!" << std::endl;
    
// //@@ Print cmp C
//   for (int i = 0; i < numCRows; i++) {
//     for (int j = 0; j < numCColumns; j++) {
//       std::cout << hostCompareC[i * numCColumns + j] << " ";
//     }
//     std::cout << std::endl;
//   }
// //@@ Print C
//   for (int i = 0; i < numCRows; i++) {
//     for (int j = 0; j < numCColumns; j++) {
//       std::cout << hostC[i * numCColumns + j] << " ";
//     }
//     std::cout << std::endl;
//   }

  free(hostA);
  free(hostB);
  free(hostC);


  return 0;
}