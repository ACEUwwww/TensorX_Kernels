#include "musa_runtime.h"
#include <iostream>
#include <chrono>
#include "musa_fp16.h"
#include <vector>
#include <cstdlib>
#include <ctime>
#include <math.h>

#define M 32
#define K 256
#define N 256
#define BLOCKSIZEX 256
#define WARPSIZE 128
#define EPSILON 1e0
#define ENABLE_DEBUG 1

#define MUSA_CHECK(expr)                                           \
  do {                                                             \
    musaError_t status = (expr);                                   \
    if (status != musaSuccess) {                                   \
      const char* err_str = musaGetErrorString(status);            \
      std::cerr << "Error on line " << __LINE__ << ": " << err_str \
                << std::endl;                                      \
    }                                                              \
  } while (0)

// @@ GPU Kernel 
// @@ Performing 32 batch GEMV
extern "C"{__global__ void gemv_32_batch (void* C,int size_C,int _0,void* A,int size_A,int _1,void* B,int size_B,int _2,int _M,int _K,int _N,void* debug_output,int sizeDebug,int _3){}}

// @@ CPU Kernel for test 
void matrix_multiply_cpu (half *C, half *A, half *B, int _M, int _K, int _N) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
          sum += ((float)A[m * K + k] * (float)B[k * N + n]);
        }
        C[m * N + n] = (half)sum;
        // C[m * N + n] = (half)0.f;
      }
    }
    return;
}

//@@ Reduce at CPU side
void reduce_sum_cpu (half* C_out, half *C, int BSIZEY, int _N) {
    for (int i = 0; i < _N; i++) {
      float sum_col = 0.f;
      for (int j = 0; j < BSIZEY; j++) {
        sum_col += (float)C[j * _N + i];
      }
      C_out[i] = (half)sum_col;
    }
    return;
}

// @@ Matrix Transpose
template<typename T>
void matrixTrans (T* out, T* in, int row, int col) {
  if (!out || !in) return;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      out[j * row + i] = in[i * col + j];
    }
  }
  return;
}

// Weight reorder in the unit of 32 * 32
template<typename T>
void weightReorder_32_32 (T* weight_out, T* weight_in, int weight_row, int weight_col) {
  if (weight_row != 32 || weight_col != 32) return;
  
  // Transpose N continues matrix to K continues matrix
  T* weight_trans = (T*)malloc(weight_row * weight_col * sizeof(T));
  T* weight_temp = (T*)malloc(weight_row * weight_col * sizeof(T));
  matrixTrans(weight_trans, weight_in, weight_row, weight_col);
  int trans_row = weight_col; 
  int trans_col = weight_row;

  // Shuffle rows for the transposed matrix
  for (int row = 0; row < trans_row; row++) {
    int new_row = (row % 4) * 8 + row / 4;
    for (int col = 0; col < trans_col; col++) {
      weight_temp[new_row * trans_col + col] = weight_trans[row * trans_col + col];
    }
  }

  // Shuffle for 8 elements align
  for (int row = 0; row < trans_row; row++) {
    for (int col = 0; col < trans_col; col++) {
      int r = row % 16;
      int c = col % 16;
      int tid = (r / 8) * 32 + (c / 8) * 64 + (c % 8) / 2 + (r % 8) * 4;
      int offset_out = 8 * tid + row / 16 * 2 + col / 16 * 4 + col % 2;
      weight_out[offset_out] = weight_temp[row * trans_col + col]; 
    }
  }
  return;
}

// @@ Zero padding reshape Matrix
template<typename T>
T* zeroPaddingReshape (T* in, int weight_row, int weight_col) {
  int padded_row = (weight_row + 31) / 32 * 32;
  int padded_col = (weight_col + 31) / 32 * 32;
  T* padded_matrix = new T[padded_row * padded_col];
  memset(padded_matrix, 0, padded_row * padded_col * sizeof(T));
  for (int i = 0; i < weight_row; ++i) {
    for (int j = 0; j < weight_col; ++j) {
        padded_matrix[i * padded_col + j] = in[i * weight_col + j];
    }
  }
  return padded_matrix;
}

// @@ Weight reorder for weight matrix
template<typename T>
void weightReorder_K_N (T* out, T* in, int weight_row, int weight_col) {
  T* tempMatrix = (T*)malloc(sizeof(T) * 32 * 32);
  for (int row = 0; row < weight_row ; row += 32) {
    for (int col = 0; col < weight_col; col += 32) {
      for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
          tempMatrix[i * 32 + j] = in[(row + i) * weight_col + col + j];
        }
      }
      weightReorder_32_32(tempMatrix, tempMatrix, 32, 32);
      for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
          out[(row + i) * weight_col + col + j] = tempMatrix[i * 32 + j];
        }
      }
    }
  }  
  return;
}

// @@ Generate a half number from 0 to 4
float generateRandomhalf() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 4.0f -2.0f;
  // return 1.f;
}

// @@ Body Program
int main(int argc, char **argv) {
  half *hostA;
  half *hostB;
  half *hostC;
  half *hostCompareC;
  half *deviceA;
  half *deviceB;
  half *deviceC;
  half *deviceWarmupA;
  half *deviceWarmupB;
  half *deviceWarmupC;
  half *host_C_out;
  float *hostDebug;
  float *deviceDebug;
  size_t sizeA = M * K * sizeof(half);
  size_t sizeB = K * N * sizeof(half);
  size_t sizeC = N * M * sizeof(half);
  size_t sizeC_out = M * N * sizeof(half);
  size_t sizeDebug = BLOCKSIZEX * sizeof(float);

  //@@ Allocate CPU memory here
  hostA = (half *)malloc(sizeA);
  hostB = (half *)malloc(sizeB);
  hostC = (half *)malloc(sizeC);
  hostCompareC = (half *)malloc(sizeC);
  host_C_out = (half *)malloc(sizeC_out);
  hostDebug = (float *)malloc(sizeDebug);

  //@@ Initialize matrix
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
        // hostA[m * K + k] = (half)generateRandomhalf();
        // hostA[m * K + k] = (half)1.f;
        hostA[m * K + k] = (half)((m * K + k) % 8 + 1);
    }
  }

  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
        // hostB[k * N + n] = (half)generateRandomhalf();
        hostB[k * N + n] = (half)((k * N + n) % 8 + 1);
        // hostB[k * N + n] = (half)1.f;
    }
  }

  //@@ Do padding and weight reorder
  int padded_K = (K + 31) / 32 * 32;
  int padded_N = (N + 31) / 32 * 32;
  size_t paddedSizeB = sizeof(half) * padded_K * padded_N;
  half* paddedHostB = zeroPaddingReshape(hostB, K, N);
  weightReorder_K_N(paddedHostB, paddedHostB, padded_K, padded_N);

  //@@ Allocate GPU memory here
  musaMalloc((void **)&deviceA, sizeA);
  musaMalloc((void **)&deviceB, paddedSizeB);
  musaMalloc((void **)&deviceC, sizeC);
  musaMalloc((void **)&deviceWarmupA, sizeA);
  musaMalloc((void **)&deviceWarmupB, sizeB);
  musaMalloc((void **)&deviceWarmupC, sizeC);
  musaMalloc((void **)&deviceDebug, sizeDebug);

  // CPU gloden func
  matrix_multiply_cpu(hostCompareC, hostA, hostB, M, K, N);

  //@@ Device Warmup here
  musaMemcpy(deviceWarmupA, hostA, sizeA, musaMemcpyHostToDevice);
  musaMemcpy(deviceWarmupA, hostB, sizeB, musaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid((padded_N + 32 - 1) / 32, 1, 1);
  dim3 dimBlock(BLOCKSIZEX, 1, 1); 
  
  //@@ Kernel warm up
  // for (int i = 0; i < 3; ++i)
  //   gemv<<<dimGrid, dimBlock>>>
  //   (deviceWarmupC, sizeC, 0, deviceWarmupA, sizeA, 0, deviceWarmupB, sizeB, 0, M, PaddedK, PaddedN, deviceDebug, sizeDebug, 0);
  // musaDeviceSynchronize();

  musaFree(deviceWarmupA);
  musaFree(deviceWarmupB); 
  musaFree(deviceWarmupC);

  //@@ Copy memory to the GPU here
  musaMemcpy(deviceA, hostA, sizeA, musaMemcpyHostToDevice);
  musaMemcpy(deviceB, paddedHostB, paddedSizeB, musaMemcpyHostToDevice);
  musaMemset(deviceC, 0xFFFF, sizeC);
  musaMemset(deviceDebug, 0xFFFF, sizeDebug);
  musaDeviceSynchronize();
  
  std::cout << "========================================" << std::endl; 
  std::cout << "Performing musa computation" << std::endl;

  //@@ Launch the GPU Kernel
  auto start_kernel = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < 10; ++i) 
    gemv_32_batch<<<dimGrid, dimBlock>>>
    (deviceC, sizeC, 0, deviceA, sizeA, 0, deviceB, paddedSizeB, 0, M, padded_K, padded_N, deviceDebug, sizeDebug, 0);
  musaDeviceSynchronize();

  //@@ Error check and Time measuring
  auto end_kernel = std::chrono::high_resolution_clock::now();
  MUSA_CHECK(musaGetLastError());
  std::chrono::duration<double> elapsed_kernel = end_kernel - start_kernel;
  std::cout << "Kernel execution time: " << elapsed_kernel.count()*1000000 << " us" << std::endl;
  std::cout << "Performing musa computation" << std::endl;
  std::cout << "========================================" << std::endl;

  //@@ Copy the GPU memory back to the CPU here
  musaMemcpy(hostC, deviceC, sizeC, musaMemcpyDeviceToHost);
  musaMemcpy(hostDebug, deviceDebug, sizeDebug, musaMemcpyDeviceToHost);

  //@@ Debug Testing
  if (ENABLE_DEBUG) {
    for (int i = 0; i < BLOCKSIZEX; i++) {
      std::cout << i << ":" << *(float*)&hostDebug[i] << std::endl;
    }
  }

  //@@ Caculate bandwidth
  double timeInSeconds = elapsed_kernel.count();
  size_t totalBytes = sizeA + paddedSizeB + sizeC;
  std::cout << "Data :" << totalBytes / (1000.0 * 1000.0 * 1000.0) << " GB" << std::endl;
  double bandwidth = (totalBytes/timeInSeconds) / (1000.0 * 1000.0 * 1000.0);
  std::cout << "Bandwidth : " << bandwidth << " GB/s" << std::endl;

  //@@ Free the GPU memory here
  musaFree(deviceA);
  musaFree(deviceB); 
  musaFree(deviceC);
  musaFree(deviceDebug);
  
  //@@ Verification
  for (int i = 0; i < M * N; i++) {
        if (fabs((float)hostC[i] - (float)hostCompareC[i]) > EPSILON || isnan((float)hostC[i])) {
            fprintf(stderr, "Result not matched at element number %d: your output = %f :: actual output = %f\n", i, (float)hostC[i], (float)hostCompareC[i]);
            exit(EXIT_FAILURE);
        }
    }
  std::cout << "All matched!" << std::endl;

//@@ Free Host Memory
  free(hostA);
  free(hostB);
  free(hostC);
  free(hostDebug);

  return 0;
}
