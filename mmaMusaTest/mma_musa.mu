#include "musa_runtime.h"
#include <iostream>
#include <chrono>
#include "musa_fp16.h"
#include <vector>
#include <cstdlib>
#include <ctime>
#include <math.h>

#define M 32
#define K 32
#define N 32
#define BLOCKSIZEX  128
#define EPSILON 1e0

#define ENABLE_DEBUG 0

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
// @@ Performing GEMV
__global__ 
void gemv(half *C, int size_C, int _0, half *A, int size_A, int _1, half* B, int size_B, int _2, 
          int _M, int _K, int _N, float* debug_output, int sizeDebug, int _3) {
    int tx = threadIdx.x;
    // Zero Padding to solve bank conflicts
    // 128 * 8 + 2 * 16 + 2 * 32 + 2 * 48
    __shared__ half shared_A[1216];
    half RegisterA[8];
    half RegisterB[8];
    float RegisterC[8] = {0.f};
    auto f = [] (int x) -> int { 
      return 8 * x * (x + 1);
    };
    int offset_share = (tx / 64) * 512 + (tx % 4) * 128 + (tx % 64) / 4 * 8 + f(tx % 4) + (tx / 64) * 96;
    *(float4*)&shared_A[offset_share] = *(float4*)&A[tx * 8];

    *(float*)&RegisterA[0] = *(float*)&shared_A[tx * 2 + tx / 64 * 16];
    *(float*)&RegisterA[2] = *(float*)&shared_A[tx * 2 + 512 + 96 + tx / 64 * 16];
    *(float*)&RegisterA[4] = *(float*)&shared_A[tx * 2 + 256 + 16 + 32 + tx / 64 * 48];
    *(float*)&RegisterA[6] = *(float*)&shared_A[tx * 2 + 768 + 96 + 16 + 32 + tx / 64 * 48];

    *(float4*)RegisterB = *(float4*)&B[8 * tx];
    
    __musa_fmma_m32n32k16_mma((int *)RegisterC, (const int *)&RegisterB[0], (const int *)&RegisterA[0], (const int *)RegisterC, 0, 0, 0, 0, 1, 1);
    __musa_fmma_m32n32k16_mma((int *)RegisterC, (const int *)&RegisterB[4], (const int *)&RegisterA[4], (const int *)RegisterC, 0, 0, 0, 0, 1, 1);
  
    int rowC = (tx % 64) / 32 * 8 + tx % 8;
    int colC = (tx / 64) * 16 + (tx % 32) / 8 * 4;

    half halfRegisterC[8];
    for (int i = 0; i < 8; i++) {
      int shuffle_idx = i % 2 * 4 + i / 2;
      halfRegisterC[shuffle_idx] = (half)RegisterC[i];
    }
    *(long*)&C[rowC * 32 + colC] = *(long*)&halfRegisterC[0];
    *(long*)&C[(rowC + 16) * 32 + colC] = *(long*)&halfRegisterC[4];

    return;
}

// @@ CPU Kernel for test 
void matrix_multiply_cpu(half *C, half *A, half *B, int _M, int _K, int _N) {
    for(int m = 0; m < M; m++) {
      for(int n = 0; n < N; n++) {
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
void reduce_sum_cpu(half* C_out, half *C, int BSIZEY, int _N) {
    for (int i = 0; i < _N; i++) {
      float sum_col = 0.f;
      for (int j = 0; j < BSIZEY; j++) {
        sum_col += (float)C[j * _N + i];
      }
      C_out[i] = (half)sum_col;
    }
    return;
}

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

// @@ Weight reorder for weight matrix
template<typename T>
void weightReorder_K_N(T* out, T* in, int weight_row, int weight_col) {
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

int main(int argc, char **argv) {
  half *hostA;
  half *hostB;
  half *hostB_reorder;
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
  hostB_reorder = (half *)malloc(sizeB);
  hostC = (half *)malloc(sizeC);
  hostCompareC = (half *)malloc(sizeC);
  host_C_out = (half *)malloc(sizeC_out);
  hostDebug = (float *)malloc(sizeDebug);

  //@@ Initialize matrix
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
        hostA[m * K + k] = (half)generateRandomhalf();
        // hostA[m * K + k] = (half)1.f;
        // hostA[m * K + k] = (half)(m * K + k);
    }
  }

  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
        hostB[k * N + n] = (half)generateRandomhalf();
        // hostB[k * N + n] = (half)(k * N + n);
        // hostB[k * N + n] = (half)1.f;
    }
  }

  //@@ Allocate GPU memory here
  musaMalloc((void **)&deviceA, sizeA);
  musaMalloc((void **)&deviceB, sizeB);
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
  dim3 dimGrid(1, 1, 1);
  dim3 dimBlock(BLOCKSIZEX, 1, 1); 

  //@@ Kernel warm up
  // for (int i = 0; i < 3; ++i)
    gemv<<<dimGrid, dimBlock>>>
    (deviceWarmupC, sizeC, 0, deviceWarmupA, sizeA, 0, deviceWarmupB, sizeB, 0, M, K, N, deviceDebug, sizeDebug, 0);
  musaDeviceSynchronize();

  musaFree(deviceWarmupA);
  musaFree(deviceWarmupB); 
  musaFree(deviceWarmupC);

  //@@ Copy memory to the GPU here
  //@@ Weight_reoorder here
  weightReorder_32_32(hostB,hostB, K ,N);
  musaMemcpy(deviceA, hostA, sizeA, musaMemcpyHostToDevice);
  musaMemcpy(deviceB, hostB, sizeB, musaMemcpyHostToDevice);
  musaMemset(deviceC, 0xFFFF, sizeC);
  musaDeviceSynchronize();
  
  std::cout << "========================================" << std::endl; 
  std::cout << "Performing musa computation" << std::endl;

  //@@ Launch the GPU Kernel here
  auto start_kernel = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; ++i) {
    gemv<<<dimGrid, dimBlock>>>
    (deviceC, sizeC, 0, deviceA, sizeA, 0, deviceB, sizeB, 0, M, K, N, deviceDebug, sizeDebug, 0);
  }
  musaDeviceSynchronize();

  //@@ Error check and Time measuring
  auto end_kernel = std::chrono::high_resolution_clock::now();
  MUSA_CHECK(musaGetLastError());
  std::chrono::duration<double> elapsed_kernel = end_kernel - start_kernel;
  std::cout << "Kernel execution time: " << elapsed_kernel.count()*1000000 / 10.f << " us" << std::endl;
  std::cout << "Performing musa computation" << std::endl;
  std::cout << "========================================" << std::endl;

  //@@ Copy the GPU memory back to the CPU here
  musaMemcpy(hostC, deviceC, sizeC, musaMemcpyDeviceToHost);
  musaMemcpy(hostDebug, deviceDebug, sizeDebug, musaMemcpyDeviceToHost);

  //@@ Debug Testing
  if (ENABLE_DEBUG) {
    for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
        std::cout << (float)hostC[i * 32 + j] << " ";
      }
      std::cout<<std::endl;
    }
    std::cout << "============== COMP C ==============" << std::endl;
    for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
        std::cout << (float)hostCompareC[i * 32 + j] << " ";
      }
      std::cout<<std::endl;
    }
  }


  //@@ Caculate bandwidth
  double timeInSeconds = elapsed_kernel.count() / 10.f;
  size_t totalBytes = sizeA + sizeB + sizeC;
  std::cout << "totalBytes here " << totalBytes / (1000.0 * 1000.0 * 1000.0) << " GB" << std::endl;
  std::cout << "TimeInSeconds here " << timeInSeconds << " seconds" << std::endl; 
  double bandwidth = (totalBytes/timeInSeconds) / (1000.0 * 1000.0 * 1000.0);
  std::cout << "Real bandwidth: " << bandwidth << " GB/s" << std::endl;

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
