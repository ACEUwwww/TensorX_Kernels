#include "musa_runtime.h"
#include <iostream>
#include <chrono>
#include "musa_fp16.h"
#include <vector>
#include <cstdlib>
#include <ctime>
#include <math.h>

#define M 1
#define K 4096
#define N 16384
#define BLOCKSIZEX  8
#define BLOCKSIZEY 32
#define EPSILON 1e0
#define LOADCOUNT 4
#define TILEV 2048


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
template <typename T, int BLOCKSIZE_x, int BLOCKSIZE_y, int LOAD_COUNT, int TILE_V>
__global__ void matrix_multiply(half *C, half *A, half *B, int _M, int _K, int _N) {
//==============================================Final optimization version ==================================================
  constexpr int max_size = (BLOCKSIZEY / 2) * (BLOCKSIZEX * 8 + 1) > TILE_V? (BLOCKSIZEY / 2) * (BLOCKSIZEX * 8 + 1) : TILE_V;
  __shared__ float reduce_sum_data[max_size];
  half* s_data = reinterpret_cast<half*>(reduce_sum_data);
  int col =  8 * (blockIdx.x * BLOCKSIZEX + threadIdx.x);
  int row = threadIdx.y;
  float sum[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
  half temp[LOAD_COUNT][8];
  half temp_sdata[8];
//=============== BIG LOOP FOR TILE_V ==================
  // K/TILE_V times
  for (int i = 0; i < (K + TILE_V - 1) / TILE_V; i++) {
    int idxA = row + i * TILE_V;
    int idxB = idxA * N + col;

//================= READ TO SMEM ======================
    __syncthreads();
    int index_in_block = threadIdx.x + threadIdx.y * blockDim.x;
    #pragma unroll
    for (int m = 0; m < (TILE_V + BLOCKSIZEY * BLOCKSIZEX * 8 - 1) / (BLOCKSIZEY * BLOCKSIZEX * 8); m++) {
      if (index_in_block * 8 < TILE_V && index_in_block * 8 + i * TILE_V < K) {
        int offset_asm = (index_in_block * 8 + i * TILE_V) * 2;
        asm("DMA.LD.B128 %0, %1, %2;"
            : "=R"(temp_sdata)
            : "R"(A), "R"(offset_asm)); 
        #pragma unroll
        for(int r = 0; r < 8; r++) {
          s_data[8 * index_in_block + r] = temp_sdata[r];
        }
      }
      index_in_block += BLOCKSIZEY * BLOCKSIZEX;
    }
    __syncthreads();

//======================= LOAD FROM MATRIX B AND CACULATION WITH LOADCOUNT ===============
// TILE_V / (LOAD_COUNT * BLOCKSIZEY) times
    if (col < N && row < K) {
      #pragma unroll
      for (int j = 0; j < (TILE_V + LOAD_COUNT * BLOCKSIZEY - 1) / (LOAD_COUNT * BLOCKSIZEY); j++) {
        // LOADCOUNT times for reading data from GMEM
        if constexpr (LOAD_COUNT == 1) {
          int stride = j * LOAD_COUNT * BLOCKSIZEY * N;
          if (idxB + stride < K * N) {
            asm("DMA.LD.B128 %0, %1, %2;"
            : "=R"(temp[0])
            : "R"(B), "R"((idxB + stride) * 2));
          }
        } else if constexpr (LOAD_COUNT == 2) {
          int stride = j * LOAD_COUNT * BLOCKSIZEY * N;
          if (idxB + stride < K * N) {
            asm("DMA.LD.B128 %0, %2, %3;\n\t"
            "DMA.LD.B128 %1, %2, %4;" 
            : "=&R"(temp[0]), "=R"(temp[1])
            : "R"(B), "R"((idxB + stride) * 2), "R"((idxB + stride + BLOCKSIZEY * N) * 2));
          }
        } else if constexpr (LOAD_COUNT == 4) {
          int stride = j * LOAD_COUNT * BLOCKSIZEY * N;
          if (idxB + stride < K * N){
            asm("DMA.LD.B128 %0, %4, %5;\n\t"
                "DMA.LD.B128 %1, %4, %6;\n\t" 
                "DMA.LD.B128 %2, %4, %7;\n\t" 
                "DMA.LD.B128 %3, %4, %8;" 
            : "=&R"(temp[0]), "=&R"(temp[1]), "=&R"(temp[2]), "=R"(temp[3])
            : "R"(B), "R"((idxB + stride) * 2), "R"((idxB + stride + BLOCKSIZEY * N) * 2), "R"((idxB + stride + 2 * BLOCKSIZEY * N) * 2), "R"((idxB + stride + 3 * BLOCKSIZEY * N) * 2));
          }
        } else {
          #pragma unroll
          for (int k = 0; k < LOAD_COUNT; k++) {
            int stride = k * BLOCKSIZEY * N + j * LOADCOUNT * BLOCKSIZEY * N;
            asm("DMA.LD.B128 %0, %1, %2;"
            : "=R"(temp[k])
            : "R"(B), "R"((idxB + stride) * 2));
          }
        }
        // LOADCOUNT times
        #pragma unroll 
        for (int k = 0; k < LOAD_COUNT; k++) {
          int offset_smem = threadIdx.y + k * BLOCKSIZEY + j * BLOCKSIZEY * LOAD_COUNT;
          if (offset_smem + i * TILE_V < K && offset_smem < TILE_V) {
            #pragma unroll
            for (int n = 0; n < 8; n++) {
              sum[n] += (float)s_data[offset_smem] * (float)temp[k][n];
            }  
          }
        }
      }
    }
  }
  __syncthreads();

//================= REDUCE SUM ACROSS X BLOCK =======================     
  if (col < N && row < K) {    
    #pragma unroll
    for (int offset = BLOCKSIZEY / 2; offset > 0; offset >>= 1) {
      if(threadIdx.y >= offset && threadIdx.y < 2 * offset) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
          reduce_sum_data[(threadIdx.y - offset) * (BLOCKSIZEX * 8 + 1) + 8 * threadIdx.x + i] = sum[i];
        }
      }
      __syncthreads();
      if (threadIdx.y < offset) {
        #pragma unroll 
        for (int i = 0; i < 8; i++) {
          sum[i] += reduce_sum_data[threadIdx.y * (BLOCKSIZEX * 8 + 1) + 8 * threadIdx.x + i];
        }
      }
      __syncthreads();
    }
    __syncthreads();

//=================== WRITE THE RESULT INTO MATRIX C ===================
    if (threadIdx.y == 0) {
      #pragma unroll 
      for (int i = 0; i < 8; i++) {
        C[col+i] = (half)sum[i];
      }
    }
  }
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

// @@ Generate a half number from 0 to 4
float generateRandomhalf() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 4.0f -2.0f;
    // return 1.f;
}


int main(int argc, char **argv){
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
  size_t sizeA = M * K * sizeof(half);
  size_t sizeB = K * N * sizeof(half);
  size_t sizeC = M * N * sizeof(half);

  //@@ Allocate CPU memory here
  hostA = (half *)malloc(sizeA);
  hostB = (half *)malloc(sizeB);
  hostC = (half *)malloc(sizeC);
  hostCompareC = (half *)malloc(sizeC);

  //@@ Initialize matrix
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
        hostA[m * K + k] = (half)generateRandomhalf();
        // hostA[m * K + k] = (half)1.f;
    }
  }

  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
        hostB[k * N + n] = (half)generateRandomhalf();
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

  // CPU gloden func
  matrix_multiply_cpu(hostCompareC, hostA, hostB, M, N, K);
  //@@ Device Warmup here
  musaMemcpy(deviceWarmupA, hostA, sizeA, musaMemcpyHostToDevice);
  musaMemcpy(deviceWarmupA, hostB, sizeB, musaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid((N + BLOCKSIZEX * 8 - 1) / (BLOCKSIZEX * 8) , 1, 1);
  dim3 dimBlock(BLOCKSIZEX, BLOCKSIZEY, 1);  //1D

  //@@ Basic Kernel!!!!!!
  // dim3 dimGrid((N + BLOCKSIZEY - 1) / BLOCKSIZEY, 1, 1);
  // dim3 dimBlock(BLOCKSIZEY, 1, 1);

  //@@ Kernel warm up
  for (int i = 0; i < 3; ++i)
  matrix_multiply<half,BLOCKSIZEX,BLOCKSIZEY,LOADCOUNT,TILEV><<<dimGrid, dimBlock>>>(deviceWarmupC, deviceWarmupA, deviceWarmupB, M, N, K);
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
  for (int i = 0; i < 10; ++i) {
    matrix_multiply<half,BLOCKSIZEX,BLOCKSIZEY,LOADCOUNT,TILEV><<<dimGrid, dimBlock>>>(deviceC, deviceA, deviceB, M, N, K);
  }
  musaDeviceSynchronize();

  //@@ Time measuring
  auto end_kernel = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_kernel = end_kernel - start_kernel;
  std::cout << "Kernel execution time: " << elapsed_kernel.count()*1000000 / 10.f << " us" << std::endl;

  std::cout << "Performing musa computation" << std::endl;
  std::cout << "========================================" << std::endl;

  MUSA_CHECK(musaGetLastError());
  //@@ Copy the GPU memory back to the CPU here
  musaMemcpy(hostC, deviceC, sizeC, musaMemcpyDeviceToHost);
  //@@ Caculate bandwidth in GB/s
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
  
  //@@ Verification
  for (int i = 0; i < M * N; i++) {
        if (fabs((float)hostC[i] - (float)hostCompareC[i]) > EPSILON || isnan((float)hostC[i])) {
            fprintf(stderr, "Result not matched at element number %d: your output = %f :: actual output = %f\n", i, (float)hostC[i], (float)hostCompareC[i]);
            exit(EXIT_FAILURE);
        }
    }
  std::cout << "All matched!" << std::endl;
    
// //@@ Print cmp C
//   for (int i = 0; i < M; i++) {
//     for (int j = 0; j < N; j++) {
//       std::cout << (float)hostCompareC[i * N + j] << " ";
//     }
//     std::cout << std::endl;
//   }
// //@@ Print C
//   for (int i = 0; i < M; i++) {
//     for (int j = 0; j < N; j++) {
//       std::cout << (float)hostC[i * N + j] << " ";
//     }
//     std::cout << std::endl;
//   }

//@@ Free Host Memory
  free(hostA);
  free(hostB);
  free(hostC);


  return 0;
}
