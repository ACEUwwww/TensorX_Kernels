
#include "../wb.h"
#include "solution.h"
#include <iostream>
#include <chrono>
#include "musa_fp16.h"
#define TILE_WIDTH 256


//@@ GPU Kernel 
//@@ Performing Matrix multiplication
// __global__ void matrix_multiply(half *C, half *A, half *B, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
//     int col = threadIdx.x + blockDim.x * blockIdx.x;
//     if (col < numCColumns){
//       float sum = 0.0;
//       for (int k = 0; k < numAColumns; k++) {
//         sum += (float)(A[k] * B[k * numBColumns + col]);
//       }
//       C[col] = (half)sum;
//     }
// }

// CPU Kernel for test 
void matrix_multiply_cpu(half *C, half *A, half *B, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    for(int m = 0; m < numARows; m++) {
      for(int n = 0; n < numBColumns; n++) {
        float sum = 0;
        for (int k = 0; k < numAColumns; k++) {
          sum += ((float)A[m * numAColumns + k] * (float)B[k * numBColumns + n]);
        }
        C[m * numCColumns + n] = (half)sum;
      }
    }
    return;
}

int main(int argc, char **argv){
  wbArg_t args;
  half *hostA;
  half *hostB;
  half *hostC;
  half *deviceA;
  half *deviceB;
  half *deviceC;
  half *deviceWarmupA;
  half *deviceWarmupB;
  half *deviceWarmupC;
  int numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns;
  
  args = wbArg_read(argc, argv);

  hostA = (half *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB = (half *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

  numCRows = numARows;
  numCColumns = numBColumns;

  size_t sizeA = numARows * numAColumns * sizeof(half);
  size_t sizeB = numBRows * numBColumns * sizeof(half);
  size_t sizeC = numCRows * numCColumns * sizeof(half);
  hostC = (half *)malloc(sizeC);

  //@@ Allocate GPU memory here
  musaMalloc((void **)&deviceA, sizeA);
  musaMalloc((void **)&deviceB, sizeB);
  musaMalloc((void **)&deviceC, sizeC);

  // //@@ Device Warmup here
  // musaMemcpy(deviceWarmupInput1, hostInput1, inputLength * sizeof(half), musaMemcpyHostToDevice);
  // musaMemcpy(deviceWarmupInput2, hostInput2, inputLength * sizeof(half), musaMemcpyHostToDevice);

  // //@@ Initialize the grid and block dimensions here
  //@@ GEMV here. Consider only 1 dimension of output
  dim3 dimGrid((numCColumns + TILE_WIDTH -1) / TILE_WIDTH, 1, 1);
  dim3 dimBlock(TILE_WIDTH, 1, 1);  //1D


  // //@@ Kernel warm up
  // vecAdd<<<dimBlock,dimThreads>>>(deviceWarmupOutput, deviceWarmupInput1, deviceWarmupInput2, inputLength);
  // musaDeviceSynchronize();

  // musaFree(deviceWarmupInput1); musaFree(deviceWarmupInput2); musaFree(deviceWarmupOutput);

  //@@ Copy memory to the GPU here
  musaMemcpy(deviceA, hostA, sizeA, musaMemcpyHostToDevice);
  musaMemcpy(deviceB, hostB, sizeB, musaMemcpyHostToDevice);
  musaMemset(deviceC, 0xFFFF, sizeC);
  musaDeviceSynchronize();
  
  std::cout << "========================================" << std::endl;
  wbTime_start("Compute", "Performing musa computation");

  //@@ Launch the GPU Kernel here
  auto start_kernel = std::chrono::high_resolution_clock::now();
  // matrix_multiply<<<dimGrid, dimBlock>>>(deviceC, deviceA, deviceB, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  musaDeviceSynchronize();

  //@@ Time measuring
  auto end_kernel = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_kernel = end_kernel - start_kernel;
  std::cout << "Kernel execution time: " << elapsed_kernel.count()*1000000 << " microseconds" << std::endl;

  wbTime_stop("Compute", "Performing musa computation");
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
  // musaFree(deviceInput1); musaFree(deviceInput2); musaFree(deviceOutput);



  /* CPU test function */
  matrix_multiply_cpu(hostC, hostA, hostB, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      std::cout << (float)hostC[i * numCColumns + j] << " ";
    }
    std::cout << std::endl;
  }
  
  wbSolution(args, hostC, numCRows, numCColumns);
  free(hostA);
  free(hostB);
  free(hostC);


  return 0;
}