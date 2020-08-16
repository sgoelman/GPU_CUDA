

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <math.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// Thread block size
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
double computeGold(float* reference, float* idata, const unsigned int len);


#define DEFAULT_MATRIX_SIZE 1024
#define DEFAULT_THREADS_PER_BLOCK 128

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runSequentialTest(float* A, float* L, const unsigned int dimensionSize);
float runCUDATest_NormalS(float* h_Adata, float* h_Ldata, const unsigned int dimensionSize,
    const int threads_per_block, const int cutoff);
float runCUDATest_NormalM(float* h_Adata, float* h_Ldata, const unsigned int dimensionSize,
    const int threads_per_block, const int cutoff);
float runCUDATest_InPlaceM(float* h_Adata, float* h_Ldata, const unsigned int dimensionSize,
    const int threads_per_block, const int cutoff);
float runCuSolverTest(float* h_Adata, float* h_Ldata, const unsigned int dimensionSize);
void writeResultToFile(char* name, float* contentGPU, double* contentCPU, int LIST_SIZE);
float computeSyncSingleKarnelOneBlock(float* h_Adata, float* h_Ldata, const unsigned int dimensionSize, const int threads_per_block);

////////////////////////////////////////////////////////////////////////////////
// matrix helper functions
float* spd_create_symetricf(unsigned int dimension, float minValue, float maxValue);
float* spd_make_positive_definitef(float* A, unsigned int dimension, float offset);
float* spd_create_blankf(unsigned int dimension);
float spd_random_float(float fMin, float fMax);
void spd_print_matrixf(float* A, unsigned int dimension, int count);
int spd_compare_matricesf(float* A, float* B, int dimension, float epsilon);
void spd_free_matrixf(float* A);
float* transpose(float* h_Adata, int dimensionSize);
////////////////////////////////////////////////////////////////////////////////
//! Cholesky Kernel for a single column. Normal Single Kernel
//! @param A              input data in global memory
//! @param L              output data in global memory
//! @param dimensionSize  width of matrices
//! @param col            current column
////////////////////////////////////////////////////////////////////////////////
__global__ void choleskyKernel_NormalS(float* A, float* L, int dimensionSize, int col){
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = col + tid;
    int k;
    float sum = 0;
    float value;
    float sum_d = 0;

    if (tid == 0) {
        // diagonal
        for (k = 0; k < col; k++) {
            sum_d += L[col * dimensionSize + k] * L[col * dimensionSize + k];
        }
        L[col * dimensionSize + col] = sqrtf(A[col * dimensionSize + col] - sum_d);
    }
    else {
        // other elements
        if (row < dimensionSize) {
            for (k = 0; k < col; k++) {
                sum += L[row * dimensionSize + k] * L[col * dimensionSize + k];
                sum_d += L[col * dimensionSize + k] * L[col * dimensionSize + k];
            }
            value = sqrt(A[col * dimensionSize + col] - sum_d);

            L[row * dimensionSize + col] = (1.0 / value * (A[row * dimensionSize + col] - sum));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Cholesky Kernels for a single column. Normal Multiple Kernel
//! @param A              input data in global memory
//! @param L              output data in global memory
//! @param dimensionSize  width of matrices
//! @param col            current column
////////////////////////////////////////////////////////////////////////////////
__global__ void
choleskyKernel_NormalM1(float* A, float* L, int dimensionSize, int col)
{
    int k;
    float sum_d = 0;

    // diagonal
    for (k = 0; k < col; k++) {
        sum_d += L[col * dimensionSize + k] * L[col * dimensionSize + k];
    }
    L[col * dimensionSize + col] = sqrtf(A[col * dimensionSize + col] - sum_d);
}
__global__ void
choleskyKernel_NormalM2(float* A, float* L, int dimensionSize, int col)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = col + tid + 1;
    int k;
    float sum = 0;

    // other elements
    if (row < dimensionSize) {
        for (k = 0; k < col; k++) {
            sum += L[row * dimensionSize + k] * L[col * dimensionSize + k];
        }

        L[row * dimensionSize + col] = (1.0 / L[col * dimensionSize + col] *
            (A[row * dimensionSize + col] - sum));
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Cholesky Kernels for a single column. In-Place Multiple Kernel
//! @param A              input/output data in global memory
//! @param dimensionSize  width of matrices
//! @param col            current column
////////////////////////////////////////////////////////////////////////////////
__global__ void
choleskyKernel_InPlaceM1(float* A, int dimensionSize, int col)
{
    int k;
    float sum_d = 0;

    // diagonal
    for (k = 0; k < col; k++) {
        sum_d += A[col * dimensionSize + k] * A[col * dimensionSize + k];
    }
    A[col * dimensionSize + col] = sqrtf(A[col * dimensionSize + col] - sum_d);
}
__global__ void choleskyKernel_InPlaceM2(float* A, int dimensionSize, int col)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = col + tid + 1;
    int k;
    float sum = 0;

    // other elements
    if (row < dimensionSize) {
        for (k = 0; k < col; k++) {
            sum += A[row * dimensionSize + k] * A[col * dimensionSize + k];
        }

        A[row * dimensionSize + col] = (1.0 / A[col * dimensionSize + col] *
            (A[row * dimensionSize + col] - sum));
    }
}

template <int BLOCK_SIZE> __global__ void chol_kernel_one_block(float* U, unsigned int num_rows)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int tx = tid + bid ;
    unsigned int i, j, k;
    for (k = 0; k < num_rows; k++)
    {
        if (tx == 0)
        {
            U[k * num_rows + k] = sqrt(U[k * num_rows + k]);
            for (j = (k + 1); j < num_rows; j++)
            {
                U[k * num_rows + j] /= U[k * num_rows + k];
            }
        }
        __syncthreads();

        for (i = (k + 1) + bid  + tid; i < num_rows; i += BLOCK_SIZE )
        {
            for (j = i; j < num_rows; j++)
            {
                U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
            }
        }
        __syncthreads();
    }
    __syncthreads();

    for (i = bid  + tid; i < num_rows; i += BLOCK_SIZE )
    {
        for (j = 0; j < i; j++)
            U[i * num_rows + j] = 0.0;
    }
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    
    int LIST_SIZE = 8;
    char* name = (char*)malloc(20 * sizeof(char));
    // Read from command line: size, algorithm (Sequential or CUDA), device
    float* timersGPU = (float*)malloc(LIST_SIZE * sizeof(float));
    double* timersCPU = (double*)malloc(LIST_SIZE * sizeof(double));
    unsigned int algorithm = (argc >= 3) ?
        atoi(argv[2]) :
        0;
    unsigned int threads_per_block = (argc >= 4) ?
        atoi(argv[3]) :
        DEFAULT_THREADS_PER_BLOCK;
    unsigned int cutoff = (argc >= 5) ?
        atoi(argv[4]) :
        0;
    unsigned int deviceId = (argc >= 6) ?
        atoi(argv[5]) :
        0;

    // consistency of inputs
    if ((algorithm == 0 || algorithm == 4) &&
        (threads_per_block != DEFAULT_THREADS_PER_BLOCK || cutoff != 0))
        return 0;

    // check if tpb and max blocks are compatible with device
   
    for (int index = 0, unsigned int dimensionSize = 32; dimensionSize <= 4096; index++, dimensionSize *= 2) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, deviceId);
        if (threads_per_block > devProp.maxThreadsPerBlock ||
            (ceil((float)(dimensionSize) / (float)threads_per_block) > devProp.maxThreadsDim[0]))
            return 0;

        if (cutoff >= dimensionSize)
            return 0; // if cutoff is greater or equals than the input size, cancel execution
      // allocate and initialize host memory for input and output
    float* h_Adata = spd_create_symetricf(dimensionSize, 1, 100);
    spd_make_positive_definitef(h_Adata, dimensionSize, 50000);
    float* h_Ldata = spd_create_blankf(dimensionSize);
    
    // run test, depending on algorithm
    switch (algorithm) {
        // Sequential
    case 0:
        name = "Sequential";
        printf("%d,Sequential,", dimensionSize);
        runSequentialTest(h_Adata, h_Ldata, dimensionSize);
        break;

        // CUDA Normal Single Kernel
    case 1:
        name = "SingleKarnel";
        printf("%d,CUDA_NormalS,%d,%d,", dimensionSize, threads_per_block, cutoff);
        timersGPU[index] =  runCUDATest_NormalS(h_Adata, h_Ldata, dimensionSize, threads_per_block, cutoff);
        break;

        // CUDA Normal Multiple Kernels
    case 2:
        name = "MultiKarnel";
        printf("%d,CUDA_NormalM,%d,%d,", dimensionSize, threads_per_block, cutoff);
        timersGPU[index] = runCUDATest_NormalM(h_Adata, h_Ldata, dimensionSize, threads_per_block, cutoff);
        break;

        // CUDA InPlace Multiple Kernels
    case 3:
        name = "InPlaceMultiKarnel";
        printf("%d,CUDA_InPlaceM,%d,%d,", dimensionSize, threads_per_block, cutoff);
        timersGPU[index] = runCUDATest_InPlaceM(h_Adata, h_Ldata, dimensionSize, threads_per_block, cutoff);
        break;

        // CuSolver
    case 4:
        name = "CUSOLVER";
        printf("%d,CUSOLVER,", dimensionSize);
        timersGPU[index] = runCuSolverTest(h_Adata, h_Ldata, dimensionSize);
        break;

    case 5:
        name = "SyncChols";
        printf("%d,SyncChols,%d,%d,", dimensionSize, threads_per_block, cutoff);
        timersGPU[index] = computeSyncSingleKarnelOneBlock(h_Adata, h_Ldata, dimensionSize, threads_per_block);
        h_Ldata = transpose(h_Ldata, dimensionSize);

        break;
        break;
    }


    // compute reference solution
    float* h_LGdata = spd_create_blankf(dimensionSize);
    timersCPU[index] = computeGold(h_Adata, h_LGdata, dimensionSize);
    printf("Input Matrix:\n");
    spd_print_matrixf(h_Adata, dimensionSize, 16);
    printf("Gold Matrix:\n");
    spd_print_matrixf(h_LGdata, dimensionSize, 16);
    printf("GPU Output Matrix:\n");
    spd_print_matrixf(h_Ldata, dimensionSize, 16);
    printf("Comparing ... ");
    spd_compare_matricesf(h_Ldata, h_LGdata, dimensionSize, 0.0001);
    spd_free_matrixf(h_LGdata);


    // free matrices
    spd_free_matrixf(h_Adata);
    spd_free_matrixf(h_Ldata);

}
    writeResultToFile(name, timersGPU, timersCPU, LIST_SIZE);
    // exit
    exit(EXIT_SUCCESS);
}

void writeResultToFile(char* name, float* contentGPU, double* contentCPU,int LIST_SIZE) {
    FILE* f = fopen(name, "a");
    for (int i = 0; i < LIST_SIZE; i++) {
        fprintf(f, "%f, %0.8f \n", contentGPU[i], contentCPU[i]);
    }
    fprintf(f, "\n");
    fclose(f);
}


float* transpose(float* h_Adata,int dimensionSize) {
    float* elements = (float*)malloc(dimensionSize * dimensionSize * sizeof(float));

    for (int i = 0; i < dimensionSize; i++)
        for (int j = 0; j < dimensionSize; j++)
            elements[i * dimensionSize + j] = h_Adata[j * dimensionSize + i];
    spd_free_matrixf(h_Adata);
    return elements;
}

////////////////////////////////////////////////////////////////////////////////
//! Run Tequential test
////////////////////////////////////////////////////////////////////////////////
void runSequentialTest(float* A, float* L, const unsigned int dimensionSize)
{
    // initialize timer
    clock_t start, end;
    double cpu_time_used;
    double total_sum = 0;
    start = clock();

    int i, j, k;
    float sum;
    for (j = 0; j < dimensionSize; j++) {
        sum = 0;
        for (k = 0; k < j; k++) {
            sum += L[j * dimensionSize + k] * L[j * dimensionSize + k];
        }
        L[j * dimensionSize + j] = sqrt(A[j * dimensionSize + j] - sum);

        for (i = j + 1; i < dimensionSize; i++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum += L[i * dimensionSize + k] * L[j * dimensionSize + k];
            }
            L[i * dimensionSize + j] = (1.0 / L[j * dimensionSize + j] *
                (A[i * dimensionSize + j] - sum));
        }
    }

    // stop timer
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("%0.8f\n", cpu_time_used);
}

////////////////////////////////////////////////////////////////////////////////
//! Run CUDA test. Normal Single Kernel
////////////////////////////////////////////////////////////////////////////////
float runCUDATest_NormalS(float* h_Adata, float* h_Ldata, const unsigned int dimensionSize,
    const int threads_per_block, const int cutoff)
{
    // set device id
    // initialize timer
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start));

    // allocate device memory ...
    unsigned int mem_size = sizeof(float) * dimensionSize * dimensionSize;
    // ... input
    float* d_Adata;
    gpuErrchk(cudaMalloc((void**)&d_Adata, mem_size));
    // copy host memory to device
    gpuErrchk(cudaMemcpy(d_Adata, h_Adata, mem_size,
        cudaMemcpyHostToDevice));
    // ... output
    float* d_Ldata;
    gpuErrchk(cudaMalloc((void**)&d_Ldata, mem_size));

    // execute the kernels
    int j;
    int num_blocks;
    for (j = 0; j < dimensionSize - cutoff; j++) {
        num_blocks = ceil((float)(dimensionSize - j) / (float)threads_per_block);
        choleskyKernel_NormalS <<< num_blocks, threads_per_block >> > (d_Adata, d_Ldata, dimensionSize, j);
    }

    // check if kernel execution generated and error

    // copy result from device to host
    gpuErrchk(cudaMemcpy(h_Ldata, d_Ldata, mem_size,
        cudaMemcpyDeviceToHost));

    // Sequenial part (based on cutoff)
    float sum;
    int i, k;
    for (j = dimensionSize - cutoff; j < dimensionSize; j++) {
        // Diagonal value
        sum = 0;
        for (k = 0; k < j; k++) {
            sum += h_Ldata[j * dimensionSize + k] * h_Ldata[j * dimensionSize + k];
        }
        h_Ldata[j * dimensionSize + j] = sqrt(h_Adata[j * dimensionSize + j] - sum);

        // Calculate all other rows
        for (i = j + 1; i < dimensionSize; i++) {   // for each row below main diagonal
            sum = 0;
            for (k = 0; k < j; k++) {
                sum += h_Ldata[i * dimensionSize + k] * h_Ldata[j * dimensionSize + k];
            }
            h_Ldata[i * dimensionSize + j] = (1.0 / h_Ldata[j * dimensionSize + j] *
                (h_Adata[i * dimensionSize + j] - sum));
        }
    }

    // stop timer
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&msecTotal, start, stop));
    float timerResoultList = msecTotal / 1000;
    printf("%0.8f\n", timerResoultList);

    // cleanup memory
    gpuErrchk(cudaFree(d_Adata));
    gpuErrchk(cudaFree(d_Ldata));
    return timerResoultList;
}

////////////////////////////////////////////////////////////////////////////////
//! Run CUDA test. Normal Multiple Kernel
////////////////////////////////////////////////////////////////////////////////
float runCUDATest_NormalM(float* h_Adata, float* h_Ldata, const unsigned int dimensionSize, const int threads_per_block, const int cutoff){
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start));

    // allocate device memory ...
    unsigned int mem_size = sizeof(float) * dimensionSize * dimensionSize;
    // ... input
    float* d_Adata;
    gpuErrchk(cudaMalloc((void**)&d_Adata, mem_size));
    // copy host memory to device
    gpuErrchk(cudaMemcpy(d_Adata, h_Adata, mem_size,
        cudaMemcpyHostToDevice));
    // ... output
    float* d_Ldata;
    gpuErrchk(cudaMalloc((void**)&d_Ldata, mem_size));

    // execute the kernels
    int j;
    int num_blocks;

    if (cutoff > 0) {
        // some processing will be on host
        for (j = 0; j < dimensionSize - cutoff; j++) {
            num_blocks = ceil((float)(dimensionSize - j) / (float)threads_per_block);
            choleskyKernel_NormalM1 <<< 1, 1 >> > (d_Adata, d_Ldata, dimensionSize, j);
            choleskyKernel_NormalM2 <<< num_blocks, threads_per_block >> > (d_Adata, d_Ldata, dimensionSize, j);
        }
    }
    else {
        // cutoff = 0, all processing will be on GPU
        for (j = 0; j < dimensionSize - 1; j++) {
            num_blocks = ceil((float)(dimensionSize - j) / (float)threads_per_block);
            choleskyKernel_NormalM1 <<< 1, 1 >> > (d_Adata, d_Ldata, dimensionSize, j);
            choleskyKernel_NormalM2 <<< num_blocks, threads_per_block >> > (d_Adata, d_Ldata, dimensionSize, j);
        }
        choleskyKernel_NormalM1 <<< 1, 1 >> > (d_Adata, d_Ldata, dimensionSize, j);
    }

    // check if kernel execution generated and error

    // copy result from device to host
    gpuErrchk(cudaMemcpy(h_Ldata, d_Ldata, mem_size,
        cudaMemcpyDeviceToHost));

    // Sequenial part (based on cutoff)
    float sum;
    int i, k;
    for (j = dimensionSize - cutoff; j < dimensionSize; j++) {
        // Diagonal value
        sum = 0;
        for (k = 0; k < j; k++) {
            sum += h_Ldata[j * dimensionSize + k] * h_Ldata[j * dimensionSize + k];
        }
        h_Ldata[j * dimensionSize + j] = sqrt(h_Adata[j * dimensionSize + j] - sum);

        // Calculate all other rows
        for (i = j + 1; i < dimensionSize; i++) {   // for each row below main diagonal
            sum = 0;
            for (k = 0; k < j; k++) {
                sum += h_Ldata[i * dimensionSize + k] * h_Ldata[j * dimensionSize + k];
            }
            h_Ldata[i * dimensionSize + j] = (1.0 / h_Ldata[j * dimensionSize + j] *
                (h_Adata[i * dimensionSize + j] - sum));
        }
    }

    // stop timer
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&msecTotal, start, stop));
    float timerResoultList = msecTotal / 1000;
    printf("%0.8f\n", timerResoultList);


    // cleanup memory
    gpuErrchk(cudaFree(d_Adata));
    gpuErrchk(cudaFree(d_Ldata));
    return timerResoultList;
}

float computeSyncSingleKarnelOneBlock(float* h_Adata, float* h_Ldata, const unsigned int dimensionSize ,const int threads_per_block) {
    float* d_Adata;

    cudaEvent_t start, stop;

    unsigned int mem_size = sizeof(float) * dimensionSize * dimensionSize;
    gpuErrchk(cudaMalloc((void**)&d_Adata, mem_size));
    gpuErrchk(cudaMemcpy(d_Adata, h_Adata, mem_size,
        cudaMemcpyHostToDevice));


    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start));
    //Operations per thread
    int num_blocks = ceil((float)(dimensionSize) / (float)threads_per_block);

   //float ops_per_thread = dimensionSize / (threads_per_block * num_blocks);

    dim3 thread_block(threads_per_block, 1, 1);
    dim3 grid(num_blocks, 1);

    chol_kernel_one_block<16><<<grid, thread_block >> > (d_Adata,dimensionSize);
    cudaDeviceSynchronize();
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&msecTotal, start, stop));
    float timerResoultList = msecTotal / 1000;

    // copy result from device to host
    gpuErrchk(cudaMemcpy(h_Ldata, d_Adata, mem_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_Adata));
    return timerResoultList;
}
////////////////////////////////////////////////////////////////////////////////
//! Run CUDA test. In-Place Multiple Kernel
////////////////////////////////////////////////////////////////////////////////
float runCUDATest_InPlaceM(float* h_Adata, float* h_Ldata, const unsigned int dimensionSize, const int threads_per_block, const int cutoff){
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start));

    // allocate device memory ...
    unsigned int mem_size = sizeof(float) * dimensionSize * dimensionSize;
    // ... input/output
    float* d_Adata;
    gpuErrchk(cudaMalloc((void**)&d_Adata, mem_size));
    // copy host memory to device
    gpuErrchk(cudaMemcpy(d_Adata, h_Adata, mem_size,
        cudaMemcpyHostToDevice));

    // execute the kernels
    int j;
    int num_blocks;
    if (cutoff > 0) {
        for (j = 0; j < dimensionSize - cutoff; j++) {
            num_blocks = ceil((float)(dimensionSize - j) / (float)threads_per_block);
            choleskyKernel_InPlaceM1 <<< 1, 1 >> > (d_Adata, dimensionSize, j);
            choleskyKernel_InPlaceM2 << < num_blocks, threads_per_block >> > (d_Adata, dimensionSize, j);
        }
    }
    else {
        for (j = 0; j < dimensionSize - 1; j++) {
            num_blocks = ceil((float)(dimensionSize - j) / (float)threads_per_block);
            choleskyKernel_InPlaceM1 <<< 1, 1 >> > (d_Adata, dimensionSize, j);
            choleskyKernel_InPlaceM2 <<< num_blocks, threads_per_block >> > (d_Adata, dimensionSize, j);
        }
        choleskyKernel_InPlaceM1 <<< 1, 1 >> > (d_Adata, dimensionSize, j);
    }


    // copy result from device to host
    gpuErrchk(cudaMemcpy(h_Ldata, d_Adata, mem_size,
        cudaMemcpyDeviceToHost));

    // reset rest of matrix
    int i;
    for (i = 0; i < dimensionSize; i++) {
        for (j = 0; j < i; j++) {
            h_Ldata[j * dimensionSize + i] = 0;
        }
    }

    // Sequenial part (based on cutoff)
    float sum;
    int k;
    for (j = dimensionSize - cutoff; j < dimensionSize; j++) {
        // Diagonal value
        sum = 0;
        for (k = 0; k < j; k++) {
            sum += h_Ldata[j * dimensionSize + k] * h_Ldata[j * dimensionSize + k];
        }
        h_Ldata[j * dimensionSize + j] = sqrt(h_Ldata[j * dimensionSize + j] - sum);

        // Calculate all other rows
        for (i = j + 1; i < dimensionSize; i++) {   // for each row below main diagonal
            sum = 0;
            for (k = 0; k < j; k++) {
                sum += h_Ldata[i * dimensionSize + k] * h_Ldata[j * dimensionSize + k];
            }
            h_Ldata[i * dimensionSize + j] = (1.0 / h_Ldata[j * dimensionSize + j] *
                (h_Ldata[i * dimensionSize + j] - sum));
        }
    }

    // stop timer
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&msecTotal, start, stop));
    float timerResoultList = msecTotal / 1000;
    printf("%0.8f\n", timerResoultList );

    // cleanup memory
    gpuErrchk(cudaFree(d_Adata));
    return timerResoultList;
}

////////////////////////////////////////////////////////////////////////////////
//! Run cuSolver test
////////////////////////////////////////////////////////////////////////////////
float runCuSolverTest(float* h_Adata, float* h_Ldata, const unsigned int dimensionSize)
{


    // initialize timer
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start));


    // allocate device memory ...
    unsigned int mem_size = sizeof(float) * dimensionSize * dimensionSize;
    // ... input
    float* d_Adata;
    gpuErrchk(cudaMalloc((void**)&d_Adata, mem_size));
    // copy host memory to device
    gpuErrchk(cudaMemcpy(d_Adata, h_Adata, mem_size,
        cudaMemcpyHostToDevice));

    // init cusolver varialbes
    int work_size = 0;
    int* devInfo;
    gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));

    // initialize cusolver handle
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    // initialize Spotrf
    cusolverDnSpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, dimensionSize, d_Adata, dimensionSize, &work_size);

    // execute Cholesky on device (potrf)
    float* work;
    gpuErrchk(cudaMalloc(&work, work_size * sizeof(float)));
    cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, dimensionSize, d_Adata, dimensionSize, work, work_size, devInfo);
    int devInfo_h = 0;
    gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0) {
        printf("Unsuccessful potrf execution\n\n");
    }
    // copy result from device to host (copy to output)
    gpuErrchk(cudaMemcpy(h_Ldata, d_Adata, mem_size,
        cudaMemcpyDeviceToHost));
    // reset rest of matrix
    int i, j;
    for (i = 0; i < dimensionSize; i++) {
        for (j = 0; j < i; j++) {
            h_Ldata[j * dimensionSize + i] = 0;
        }
    }

    // destroy cuSolver handle
    cusolverDnDestroy(solver_handle);

    // stop timer
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&msecTotal, start, stop));
    float timerResoultList = msecTotal / 1000;
    printf("%0.8f\n", timerResoultList );
    // cleanup memory
    gpuErrchk(cudaFree(d_Adata));
    return timerResoultList;
}

////////////////////////////////////////////////////////////////////////////////
// matrix helper functions
float* spd_create_symetricf(unsigned int dimension, float minValue, float maxValue)
{
    float* m = (float*)calloc(dimension * dimension, sizeof(float));
    unsigned int i, j;
    for (i = 0; i < dimension; i++) {
        for (j = 0; j <= i; j++) {
            m[i * dimension + j] = spd_random_float(minValue, maxValue);
            m[j * dimension + i] = m[i * dimension + j];
        }
    }
    return m;
}
float* spd_make_positive_definitef(float* A, unsigned int dimension, float offset)
{
    unsigned int i;

    for (i = 0; i < dimension; i++) // A = A + n*I(n);
        A[i * dimension + i] = A[i * dimension + i] + offset;

    return A;
}
float* spd_create_blankf(unsigned int dimension)
{
    float* m = (float*)calloc(dimension * dimension, sizeof(float));
    unsigned int i;
    for (i = 0; i < dimension * dimension; i++)
        m[i] = 0;
    return m;
}
float spd_random_float(float fMin, float fMax)
{
    float f = (float)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
void spd_print_matrixf(float* A, unsigned int dimension, int count)
{
    unsigned int i, j;
    if (dimension < count)
        count = dimension;

    for (i = 0; i < count; i++)
    {
        for (j = 0; j < count; j++)
        {
            printf("%0.2f\t", A[i * dimension + j]);
        }
        printf("\n");
    }
}
int spd_compare_matricesf(float* A, float* B, int dimension, float epsilon)
{
    int correct = 1;
    int errors = 0;
    int i, j;
    for (i = 0; i < dimension; i++) {
        for (j = 0; j < dimension; j++) {
            if (fabs(A[i * dimension + j] - B[i * dimension + j]) > epsilon) {
                if (correct)
                    printf("  (%d, %d): %0.5f != %0.5f\n", i, j, A[i * dimension + j], B[i * dimension + j]);
                correct = 0;
                errors++;
            }
        }
    }
    printf("  Total errors: %d\n", errors);
    return errors;
}
void spd_free_matrixf(float* A)
{
    free(A);
}




double computeGold(float* A, float* L, const unsigned int dimensionSize)
{
    clock_t start, end;
    double cpu_time_used;
    double total_sum = 0;
    start = clock();
    int i, j, k;
    float sum;
    for (j = 0; j < dimensionSize; j++) {
        sum = 0;
        for (k = 0; k < j; k++) {
            sum += L[j * dimensionSize + k] * L[j * dimensionSize + k];
        }
        L[j * dimensionSize + j] = sqrt(A[j * dimensionSize + j] - sum);

        for (i = j + 1; i < dimensionSize; i++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum += L[i * dimensionSize + k] * L[j * dimensionSize + k];
            }
            L[i * dimensionSize + j] = (1.0 / L[j * dimensionSize + j] *
                (A[i * dimensionSize + j] - sum));
        }
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("%0.8f\n", cpu_time_used);
    return cpu_time_used;
}
