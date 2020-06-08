#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <math.h>


// Thread block size
#define MAT_MAX_SIZE 1024
#define BLOCKSIZE 32
#define ISPRINT false
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
// Matrix Structure declaration
typedef struct {
	//width of the matrix represented
	unsigned int width;
	//height of the matrix represented
	unsigned int height;
	//number of elements between the beginnings of adjacent
	// rows in the memory layout (useful for representing sub-matrices)
	unsigned int pitch;
	//Pointer to the first element of the matrix represented
	float* elements;
} Matrix;
int maxInt(int x, int y);
int compliteToFullTile(int number, int numberOfTiles);
bool isMetrixsNeedToBeFixedToSquered(const Matrix Mhost, const Matrix Nhost);
void expendMatrix(Matrix dest, Matrix src, bool isprint);
void reduceMatrix(Matrix dest, Matrix src, bool isprint);
Matrix fixMatrixDim(const Matrix candidate, int h, int w, bool isPrint);
int CompareData(float* reference, float* data, unsigned int len, float epsilon);
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);
int main()
{
	// Matrices for the program
	Matrix  M;
	Matrix  N;
	Matrix  P;
	// Number of elements in the solution matrix
	//  Assuming square matrices, so the sizes of M, N and P are equal
	srand(52);

	// Allocate and initialize the matrices
	int dummy = MAT_MAX_SIZE;
	//dummy = rand() % MAT_MAX_SIZE;
	int HM = (dummy == 0 ? 1 : dummy);
	//dummy = rand() % MAT_MAX_SIZE;
	int WM = (dummy == 0 ? 1 : dummy);
	M = AllocateMatrix(HM, WM, 1);
	//dummy = rand() % MAT_MAX_SIZE;
	int WN = (dummy == 0 ? 1 : dummy);
	N = AllocateMatrix(WM, WN, 1);
	int HP = HM;
	int WP = WN;
	P = AllocateMatrix(HM, WN, 0);
	unsigned int size_elements = WP * HP;
	int res;
	printf("-------------------------------------------before padding----------------------------------------------------------\n");
	printf("-----------------------------------------M--%d-X-%d---------------------------------------------------\n", HM, WM);
	printf("-----------------------------------------N--%d-X-%d---------------------------------------------------\n", WM, WN);

	printf("--------------------------------------------------------------------------------------------------------\n");

	// M * N on the device
	MatrixMulOnDevice(M, N, P);




	Matrix reference = AllocateMatrix(HM, WN, 0);
	computeGold(reference.elements, M.elements, N.elements, HM, WM, WN);

	// check if the device result is equivalent to the expected solution
	res = CompareData(reference.elements, P.elements, size_elements, 0.0001f);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	// Free host matrices
	free(M.elements);
	M.elements = NULL;
	free(N.elements);
	N.elements = NULL;
	free(P.elements);
	P.elements = NULL;

	return 0;
}


// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
	Matrix Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&Mdevice.elements, size));
	return Mdevice;
}

// Allocate a matrix of dimensions height*width
// If init == 0, initialize to all zeroes.  
// If init == 1, perform random initialization.
Matrix AllocateMatrix(int height, int width, int init)
{
	Matrix M;
	M.width = M.pitch = width;
	M.height = height;
	int size = M.width * M.height;
	M.elements = NULL;

	M.elements = (float*)malloc(size * sizeof(float));

	for (unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);

	}
	return M;
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	Mdevice.height = Mhost.height;
	Mdevice.width = Mhost.width;
	Mdevice.pitch = Mhost.pitch;
	cudaMemcpy(Mdevice.elements, Mhost.elements, size,
		cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size,
		cudaMemcpyDeviceToHost);
}

void computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{



	clock_t start, end;
	double cpu_time_used;

	start = clock();


	for (unsigned int i = 0; i < hA; ++i)
		for (unsigned int j = 0; j < wB; ++j) {
			double sum = 0;
			for (unsigned int k = 0; k < wA; ++k) {
				double a = A[i * wA + k];
				double b = B[k * wB + j];
				sum += a * b;
			}
			C[i * wB + j] = (float)sum;
		}
	end = clock();
	double extime = (double)(end - start) / CLOCKS_PER_SEC;
	printf("cpu time take %f seconds\n ", extime);
	if (ISPRINT == true) {
		printf("gold compute \n\n");
		printf("---------------------------------------------------------------------------\n");
		for (int hight = 0; hight < hA; hight++) {
			for (int width = 0; width < wB; width++) {
				int index = width + (hight * wB);
				printf(": %.2f ", C[index]);
			}
			printf("\n");
		}
		printf("\n\n");
		printf("---------------------------------------------------------------------------\n");
	}

}

Matrix fixMatrixDim(const Matrix candidate, int h, int w, bool isprint) {
	Matrix fixed = AllocateMatrix(h, w, 0);
	expendMatrix(fixed, candidate, isprint);
	return fixed;
}

void reduceMatrix(Matrix dest, Matrix src, bool isprint) {

	if (isprint) {
		printf("\n\n");
		printf("------------------------------before reduce----------------------------------------\n");
		for (int hight = 0; hight < src.height; hight++) {
			for (int width = 0; width < src.width; width++) {
				int index = width + (hight * src.width);
				printf(": %.2f ", src.elements[index]);
			}
			printf("\n");
		}
		printf("\n\n");
		printf("---------------------------------------------------------------------------\n");
	}

	for (int hight = 0; hight < dest.height; hight++) {
		for (int width = 0; width < dest.width; width++) {
			int indexdest = width + (hight * dest.width);
			int indexsrc = width + (hight * src.width);
			dest.elements[indexdest] = src.elements[indexsrc];
		}
	}
	if (isprint) {
		printf("\n\n");
		printf("-------------------------------after reduce-------------------------------------\n");
		for (int hight = 0; hight < dest.height; hight++) {
			for (int width = 0; width < dest.width; width++) {
				int index = width + (hight * dest.width);
				printf(": %.2f ", dest.elements[index]);
			}
			printf("\n");
		}
		printf("\n\n");
		printf("---------------------------------------------------------------------------\n");
	}



}


void expendMatrix(Matrix dest, Matrix src, bool isprint) {

	for (int hight = 0; hight < src.height; hight++) {
		for (int width = 0; width < src.width; width++) {
			int indexdest = width + (hight * dest.width);
			int indexsrc = width + (hight * src.width);
			dest.elements[indexdest] = src.elements[indexsrc];
		}
	}
	if (isprint) {
		printf("\n\n");
		printf("---------------------------------------------------------------------------\n");
		for (int hight = 0; hight < dest.height; hight++) {
			for (int width = 0; width < dest.width; width++) {
				int index = width + (hight * dest.width);
				printf(": %.2f ", dest.elements[index]);
			}
			printf("\n");
		}
		printf("\n\n");
		printf("---------------------------------------------------------------------------\n");
	}

}

bool IsPowerOfTwo(int x)
{
	return (x != 0) && ((x & (x - 1)) == 0);
}
bool isMetrixsNeedToBeFixedToSquered(const Matrix Mhost, const Matrix Nhost) {
	return !(Mhost.height == Nhost.width);
}

int maxInt(int x, int y)
{
	return x ^ ((x ^ y) & -(x < y));
}

int compliteToFullTile(int number, int numberOfTiles) {
	if (number % (numberOfTiles * BLOCKSIZE) == 0) {
		return number;
	}
	return ((numberOfTiles * BLOCKSIZE) - (number % (numberOfTiles * BLOCKSIZE))) + number;
}

template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(Matrix Mdevice, Matrix Ndevice, Matrix result) {

	int wA = Mdevice.width;
	int wB = Ndevice.width;
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = Mdevice.width * BLOCK_SIZE * by;


	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1 + BLOCK_SIZE;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * 2*bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * Ndevice.width;

	// Declaration of the shared memory array As used to
	// store the sub-matrix of A
	__shared__ double AsT1[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double BsT1[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double BsT2[BLOCK_SIZE][BLOCK_SIZE];

	// Load first tile into registers
	double mItem = Mdevice.elements[aBegin + wA * ty + tx];
	double nItem1 = Ndevice.elements[bBegin + wB * ty + tx];
	double nItem2 = Ndevice.elements[bBegin + wB * ty + tx + BLOCK_SIZE];
	__syncthreads();

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	double Csub1 = 0;
	double Csub2 = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin + aStep, b = bBegin + bStep;
		a <= aEnd;
		a += aStep, b += bStep) {

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		// Deposit registers into shared memory
		AsT1[ty][tx] = mItem;
		BsT1[ty][tx] = nItem1;
		BsT2[ty][tx] = nItem2;


		// Synchronize to make sure the matrices are loaded
		__syncthreads();
		// Load next tile into registers
		mItem = Mdevice.elements[a + wA * ty + tx];
		nItem1 = Ndevice.elements[b + wB * ty + tx];
		nItem2 = Ndevice.elements[b + wB * ty + tx + BLOCK_SIZE];

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll

		// Accumulate dot product
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub1 += AsT1[ty][k] * BsT1[k][tx];
			Csub2 += AsT1[ty][k] * BsT2[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	//---------------------------------------------
	//------------------------------------------
	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c1 = wB * BLOCK_SIZE * by + BLOCK_SIZE * 2 * bx;
	int c2 = wB * BLOCK_SIZE * by + BLOCK_SIZE * 2 * bx + BLOCK_SIZE;

	result.elements[c1 + wB * ty + tx] = Csub1;
	result.elements[c2 + wB * ty + tx] = Csub2;
}


void MatrixMulOnDevice(const Matrix Mhost, const Matrix Nhost, Matrix Phost)
{
	bool isFixed = false;
	bool isPrint = ISPRINT;
	cudaEvent_t start, stop;
	//open timers
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&stop));
	Matrix _Mhost;
	Matrix _Nhost;
	Matrix squredPhost;
	int dimMaxHight = compliteToFullTile(Mhost.height, 1);

	int dimMaxWidth = compliteToFullTile(Nhost.width, 2);

	printf("print _Mhost \n");
	_Mhost = fixMatrixDim(Mhost, dimMaxHight, compliteToFullTile(Mhost.width, 1), isPrint);
	printf("print _Nhost \n");
	_Nhost = fixMatrixDim(Nhost, compliteToFullTile(Nhost.height, 1), dimMaxWidth, isPrint);

	printf("print squredPhost \n");
	squredPhost = AllocateMatrix(dimMaxHight, dimMaxWidth, 0);

	printf("-------------------------------------------after padding----------------------------------------------------------\n");
	printf("-----------------------------------------M--%d-X-%d---------------------------------------------------\n", _Mhost.height, _Mhost.width);
	printf("-----------------------------------------N--%d-X-%d---------------------------------------------------\n", _Nhost.height, _Nhost.width);
	printf("-----------------------------------------P--%d-X-%d---------------------------------------------------\n", squredPhost.height, squredPhost.width);

	printf("--------------------------------------------------------------------------------------------------------\n");
	printf("--------------------------------------------------------------------------------------------------------\n");
	//device == gpu
	Matrix Mdevice = AllocateDeviceMatrix(_Mhost);
	Matrix Ndevice = AllocateDeviceMatrix(_Nhost);
	Matrix Pdevice = AllocateDeviceMatrix(squredPhost);
	//(N->(n_row,n_col) * (M->m_row,m_col) = P->(n_row * m_col)

	CopyToDeviceMatrix(Mdevice, _Mhost);
	CopyToDeviceMatrix(Ndevice, _Nhost);

	gpuErrchk(cudaEventRecord(start));
	dim3 threads(BLOCKSIZE, BLOCKSIZE);
	dim3 grid(Nhost.width / (2*threads.x) + 1, (Mhost.height / threads.y) + 1);
	MatrixMulCUDA<BLOCKSIZE> << < grid, threads >> > (Mdevice, Ndevice, Pdevice);

	gpuErrchk(cudaEventRecord(stop));
	gpuErrchk(cudaEventSynchronize(stop));
	CopyFromDeviceMatrix(squredPhost, Pdevice);

	float msecTotal = 0.0f;
	gpuErrchk(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("Time GPU= % .3f \n", msecTotal / 1000);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaFree(Mdevice.elements));
	gpuErrchk(cudaFree(Ndevice.elements));
	gpuErrchk(cudaFree(Pdevice.elements));

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	printf("print Phost \n");
	reduceMatrix(Phost, squredPhost, isPrint);
	free(squredPhost.elements);
	squredPhost.elements = NULL;

}



int  CompareData(float* reference, float* data, unsigned int len, float epsilon)
{
	int result = 1;
	float diff;
	int comp;

	for (unsigned int i = 0; i < len; ++i) {

		diff = reference[i] - data[i];
		comp = (diff <= epsilon) && (diff >= -epsilon);
		result &= comp;
	}

	return (result);
}

