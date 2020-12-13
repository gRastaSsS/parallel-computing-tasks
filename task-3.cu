
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

#define ROWS 16
#define COLUMNS 3


int rand_int(int fMin, int fMax)
{
	return fMin + (std::rand() % (fMax - fMin + 1));
}


void printMatrix(int* A, int rows, int columns)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
			std::cout << " " << A[i*columns+j];
	
		std::cout << std::endl;
	}
}

__global__ void calculateCloseness(const int *A, int *D)
{
	int row0 = (blockIdx.x * blockDim.x) + threadIdx.x;
	int row1 = (blockIdx.y * blockDim.y) + threadIdx.y;
	int result = 0;

	for (int i = 0; i < COLUMNS; i++)
	{
		int val0 = A[row0 * COLUMNS + i];
		int val1 = A[row1 * COLUMNS + i];
		result += (val0 - val1) * (val0 - val1);
	}
	
	D[row0 * ROWS + row1] = result;
}

void clean(int* dev_A, int* dev_D)
{
	cudaFree(dev_A);
	cudaFree(dev_D);
}

cudaError_t runClosenessCalculation(int *A, int *D)
{
	cudaError_t cudaStatus;
	int *dev_A = 0;
	int *dev_D = 0;

	// Choose which GPU to run on, change this on a multi-GPU system
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
		clean(dev_A, dev_D);
		return cudaStatus;
	}

	// Allocate GPU buffers
	cudaStatus = cudaMalloc((void**)&dev_A, ROWS * COLUMNS * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc failed!";
		clean(dev_A, dev_D);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_D, ROWS * ROWS * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc failed!";
		clean(dev_A, dev_D);
		return cudaStatus;
	}

	// Copy input from host memory to GPU buffers
	cudaStatus = cudaMemcpy(dev_A, A, ROWS * COLUMNS * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc failed!";
		clean(dev_A, dev_D);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_D, D, ROWS * ROWS * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMemcpy failed!";
		clean(dev_A, dev_D);
		return cudaStatus;
	}

	dim3 blockDim(16, 16);
	dim3 gridDim(ROWS / 16, ROWS / 16);

	calculateCloseness<<<gridDim, blockDim>>>(dev_A, dev_D);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "calculateCloseness launch failed";
		//fprintf(stdout, "calculateCloseness launch failed: %s\n", cudaGetErrorString(cudaStatus));
		clean(dev_A, dev_D);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceSynchronize";
		clean(dev_A, dev_D);
		return cudaStatus;
	}

	// Copy output from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(D, dev_D, ROWS * ROWS * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMemcpy failed!";
		clean(dev_A, dev_D);
		return cudaStatus;
	}

	return cudaStatus;
}


int main()
{
	int A[ROWS * COLUMNS];
	int D[ROWS * ROWS];

	for (int w = 0; w < ROWS; w++)
		for (int h = 0; h < COLUMNS; h++)
			A[w*COLUMNS + h] = rand_int(0, 10);

	std::cout << "A" << std::endl;
	printMatrix(A, ROWS, COLUMNS);
	std::cout << std::endl;

	cudaError_t cudaStatus = runClosenessCalculation(A, D);

	if (cudaStatus != cudaSuccess) {
		std::cout << "runClosenessCalculation failed!";
		return 1;
	}

	std::cout << "D" << std::endl;
	printMatrix(D, ROWS, ROWS);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceReset failed!";
		return 1;
	}

	return 0;
}