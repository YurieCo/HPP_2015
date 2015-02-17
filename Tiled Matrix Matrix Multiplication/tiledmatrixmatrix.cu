#include <wb.h>

#define TILE_WIDTH 16

#define wbCheck(stmt)                                                          \
  do {                                                                         \
	cudaError_t err = stmt;                                                    \
	if (err != cudaSuccess) {                                                  \
	  wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
	  wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
	  return -1;                                                               \
	}                                                                          \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float* A, float* B, float* C, int numARows,
									 int numAColumns, int numBRows,
									 int numBColumns, int numCRows,
									 int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
	__shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

	int tIdX = threadIdx.x;
	int tIdY = threadIdx.y;
 
	int Row = blockIdx.y * blockDim.y + tIdY;
	int Col = blockIdx.x * blockDim.x + tIdX;

	float Csum = 0.0;
	for (size_t tile = 0; tile < (numAColumns - 1) / TILE_WIDTH + 1; ++tile)	//Interating through the tiles
	{
		int aColumnOffset = tile * TILE_WIDTH + tIdX;
		int aIdx = Row * numAColumns + aColumnOffset;
		//sharedA[tIdY][tIdX] = A[aIdx]; //Each thread puts a value from A to the shared memory reserved for A
		sharedA[tIdY][tIdX] = (aColumnOffset < numAColumns && Row < numCRows) ? A[aIdx] : 0.0; 		//Works if numAColumns is dividable by TILE_WIDTH		

		int bRowOffset = tile * TILE_WIDTH + tIdY;
		int bIdx = Col + bRowOffset  * numCColumns;
		//sharedB[tIdY][tIdX] = B[bIdx];	//Each thread puts a value from B to the shared memory reserved for B 		//Works if numAColumns is dividable by TILE_WIDTH
		sharedB[tIdY][tIdX] = (bRowOffset < numAColumns && Col < numCColumns) ? B[bIdx] : 0.0;
		__syncthreads(); //Syncs threads meaning all threads will wait until each of them has come to this point, this means shared memory is fully set for the tile

		//Iterating through the row and col of shared memory
		for (size_t i = 0; i < TILE_WIDTH; ++i)
		{
			Csum += sharedA[tIdY][i] * sharedB[i][tIdX]; //Row * Column multiplication and addition to the Sum
		}
		__syncthreads();	//Syncs threads again so all threads wait for eachother
	}

	//Checking if inside the valid range of output matrix
	if ((Row < numCRows) && (Col < numCColumns)) {
		C[Row * numCColumns + Col] = Csum; //assignment of the final sum to the output
	}

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
	  ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
	  ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaError_t err;	//For fetching cuda errors
  err = cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float));
  if (err != cudaSuccess) {
	  printf("%s at line %d\n", cudaGetErrorString(err), __LINE__);
	  exit(EXIT_FAILURE);
  }

  err = cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float));
  if (err != cudaSuccess) {
	  printf("%s at line %d\n", cudaGetErrorString(err), __LINE__);
	  exit(EXIT_FAILURE);
  }

  err = cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float));
  if (err != cudaSuccess) {
	  printf("%s at line %d\n", cudaGetErrorString(err), __LINE__);
	  exit(EXIT_FAILURE);
  }

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  err = cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
	  printf("%s at line %d\n", cudaGetErrorString(err), __LINE__);
	  exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
	  printf("%s at line %d\n", cudaGetErrorString(err), __LINE__);
	  exit(EXIT_FAILURE);
  }

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared << <DimGrid, DimBlock >> >(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
	  printf("%s at line %d\n", cudaGetErrorString(err), __LINE__);
	  exit(EXIT_FAILURE);
  }

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
