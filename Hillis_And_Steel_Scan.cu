#include <cuda.h>
#include <stdio.h>
#include <numeric>
#include <stdlib.h>
#define BLOCK_SIZE 32

void scan(int N);

__global__ void scanKernel(float *input,int n,int sIndex=0);
__global__ void maxValArrKernel(float* sData,float* output);
__global__ void finScanKernel(float* initial,float* max);

int main(void)
{
    for (int i = 1; i < 20; i++)
        scan(1<<i);
    return 0;
}

void scan(int N){
    float *input;

    //begin event monitoring, allocate unified mem, init input data on host
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t err;

    cudaMallocManaged(&input, N * sizeof(float));

    for (int i = 0; i < N; i++)
        input[i] = 1.0f;
    
    // run first scan
    int grid_size = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaEventRecord(start);
    scanKernel<<<grid_size, BLOCK_SIZE>>>(input,BLOCK_SIZE);
    err = cudaDeviceSynchronize();                                  // Wait for GPU to finish before accessing on host
    cudaEventRecord(stop);
    float timeElapsed = 0;
    cudaEventElapsedTime(&timeElapsed, start, stop);


    // run scan of the max output values, then run another scan on the kernell call
    float *scanMax;
    cudaMallocManaged(&scanMax, grid_size * sizeof(float));
    cudaEventRecord(start);
    maxValArrKernel<<<grid_size, BLOCK_SIZE>>>(input,scanMax);
    err = cudaDeviceSynchronize();
    cudaEventRecord(stop);
    float tmp=0;
    cudaEventElapsedTime(&tmp, start, stop);
    timeElapsed+=tmp;


    // run scan of the max output values again, this time with cross-block communication
    int tmpBlockSize=512;
    int scanGridSize=((grid_size + tmpBlockSize - 1) / tmpBlockSize);
    int sIndex=0;
    int endIndex=0;

    while(scanGridSize>0){
        scanGridSize=scanGridSize-1;                                        //initial values
        endIndex=grid_size-(tmpBlockSize*(scanGridSize));
        
        cudaEventRecord(start);                                             //Timing and running the scan Kernel.
        scanKernel<<<1, tmpBlockSize>>>(scanMax,grid_size,sIndex);

        cudaDeviceSynchronize();                                            //Synchronising, getting elapsed time
        cudaEventRecord(stop);
        cudaEventElapsedTime(&tmp, start, stop);                            
        timeElapsed+=tmp;
        sIndex=endIndex;                                                    //setting the startpoint for the next iteration as the endpoint of this one

        if(scanGridSize>0)
            scanMax[endIndex]+=scanMax[endIndex-1];
    }

    // Summing remaining values
    cudaEventRecord(start);
    finScanKernel<<<grid_size, BLOCK_SIZE>>>(input,scanMax);
    err = cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventElapsedTime(&tmp, start, stop);
    timeElapsed+=tmp;
    
    //printf("N - %d ,time elapsed: %f , final element: %f\n",N,timeElapsed,input[N-1]);
    
    //Free up mem
    cudaFree(input);
    cudaFree(scanMax);
    cudaDeviceReset();
}

__global__ void finScanKernel(float* initial,float* max){
    int thIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if(blockIdx.x!=0)
        initial[thIdx]+=max[blockIdx.x-1];
}



__global__ void scanKernel(float *input,int n,int sIndex)
{
    int thIdx = threadIdx.x + blockIdx.x * blockDim.x+sIndex;
    int tid = threadIdx.x;

    __shared__ float tmp[1024];
    tmp[tid] = input[thIdx];
    __syncthreads();

    //offsetting the values by i so they match the previous ones
    //creating an initial scan
    for (int i = 1; i < n; i *= 2)
    {
        if (tid >= i)
            tmp[tid] += tmp[tid - i];
        __syncthreads();
    }
    input[thIdx] = tmp[tid];
}

__global__ void maxValArrKernel(float* input,float* output){
    int thIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if(tid==BLOCK_SIZE-1)
        output[blockIdx.x]=input[thIdx];
}