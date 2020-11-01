#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <cstdlib>

cudaError_t cudaMul(int* c, const int* a, const int* b, unsigned int size);
void multi(int row1, int col1, int col2, int* a, int* b, int* c);

const int BLOCK_SIZE = 16;

__global__ void matMult(int* a, int* b, int* c, int n) {
    int bx = blockIdx.x, blocky = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = n * BLOCK_SIZE * by;
    int aEnd = aBegin + n - 1;
    int bBegin = BLOCK_SIZE * bx;
    int aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
    int sum = 0;
    for (int iaA = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        __shared__ int as[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int bs[BLOCK_SIZE][BLOCK_SIZE];
        as [ty][tx] = a[ia + n * ty + tx];
        bs [ty][tx] = b[ib + n * ty + tx];
        __syncthreads(); 
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += as[ty][k] * bs[k][tx];
        __syncthreads();
    }
    c[n * BLOCK_SIZE * by + BLOCK_SIZE * bx + n * ty + tx] = sum;
}

void multi(int row1, int col1, int col2, int* a, int* b, int* c) {
    int size = row1 * col2;
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            int sum = 0;
            for (int k = 0; k < col1; k++)
                sum = sum + a[i * col1 + k] * b[k * col2 + j];
            c[i * col2 + j] = sum;
        }
    }
}

int main()
{
    setlocale(LC_ALL, "Rus");
    int N = 2000;
    const int arraySize1 = N * N; const int arraySize2 = N * N; const int arraySize3 = N * N;
    int* a = new int[arraySize1]; int* b = new int[arraySize2]; int* c = new int[arraySize3]; int* d = new int[arraySize3];
    for (int i = 0; i < arraySize1; ++i) { a[i] = rand() % 20; }
    for (int i = 0; i < arraySize2; ++i) { b[i] = rand() % 20; }
    if (col1 != row2) { cout << "Умножение невозможно!"; }
    clock_t begin = clock();
    multi(N, N, N, a, b, c);
    double t = double(clock() - begin) * 1000 / CLOCKS_PER_SEC;
    cout << "Время вычислений на CPU = " << t;
    cudaError_t cudaStatus = cudaMul(a, b, c, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ошибка\n");
        return 1;
    }
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ошибка\n");
        return 1;
    }
    free(a);
    free(b);
    free(c);
    free(d);
    return 0;
}

cudaError_t cudaMul(int* c, const int* a, const int* b, unsigned int N)
{
    const int size = N * N;
    int* dev_a = 0; int* dev_b = 0; int* dev_c = 0;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ошибка");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ошибка");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ошибка");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ошибка");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ошибка");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ошибка");
        goto Error;
    }
    clock_t beginD = clock();
    multiplyOnDevice << <dim3(N / BLOCK_SIZE, N / BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (dev_c, dev_a, dev_b, N);
    cudaThreadSynchronize();
    double deviceTime = double(clock() - beginD) * 1000 / CLOCKS_PER_SEC;
    printf("Ошибка", deviceTime);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ошибка", cudaGetErrorString(cudaStatus));
        goto Error;
    }
        cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ошибка", cudaStatus);
        goto Error;
    }
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ошибка");
        goto Error;
    }
Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return cudaStatus;
}