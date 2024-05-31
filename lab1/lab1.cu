#include <stdio.h>

/*
    threadIdx.x, threadIdx.y, threadIdx.z
    blockIdx.x, blockIdx.y, blockIdx.z
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z,
*/
// __device__ __host__


__global__ void kernel(double* arr1, double* arr2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    while(idx < n) {
        arr1[idx] = min(arr1[idx], arr2[idx]);
        idx += offset;
    }
}


int main() {
    int n;
    double a;
    scanf("%d", &n);
    double* arr1 = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        scanf("%lf", &a);
        arr1[i] = a;
    }
    double* arr2 = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        scanf("%lf", &a);
        arr2[i] = a;
    }

    double* device_arr1;
    cudaMalloc(&device_arr1, sizeof(double) * n);
    cudaMemcpy(device_arr1, arr1, sizeof(double) * n, cudaMemcpyHostToDevice);
    double* device_arr2;
    cudaMalloc(&device_arr2, sizeof(double) * n);
    cudaMemcpy(device_arr2, arr2, sizeof(double) * n, cudaMemcpyHostToDevice);

    kernel<<<512,512>>>(device_arr1, device_arr2, n);

    cudaMemcpy(arr1, device_arr1, sizeof(double) * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%f ", arr1[i]);
    }
    printf("\n");

    free(arr1);
    cudaFree(device_arr1);
    free(arr2);
    cudaFree(device_arr2);
    return 0;
}
