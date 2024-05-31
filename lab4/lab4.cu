#include <stdio.h>
#include <time.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>


#define CSC(call) 							\
do { 										\
	cudaError_t status = call;				\
	if (status != cudaSuccess) {																				\
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));		\
		exit(0);								\
	}											\
} while(0)


struct comparator {												
	__host__ __device__ bool operator()(double a, double b) {		// Функция которая сравнивает объекты на "<"
		return abs(a) < abs(b); 									// operator() - переопределение оператора "()" для экземпляра этой структуры
	}
};


__global__ void gauss_method(double* m, int n, int cur_row) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = cur_row + idx; i < n; i += offsetx) {
        for (int j = cur_row + idy; j < n + 1; j += offsety) {
            m[i + j * n] -= m[cur_row - 1 + j * n] / m[cur_row - 1 + (cur_row - 1) * n] * m[i + (cur_row - 1) * n];
        }
    }
}


__global__ void swap(double* m, int n, int row1, int row2) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int i = idx; i < n + 1; i += offset) {
        double tmp = m[row1 + i * n];
        m[row1 + i * n] = m[row2 + i * n];
        m[row2 + i * n] = tmp;
    }
}


int main() {
    int n;
    comparator comp;
    scanf("%d", &n);
    double* m = (double*)malloc(sizeof(double) * n * (n + 1));
    double* res = (double*)malloc(sizeof(double) * n);
	  for (int i = 0; i < n; i++) {
		    for (int j = 0; j < n; j++) {
			      scanf("%lf", &m[i + j * n]);
		    }
	  }
    for(int i = 0; i < n; i++) {
        scanf("%lf", &m[i + n * n]);
    }

    double* device_m;
    cudaMalloc(&device_m, sizeof(double) * n * (n + 1));
    cudaMemcpy(device_m, m, sizeof(double) * n * (n + 1), cudaMemcpyHostToDevice);

    thrust::device_ptr<double> m_ptr = thrust::device_pointer_cast(device_m);	// Трастовские функции принимают свой тип указателей, поэтому выполняем приведение типов.

    cudaEvent_t start, stop;
	  CSC(cudaEventCreate(&start));
	  CSC(cudaEventCreate(&stop));
	  CSC(cudaEventRecord(start));

    for (int i = 0; i < n; i++) {
        thrust::device_ptr<double> res = thrust::max_element(m_ptr + i * n + i, m_ptr + (i + 1) * n, comp);		// Ищем максимум в массиве на GPU
        int cur_i = (int)(res - (m_ptr + i * n));
        if (cur_i != i) {
            swap<<<512, 512>>>(device_m, n, cur_i, i);
            CSC(cudaDeviceSynchronize());
	          CSC(cudaGetLastError());
        }
        gauss_method<<<512, 512>>>(device_m, n, i + 1);
        CSC(cudaDeviceSynchronize());
	      CSC(cudaGetLastError());
    }

	  CSC(cudaEventRecord(stop));
	  CSC(cudaEventSynchronize(stop));
	  float t;
	  CSC(cudaEventElapsedTime(&t, start, stop));
	  CSC(cudaEventDestroy(start));
	  CSC(cudaEventDestroy(stop));

	  //printf("time = %f ms\n", t);

    cudaMemcpy(m, device_m, sizeof(double) * n * (n + 1), cudaMemcpyDeviceToHost);

    for (int i = n - 1; i >= 0; i--) {
		    res[i] = m[i + n * n];
		    for (int j = i + 1; j < n; j++) {
			      res[i] -= res[j] * m[i + j * n];
		    }
		    res[i] /= m[i + i * n];
	  }

    for (int i = 0; i < n; i++) {
        printf("%.10lf ", res[i]);
    }

    free(m);
    free(res);
    cudaFree(device_m);
    return 0;
}
