#include <stdio.h>
#include <stdlib.h>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)


typedef struct {
		double r;
		double g;
		double b;
} RGB;


__constant__ RGB device_normed_avgs[32];


__global__ void kernel(uchar4* arr, int classes, int w, int h) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;

	for (int x = idx; x < w * h; x+=offsetx) {
			double argmax = 0;
			argmax += ((double)arr[x].x) * device_normed_avgs[0].r;
			argmax += ((double)arr[x].y) * device_normed_avgs[0].g;
			argmax += ((double)arr[x].z) * device_normed_avgs[0].b;
			int pos = 0;
			for (int i = 1; i < classes; i++) {
					double cur_arg = 0;
					cur_arg += ((double)arr[x].x) * device_normed_avgs[i].r;
					cur_arg += ((double)arr[x].y) * device_normed_avgs[i].g;
					cur_arg += ((double)arr[x].z) * device_normed_avgs[i].b;
					if (cur_arg > argmax) {
							pos = i;
							argmax = cur_arg;
					}
			}
			arr[x].w = (unsigned char)pos;
	}
}


int main() {
	int w, h;
	char input_file[100], output_file[100];
	scanf("%s", input_file);
	scanf("%s", output_file);
	FILE* fp = fopen(input_file, "rb");
	if (fp == NULL) {
	    fprintf(stderr, "Can't open file\n");
	}
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	int classes;
	scanf("%d", &classes);
	RGB normed_avgs[32];
	for (int i = 0; i < classes; i++) {
			int pixels;
			scanf("%d", &pixels);
			normed_avgs[i].r = 0;
			normed_avgs[i].g = 0;
			normed_avgs[i].b = 0;
			for (int j = 0; j < pixels; j++) {
					int x, y;
					scanf("%d", &x);
					scanf("%d", &y);
					normed_avgs[i].r += (double)data[x + y * w].x;
					normed_avgs[i].g += (double)data[x + y * w].y;
					normed_avgs[i].b += (double)data[x + y * w].z;
			}
			normed_avgs[i].r /= pixels;
			normed_avgs[i].g /= pixels;
			normed_avgs[i].b /= pixels;
			double norm = sqrt(pow(normed_avgs[i].r, 2) + pow(normed_avgs[i].g, 2) + pow(normed_avgs[i].b, 2));
			normed_avgs[i].r /= norm;
			normed_avgs[i].g /= norm;
			normed_avgs[i].b /= norm;
	}

	CSC(cudaMemcpyToSymbol(device_normed_avgs, normed_avgs, 32 * sizeof(RGB)));

	uchar4* device_out;
	CSC(cudaMalloc(&device_out, sizeof(uchar4) * w * h));
	CSC(cudaMemcpy(device_out, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

	kernel<<<512, 512>>>(device_out, classes, w, h);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, device_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	CSC(cudaFree(device_out));

	fp = fopen(output_file, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);
	return 0;
}
