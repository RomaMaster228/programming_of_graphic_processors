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


// текстурная ссылка <тип элементов, размерность, режим нормализации>
texture<uchar4, 2, cudaReadModeElementType> tex;


__device__ double rgb_to_luma(uchar4 p) {
  return 0.299 * p.x + 0.587 * p.y + 0.114 * p.z;
}


__global__ void kernel(uchar4* out_arr, int w, int h) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	for(int y = idy; y < h; y += offsety) {
	    for(int x = idx; x < w; x += offsetx) {
          double w11 = rgb_to_luma(tex2D(tex, x - 1, y - 1));
          double w12 = rgb_to_luma(tex2D(tex, x - 1, y));
          double w13 = rgb_to_luma(tex2D(tex, x - 1, y + 1));
          double w21 = rgb_to_luma(tex2D(tex, x, y - 1));
          double w22 = rgb_to_luma(tex2D(tex, x, y));
          double w23 = rgb_to_luma(tex2D(tex, x, y + 1));
          double w31 = rgb_to_luma(tex2D(tex, x + 1, y - 1));
          double w32 = rgb_to_luma(tex2D(tex, x + 1, y));
          double w33 = rgb_to_luma(tex2D(tex, x + 1, y + 1));

          double Gx = w13 + 2 * w23 + w33 - w11 - 2 * w21 - w31;
          double Gy = w31 + 2 * w32 + w33 - w11 - 2 * w12 - w13;
          int G = min(255, (int)sqrt(Gx * Gx + Gy * Gy));

          out_arr[y * w + x] = make_uchar4(G, G, G, 0);
      }
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

	// Подготовка данных для текстуры
	cudaArray* arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));

	CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

	// Подготовка текстурной ссылки, настройка интерфейса работы с данными
	tex.addressMode[0] = cudaAddressModeClamp;	// Политика обработки выхода за границы по каждому измерению
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;		// Без интерполяции при обращении по дробным координатам
	tex.normalized = false;						// Режим нормализации координат: без нормализации

	// Связываем интерфейс с данными
	CSC(cudaBindTextureToArray(tex, arr, ch));

	uchar4 *device_out;
	CSC(cudaMalloc(&device_out, sizeof(uchar4) * w * h));

	kernel<<<dim3(32, 32), dim3(32, 32)>>>(device_out, w, h);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, device_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	// Отвязываем данные от текстурной ссылки
	CSC(cudaUnbindTexture(tex));

	CSC(cudaFreeArray(arr));
	CSC(cudaFree(device_out));

	fp = fopen(output_file, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);
	return 0;
}
