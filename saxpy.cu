#include <stdio.h>

// Función kernel que se ejecuta en la GPU
__global__ void saxpy(float *x, float *y, float a, int numElementos)
{
    int tid= blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < numElementos)
    {
        y[tid] = a * x[tid] + y[tid]; 
    }
}


int main(void)
{
 
    // Definimos tamaños: 1M de elementos (2^20)
    int numElementos = 1<<20; // 
    size_t tamano = numElementos * sizeof(float);
	
    //Definimos variables
	float *h_x, *h_y, *d_x, *d_y;
	
	// Reservamos memoria  en el host
    h_x = (float *)malloc(tamano);
    h_y = (float *)malloc(tamano);
    
    // Asignamos valores en el host
    for (int i = 0; i < numElementos; i++)
    {
		h_x[i] = 1.0f;
		h_y[i] = 2.0f;
    }
	float a = 5.0f;

    // Reservamos memoria en el device
    cudaMalloc(&d_x, tamano);
    cudaMalloc(&d_y, tamano);

    // Traspasamos datos de host a device
    cudaMemcpy(d_x, h_x, tamano, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, tamano, cudaMemcpyHostToDevice);


    //Lanzamos el kernel CUDA
    int hilosPorBloque = 8;
    int totalBloques =(numElementos + hilosPorBloque - 1) / hilosPorBloque;

    saxpy<<<totalBloques, hilosPorBloque>>>(d_x, d_y, 5.0f, tamano);

    // Traspasamos datos de device a host
    cudaMemcpy(h_y, d_y, tamano, cudaMemcpyDeviceToHost);

    // Verificamos resultado
    float Error = 0.0f;
    int i;
	for (i = 0; i < 10; i++) {
		Error = Error + abs(h_y[i] - 7.0f);
	printf("%2.8f", h_x[i]);
	}
	printf("Error: %f\n", Error);

    // Liberamos memoria device
    cudaFree(d_x);
    cudaFree(d_y);

    // Liberamos memoria host
    free(h_x);
    free(h_y);

    return 0;
}

