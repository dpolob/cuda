#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>


#define ANCHOMATRIZ 64

// Definición del kernel
__global__ void matrixMul(float *a, float *b, float *c)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < ANCHOMATRIZ && col < ANCHOMATRIZ)
	{
		float sum = 0.0f;
		for (int i = 0; i < ANCHOMATRIZ; i++) {
            sum += a[row * ANCHOMATRIZ + i] * b[i * ANCHOMATRIZ + col];
        }
    c[row * ANCHOMATRIZ + col] = sum;
	}
}


void matrixMul_CPU(float *a, float *b, float *c)
{

    float acum;

    for (int i = 0 ; i<ANCHOMATRIZ ; i++)
    {
		for (int j = 0 ; j<ANCHOMATRIZ ; j++)
    	{  
			acum = 0;
  
			for (int k = 0 ; k<ANCHOMATRIZ ; k++)
			{
				acum = acum + a[i*ANCHOMATRIZ + k]*b[k*ANCHOMATRIZ + j];
			} 
	
			c[i*ANCHOMATRIZ+j] = acum;
    	} 
    }   
}


long long milisegundos()
// Devuelve el tiempo en milisegundos desde la época Unix (01/01/1970)
{
        struct timeval t;
        gettimeofday(&t, NULL);
        return t.tv_sec*1000 + t.tv_usec/1000;
}


int main(void)
{
    int numElementos = ANCHOMATRIZ*ANCHOMATRIZ;
    size_t tamano = numElementos * sizeof(int);


    // Reservamos memoria host (memoria principal)
    float *h_a = (float *)malloc(tamano);
    float *h_b = (float *)malloc(tamano);
    float *h_c = (float *)malloc(tamano);

	long long ti,tf;
    // Inicializar con números arbitrarios
    for (int i = 0; i < ANCHOMATRIZ; ++i)
    {
	    for (int j = 0; j < ANCHOMATRIZ; j++)
	    {
		h_a[i*ANCHOMATRIZ+j] = rand()/(float)RAND_MAX;
		h_b[i*ANCHOMATRIZ+j] = rand()/(float)RAND_MAX;
	    }
    }
	
	ti=milisegundos(); //tiempo inicial
    // Ejecutamos la multiplicación de matrices
    matrixMul_CPU(h_a, h_b, h_c);
	tf=milisegundos(); //tiempo final
    printf("Tiempo invertido en multiplicar CPU: %f\n", (tf-ti));
	
	printf("------- CPU --------\n");
	for (int i = 0; i < 10; ++i)
    {
        printf("Componente [%d] = %f\n", i, h_c[i]);
    }
	

	
	
    //Crear variables para la parte device (d_a, d_b, d_c)
    float *d_a, *d_b, *d_c;

    // Reservar memoria en la parte device
	cudaMalloc(&d_a, tamano);
    cudaMalloc(&d_b, tamano);
	cudaMalloc(&d_c, tamano);

	//Pasar datos de la memoria host a memoria device
    cudaMemcpy(d_a, h_a, tamano, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, tamano, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, tamano, cudaMemcpyHostToDevice);
 

    // Cada bloque tendrá 256 hilos y habrá 4096 bloques
	dim3 dimBlock(16,16);
    dim3 dimGrid(64,64);
	
	ti=milisegundos(); //tiempo inicial
	// Lanzar Kernel
	matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
     
    //Pasar datos de la memoria device a memoria host
    cudaMemcpy(h_c, d_c, tamano, cudaMemcpyDeviceToHost);
    
	tf=milisegundos(); //tiempo final
    printf("Tiempo invertido en multiplicar GPU: %f\n", (tf-ti));

    // Verificamos los primeros valores
	printf("------- GPU --------\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("Componente [%d] = %f\n", i, h_c[i]);
    }

	// Liberamos memoria device
    cudaFree(d_a);
    cudaFree(d_b);
	cudaFree(d_c);

    // Liberamos memoria host
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

