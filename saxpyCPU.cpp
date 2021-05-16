#include <stdio.h>
#include <stdlib.h>

// Función que se ejecuta en la CPU
void saxpy_cpu(float *x, float *y, float a, int numElementos)
{
    for (int i = 0; i < numElementos; ++i)
    {
        y[i] = a*x[i] + y[i];
    }
}

int main(void)
{
 
	// Definimos tamaños: 1M de elementos (2^20)
	int numElementos = 1<<20; // 
    size_t tamano = numElementos * sizeof(float);
	
    //Definimos variables
	float *x, *y;
	
	// Reservamos memoria
    x = (float *)malloc(tamano);
    y = (float *)malloc(tamano);
    
    // Asignamos valores
    for (int i = 0; i < numElementos; i++)
    {
		x[i] = 1.0f;
		y[i] = 2.0f;
    }
	float a = 5.0f;

	// LLamamos a función saxpy_cpu
    saxpy_cpu(x, y, a, numElementos);
    
    // Verificamos resultado
    float Error = 0.0f;
	for (int i = 0; i < numElementos; i++)
		Error = Error + abs(y[i]-7.0f);
	printf("Error: %f\n", Error);

    // Liberamos memoria host
    free(x);
    free(y);

    return 0;
}

