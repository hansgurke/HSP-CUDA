
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
//#include "Quaternion.h"
#include "Constants.h"

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


/*START CLASS DEFINITIONS*/
/**/
/**/

class Quaternion
{

	public:

		__device__ Quaternion(float ar, float ai, float aj, float ak)
			{
				real=ar;
				i=ai;
				j=aj;
				k=ak;
			};

		__device__ Quaternion operator+(Quaternion b)
			{
				return Quaternion(real + b.real, i + b.i, j + b.j, k + b.k);
			};

		__device__ Quaternion operator*(Quaternion b)
			{
				float cr = real*b.real - i*b.i - j*b.j - k*b.k;
				float ci = real*b.i + i*b.real + j*b.k - k*b.j;
				float cj = real*b.j - i*b.k + j*b.real + k*b.i;
				float ck = real*b.k + i*b.j - j*b.i + k*b.real;
				return Quaternion(cr, ci, cj, ck);
			};

		__device__ Quaternion operator=(Quaternion b)
			{
				real=b.real;
				i=b.i;
				j=b.j;
				k=b.k;
				return *this;
			};

		__device__ Quaternion operator+=(Quaternion b)
			{
				real+=b.real;
				i+=b.i;
				j+=b.j;
				k+=b.k;
				return *this;
			};

		__device__ Quaternion operator*=(Quaternion b)
			{
				float cr = real*b.real - i*b.i - j*b.j - k*b.k;
				float ci = real*b.i + i*b.real + j*b.k - k*b.j;
				float cj = real*b.j - i*b.k + j*b.real + k*b.i;
				float ck = real*b.k + i*b.j - j*b.i + k*b.real;
				real=cr;
				i=ci;
				j=cj;
				k=ck;
				return *this;
			};

		__device__  float abs(void)
			{
				return sqrt(pow(real,2)+pow(i,2)+pow(j,2)+pow(k,2));
			};
		__device__ ~Quaternion()
			{

			};


	private:

		float real;
		float i;
		float j;
		float k;
};



/*END CLASS DEFINITIONS*/
/**/
/**/



/*START GLOBAL DEFINITIONS*/
/**/
/**/

__host__ __device__ float getCoordinateValue(int Index)
{
	return ((float)(Index*4)/(float)(DIMENSION-1))-2;
}

__host__ __device__ int getXIndexFromArrayIndex(int Index)
{
	return (int)Index/(DIMENSION*DIMENSION);
}

__host__ __device__ int getYIndexFromArrayIndex(int Index)
{
	return (int)(Index-DIMENSION)/DIMENSION;
}

__host__ __device__ int getZIndexFromArrayIndex(int Index)
{
	return (int)(Index-DIMENSION-DIMENSION)/DIMENSION;
}


/**/
/**/
/*END GLOBAL DEFINITIONS*/



/*START KERNEL DEFINITION*/
/**/
/**/

__global__ void calc_JuliaSet_quat_3D_Part(unsigned char* A, float k_Index, float C_real, float C_i, float C_j, float C_k)
{
	unsigned long tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned long d = DIMENSION;
	unsigned long DIM = d*d*d;
	if(tid+(blockDim.x*gridDim.x)<DIM)
	{
		unsigned int number_of_works=DIM/(blockDim.x*gridDim.x);

		for(int i=0; i<=number_of_works; i++)
		{
			Quaternion Z = Quaternion(getXIndexFromArrayIndex(tid+(i*blockDim.x*gridDim.x)),getXIndexFromArrayIndex(tid+(i*blockDim.x*gridDim.x)), getXIndexFromArrayIndex(tid+(i*blockDim.x*gridDim.x)), k_Index);
			Quaternion C = Quaternion(C_real, C_i, C_j, C_k);
			//function to calculate MandelbrotSet  z(1) = z(0)² + c
			int k=0;
			while(k<MAX_ITERATIONS && Z.abs()<2)
			{
				Z=Z*Z+C;
				k++;
			}
			A[tid+(i*blockDim.x*gridDim.x)]=k/(MAX_ITERATIONS/256);
		}
	}
}

/**/
/**/
/*END KERNEL DEFINITION*/






void start_Calculation()
{
	//Quaternion C = Quaternion(0.1, 0.1, 0.1, 0.1);
	int devices = 0;
	unsigned char* host_arrays[MAX_DEVICES_POSSIBLE];
	unsigned char* device_arrays[MAX_DEVICES_POSSIBLE];
	size_t size = DIMENSION * DIMENSION * DIMENSION * sizeof(unsigned char);
	cudaError_t error;
	//float k_index = 0;

	/*Cuda Pre-Condition-Checking*/
	/**/

	//Look how many Devices are present
	error = cudaGetDeviceCount(&devices);
	if (error != cudaSuccess)
	{
	    printf("cudaGetDeviceCount returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}
	if(devices>MAX_DEVICES_POSSIBLE)
	{
		printf("too much devices found! increment MAX_DEVICES_POSSIBLE");
		exit(EXIT_FAILURE);
	}
	printf("DeviceCount: %d\n", devices);

	//create Array for each Device
	for(int i=0; i<devices; i++)
	{
		host_arrays[i] = (unsigned char *)malloc(size);

		//set context to specific device
		error = cudaSetDevice(i);
		if (error != cudaSuccess)
		{
		    printf("cudaSetDeviceCount returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		    exit(EXIT_FAILURE);
		}

		//allocate memory on device
		error = cudaMalloc((void **) &device_arrays[i], size);
		if (error != cudaSuccess)
		{
		    printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		    exit(EXIT_FAILURE);
		}
	}

	int pos=0;
	while(pos<DIMENSION)
	{
		//starting Kernels
		for(int i=0; i<devices; i++)
		{
			//set context to specific device
			error = cudaSetDevice(i);
			if (error != cudaSuccess)
			{
			    printf("cudaSetDeviceCount returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
			    exit(EXIT_FAILURE);
			}
			//printf("%f\n", getCoordinateValue(pos));
			calc_JuliaSet_quat_3D_Part<<<MAX_BLOCKS_PER_GRID, MAX_THREADS_PER_BLOCK>>>(device_arrays[i], getCoordinateValue(pos), 0.1, 0.1, 0.1, 0.1);
			pos++;
		}
		/*
		if(pos%20==0)
		{
			printf("started %d Kernels\n position is now %d\n", devices, pos);
		}
		*/
		//reading back results
		for(int i=0; i<devices; i++)
		{
			//set context to specific device
			error = cudaSetDevice(i);
			if (error != cudaSuccess)
			{
			    printf("cudaSetDeviceCount returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
			    exit(EXIT_FAILURE);
			}
			//printf("cudaMemcpy");
			cudaMemcpy(host_arrays[i], device_arrays[i], size, cudaMemcpyDeviceToHost);
		}
	}

	//free all host and device array memory
	for(int i=0; i<devices; i++)
	{
		//set context to specific device
		error = cudaSetDevice(i);
		if (error != cudaSuccess)
		{
		    printf("cudaSetDeviceCount returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		    exit(EXIT_FAILURE);
		}
		cudaFree(device_arrays[i]);
		free(host_arrays[i]);
	}


}



int main(int argc, char* argv[])
{
	clock_t prgstart, prgende;
	printf("start with %d DIMs and %d BLOCKS\n", DIMENSION, MAX_BLOCKS_PER_GRID);
	prgstart=clock();
	start_Calculation();
	prgende=clock();//CPU-Zeit am Ende des Programmes
	printf("Laufzeit %.2f Sekunden\n",(float)(prgende-prgstart) / CLOCKS_PER_SEC);
	printf("stop\n");
	return 0;
}
