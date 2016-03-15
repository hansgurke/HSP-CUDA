#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "Constants.h"
#include <sstream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#define MODUS ,0711)

/*
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
*/

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
				float tmp = real*real;
				tmp += i*i;
				tmp += j*j;
				tmp += k*k;
				return sqrtf(tmp);
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

/*conversion from 1D representation of a 3D array into a 3D array*/

//get the exact float value for a given 3D index value for -1<x,y,z<1
__host__ __device__ float getCoordinateValue(int Index)
{
	return ((float)(Index*2)/(float)(DIMENSION-1))-1;
}

//get X Coordinate of 3D array at given 1D array index value
__host__ __device__ int getXIndexFromArrayIndex(int Index)
{
	return (int)Index/(DIMENSION*DIMENSION);
}

//get Y Coordinate of 3D array at given 1D array index value
__host__ __device__ int getYIndexFromArrayIndex(int Index)
{
	return (int) (Index%(DIMENSION*DIMENSION))/DIMENSION;
}

//get Z Coordinate of 3D array at given 1D array index value
__host__ __device__ int getZIndexFromArrayIndex(int Index)
{
	return (int) Index%DIMENSION;
}


/**/
/**/
/*END GLOBAL DEFINITIONS*/



/*START KERNEL DEFINITION*/
/**/
/**/


//kernel to calculate the k_Index-th 3D part of Julia Set in quaternion space at given C
__global__ void calc_JuliaSet_quat_3D_Part(unsigned char* A, float k_Index, float C_real, float C_i, float C_j, float C_k)
{
	unsigned long tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned long d = DIMENSION;
	unsigned long DIM = d*d*d;
	unsigned long number_of_works=DIM/(blockDim.x*gridDim.x);

	for(int i=0; i<=number_of_works; i++)
	{
		if(i==number_of_works && tid>(DIM%(blockDim.x*gridDim.x)))
		{
		
		}
		else
		{
			Quaternion Z = Quaternion(getCoordinateValue(getXIndexFromArrayIndex(tid+(i*blockDim.x*gridDim.x))),getCoordinateValue(getYIndexFromArrayIndex(tid+(i*blockDim.x*gridDim.x))), getCoordinateValue(getZIndexFromArrayIndex(tid+(i*blockDim.x*gridDim.x))), k_Index);
			Quaternion C = Quaternion(C_real, C_i, C_j, C_k);
			//function to calculate JuliaSet  z(n+1) = z(n)² + c
			int k=0;
			while(k<MAX_ITERATIONS && Z.abs()<2)
			{
				Z=Z*Z+C;
				k++;
			}
			//int f = (MAX_ITERATIONS/256.0)+0.5;
			A[tid+(i*blockDim.x*gridDim.x)]=k;//(MAX_ITERATIONS/256);
			//A[tid+(i*blockDim.x*gridDim.x)]='a';
		}
	}
}

/**/
/**/
/*END KERNEL DEFINITION*/






void start_Calculation(float creal, float ci, float cj, float ck, int maxBlocksPerGrid, int maxThreadsPerBlock)
{
	int devices = 0;
	unsigned char* host_array;
	unsigned char* device_arrays[MAX_DEVICES_POSSIBLE];
	size_t size = DIMENSION * DIMENSION * DIMENSION * sizeof(unsigned char);
	cudaError_t error;
	host_array = (unsigned char*) malloc(size);
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

	FILE* dotfile;
	
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

			calc_JuliaSet_quat_3D_Part<<<maxBlocksPerGrid, maxThreadsPerBlock>>>(device_arrays[i], getCoordinateValue(pos), creal, ci, cj, ck);
			pos++;
			error = cudaGetLastError();

    			if (error != cudaSuccess)
    			{
        			fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(error));
        			exit(EXIT_FAILURE);
    			}
		}

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

			cudaMemcpy(host_array, device_arrays[i], size, cudaMemcpyDeviceToHost);

			//print results into .dat file
			std::ostringstream file;
			file << "out/dots" << std::setfill('0') << std::setw(4) << (pos-(devices-i)) << "_k_" << getCoordinateValue(pos) << ".dat";
			dotfile = fopen(file.str().c_str(), "w+");
			for(int j=0; j<DIMENSION*DIMENSION*DIMENSION; j++)
			{
				
				if(host_array[j] >= (unsigned char)255)
				{
					fprintf(dotfile, "%f %f %f\n", getCoordinateValue(getXIndexFromArrayIndex(j)), getCoordinateValue(getYIndexFromArrayIndex(j)), getCoordinateValue(getZIndexFromArrayIndex(j)));//, host_array[j]);
				}
			}
			fclose(dotfile);
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
	}
	free(host_array);

}



int main(int argc, char* argv[])
{
	printf("start\n");//"with %d DIMs, %d BLOCKS with each %d THREADS and %d ITERATIONS\n", DIMENSION, MAX_BLOCKS_PER_GRID, MAX_THREADS_PER_BLOCK, MAX_ITERATIONS);
	clock_t prgstart, prgende; 

	//Quaternion C
	float creal = C_REAL;
	float ci = C_I;
	float cj = C_J;
	float ck = C_K;
	int maxblocks = MAX_BLOCKS_PER_GRID;
	int maxthreads = MAX_THREADS_PER_BLOCK;	
	
	//start time measurement
	prgstart=clock();

	start_Calculation(creal, ci, cj, ck, maxblocks, maxthreads);

	//stop time measurement
	prgende=clock(); 

	//print result
	FILE* infofile;
	std::ostringstream file;
	file << "out/info.txt";
	infofile = fopen(file.str().c_str(), "w+");
	fprintf(infofile, "quaternion C:\n\treal= %f\n\ti= %f\n\tj= %f\n\tk= %f\nblocksize= %d threads per block= %d\n\nproblemdescription:\ndimension: %d x %d x %d x %d\niterations= %d\nexecutiontime= %fsec\n",creal, ci, cj, ck, maxblocks, maxthreads, DIMENSION, DIMENSION, DIMENSION, DIMENSION, MAX_ITERATIONS, (float)(prgende-prgstart) / CLOCKS_PER_SEC); 
	fclose(infofile);
	//printf("Laufzeit %.2f Sekunden\n",(float)(prgende-prgstart) / CLOCKS_PER_SEC);
	printf("stop\n");
	return 0;
}
