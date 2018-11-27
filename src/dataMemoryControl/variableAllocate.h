/*file statement



*/

#ifndef variableAllocate
#define variableAllocate 



#include <stdio.h>
#include <stdlib.h>
 
/*HOST SECTION*/
template <typename T >
void   _malloc_host_np(T **a1, int   length){
	*a1 = (T *)malloc(length * sizeof(T));
}

template <typename T >
void   _malloc_host_np(T **a1, T **a2, int   length){
	*a1 = (T *)malloc(length * sizeof(T));
	*a2 = (T *)malloc(length * sizeof(T));
}

template <typename T >
void   _malloc_host_np(T **a1, T **a2, T **a3, int   length){
	*a1 = (T *)malloc(length * sizeof(T));
	*a2 = (T *)malloc(length * sizeof(T));
	*a3 = (T *)malloc(length * sizeof(T));
}

template <typename T >
void   _malloc_host_np(T **a1, T **a2, T **a3, T **a4, int   length){
	*a1 = (T *)malloc(length * sizeof(T));
	*a2 = (T *)malloc(length * sizeof(T));
	*a3 = (T *)malloc(length * sizeof(T));
	*a4 = (T *)malloc(length * sizeof(T));
}

template <typename T >
void   _malloc_host_np(T **a1, T **a2, T **a3, T **a4, T **a5, int   length){
	*a1 = (T *)malloc(length * sizeof(T));
	*a2 = (T *)malloc(length * sizeof(T));
	*a3 = (T *)malloc(length * sizeof(T));
	*a4 = (T *)malloc(length * sizeof(T));
	*a5 = (T *)malloc(length * sizeof(T));
}

template <typename T >
void   _malloc_host_np(T **a1, T **a2, T **a3, T **a4, T **a5, T **a6, int   length){
	*a1 = (T *)malloc(length * sizeof(T));
	*a2 = (T *)malloc(length * sizeof(T));
	*a3 = (T *)malloc(length * sizeof(T));
	*a4 = (T *)malloc(length * sizeof(T));
	*a5 = (T *)malloc(length * sizeof(T));
	*a6 = (T *)malloc(length * sizeof(T));
}



/*DEVICE SECTION*/
template <typename T >
void   _malloc_device_np(T **a1, int   length, int  numGPUs)
{
	if (numGPUs > 0)	{
		cudaMalloc((void**)a1, length * sizeof(T));
	}
	else {
		cout << "numGPU<=0" << endl;
	}
}

template <typename T >
void   _malloc_device_np(T **a1, T **a2, int   length, int  numGPUs)
{
	if (numGPUs > 0)	{
		cudaMalloc((void**)a1, length * sizeof(T));
		cudaMalloc((void**)a2, length * sizeof(T));
	}
	else {
		cout << "numGPU<=0" << endl;
	}
}

template <typename T >
void   _malloc_device_np(T **a1, T **a2, T **a3, int   length, int  numGPUs)
{
	if (numGPUs > 0)	{
		cudaMalloc((void**)a1, length * sizeof(T));
		cudaMalloc((void**)a2, length * sizeof(T));
		cudaMalloc((void**)a3, length * sizeof(T));
	}
	else {
		cout << "numGPU<=0" << endl;
	}
}

template <typename T >
void   _malloc_device_np(T **a1, T **a2, T **a3, T **a4, int   length, int  numGPUs)
{
	if (numGPUs > 0)	{
		cudaMalloc((void**)a1, length * sizeof(T));
		cudaMalloc((void**)a2, length * sizeof(T));
		cudaMalloc((void**)a3, length * sizeof(T));
		cudaMalloc((void**)a4, length * sizeof(T));
	}
	else {
		cout << "numGPU<=0" << endl;
	}
}

template <typename T >
void   _malloc_device_np(T **a1, T **a2, T **a3, T **a4, T **a5,   int   length, int  numGPUs)
{
	if (numGPUs > 0){
		cudaMalloc((void**)a1, length*sizeof(T));
		cudaMalloc((void**)a2, length*sizeof(T));
		cudaMalloc((void**)a3, length*sizeof(T));
		cudaMalloc((void**)a4, length*sizeof(T));
		cudaMalloc((void**)a5, length*sizeof(T));		 
	}
	else{
		cout << "numGPU<=0" << endl;
	}
} 

template <typename T >
void   _malloc_device_np(T **a1, T **a2, T **a3, T **a4, T **a5, T **a6, int   length, int  numGPUs)
{  
	if (numGPUs > 0){
		cudaMalloc((void**)a1, length*sizeof(T));
		cudaMalloc((void**)a2, length*sizeof(T));
		cudaMalloc((void**)a3, length*sizeof(T));
		cudaMalloc((void**)a4, length*sizeof(T));
		cudaMalloc((void**)a5, length*sizeof(T));
		cudaMalloc((void**)a6, length*sizeof(T));		 
	}
	else	{		cout << "numGPU<=0" << endl;	}
}






#endif
