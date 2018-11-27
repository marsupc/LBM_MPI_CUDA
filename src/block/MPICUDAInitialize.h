/*file statement



*/

#ifndef MPICUDA
#define MPICUDA
#include <mpi.h> 
#include <cuda.h>

 
inline void MPICUDAInitial( int NumCPU_CORE,int ID_CPU_CORE, int numGPUs, int* ID_GPUp )
{
cudaDeviceProp deviceProp;	
	if (ID_CPU_CORE == 0){
		printf(" numCPU = %d ,  numGPUs =  %d\n",NumCPU_CORE, numGPUs);
		if (numGPUs > 0){			 		 
			for (int nDevice = 0; nDevice < numGPUs; ++nDevice)	{				
				cudaGetDeviceProperties(&deviceProp, nDevice);
				printf("  DeviceID %d: %s\n", nDevice, deviceProp.name);
			}
		}
	}
//////////////////////////////////////////////

	MPI_Barrier(MPI_COMM_WORLD); 
	cudaDeviceProp devicePropID;	 

	if (ID_CPU_CORE > 0)
	{
		*ID_GPUp = ID_CPU_CORE - 1;
		cudaSetDevice(*ID_GPUp);	
		cudaGetDeviceProperties(&devicePropID, *ID_GPUp);
		printf("ID_CPU  %d  Now using %d: %s\n", ID_CPU_CORE, *ID_GPUp, devicePropID.name);
	}

}


#endif
