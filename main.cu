
#include <stdio.h>
#include <stdlib.h>
#include "lbm.h" 
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <iostream>
#include <sys/stat.h> 

#include <mpi.h>
#include <cmath> 
#include <ctime>
#include <iomanip> 
using namespace std;

#define TAG_image     10

#define TAG_fS     11
#define TAG_rhoS   12 
#define TAG_uzS    13

#define TAG_TT     101

int NX =64, NY =64, NZ = 60;

double ex[19] = { 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, -1, 1, -1, 0, 0, 0, 0 };             //notice   point----------------------
double ey[19] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1 };
double ez[19] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, -1, 1, -1 };
double   t_k[19], rsq[19], Bi[19];



void setImageSimpleBlock(int *imageOver);

void initialSinglePhase_3D(int *image, double *ux, double *uy, double *uz, double *rho, double *f,  double rhoo ,
	double *t_k, double *ex, double *ey, double *ez,   int NX, int NY, int NZ);

__global__ void _GetMacro3D_SF_LocalG(int *image, double *fL, double *uxL, double *uyL, double *uzL, double *ux_oldL, double *uy_oldL, double *uz_oldL,
	double *rhoL,    int NX, int NY, int NZ_sub);
__global__ void _CollisionSF_localG (int *imageL, double *uxLocal, double *uyLocal, double *uzLocal, double *rhoLocal, double *fLocal,  
	double taur,  double *t_k, double *ex, double *ey, double *ez,double c_squ,  double *f_body, int NX, int NY, int NZ_sub);

int main(int argc, char** argv) {
   
   // cpu parallel parameter
	int  NumCPU_CORE, ID_CPU_CORE;
	int  numGPUs, ID_GPU;
//	cudaError_t cudaStatus;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &ID_CPU_CORE);
	MPI_Comm_size(MPI_COMM_WORLD, &NumCPU_CORE);


        cudaDeviceProp deviceProp;	
	    cudaDeviceProp devicePropID;	
		cudaGetDeviceCount(&numGPUs);
 
			if (ID_CPU_CORE > 0)
	{
		ID_GPU = ID_CPU_CORE - 1;
		cudaSetDevice(ID_GPU);	
		cudaGetDeviceProperties(&devicePropID, ID_GPU);
		printf("ID_CPU  %d  Now using %d: %s\n", ID_CPU_CORE, ID_GPU, devicePropID.name);
	}
	 
		MPI_Barrier(MPI_COMM_WORLD);
	if (ID_CPU_CORE == 0)
	{
		cudaGetDeviceCount(&numGPUs);
		numGPUs = set_numGPUs(numGPUs, argc, argv);
		printf(" numCPU = %d ,  numGPUs =  %d\n",NumCPU_CORE, numGPUs);
		if (numGPUs > 0)
		{
			int  nDevice;			 
			for (nDevice = 0; nDevice < numGPUs; ++nDevice)
			{				
				cudaGetDeviceProperties(&deviceProp, nDevice);
				printf("  DeviceID %d: %s\n", nDevice, deviceProp.name);
			}
		}
	}
	
	dim3  grid(NX / blockDim_x, NY / blockDim_y, 1), block(blockDim_x, blockDim_y, 1);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////// ^^^^ device allocation
   // cpu parallel parameter 

 

	//Auxiliary parameter
	int tStep = 0 ,i,k; 
	//int t_in = 0, Nwri, t_max;
	//int j_point;

    int numOfRowPerPro;
	if (NumCPU_CORE > 1)	{		numOfRowPerPro = NZ / (NumCPU_CORE - 1);	}
	else	{		numOfRowPerPro = NZ;	} 

	int LRhoGlobal = (NX + 2)*(NY + 2)*(NZ + 2);
	int LDisGlobal = (NX + 2)*(NY + 2)*(NZ + 2) * 19;
	int LRhoLocal = (NX + 2)*(NY + 2)*(numOfRowPerPro + 2);
	int LDisLocal = (NX + 2)*(NY + 2)*(numOfRowPerPro + 2) * 19;
	int numExRho = (NX + 2)*(NY + 2);
	int numExDis = (NX + 2)*(NY + 2) * 19; 
	// exchange control
	int ID_left, ID_right;
		int   j_startR;
//	int iz_start, j_startD, L_toTrans_single_local, L_toTrans_nineteen_local;
//	int   L_toTrans_single, L_toTrans_nineteen;
	int   L_single_extend, L_nineteen_extend;
	int   L_single_significant;
//	int L_nineteen_significant;



//PHYSICAL MODEL PARAMETERS
//	double  Re = 1000.0;
	

	//EXTERNAL PARAMETERS
	double   f_body[3];





//EVALUATION PARAMETERS
	//double  error;

	//PARAMETERS
	//Global

	int *image, *imageT;
	//local
    int *imageLocal;
	



	//local GPU
	int *imageG;
	double *exG, *eyG, *ezG, *t_kG, *f_bodyG,  *rsqG;

	///////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/  VVVVVVVVVVVVVVVVVVV
double tau=1.0;
	double   *uxS, *uyS, *uzS, *rhoS;
	double   *uxS_old, *uyS_old, *uzS_old, *rhoS_old;
	double   *fS;
	int *imageS;

	//int *imageSLocal;
	double   *uxSLocal, *uySLocal, *uzSLocal, *rhoSLocal ;
	double   *uxS_oldLocal, *uyS_oldLocal, *uzS_oldLocal, *rhoS_oldLocal;
	double   *fSLocal ;

 	double   *uxSG, *uySG, *uzSG, *rhoSG;
	double   *uxS_oldG, *uyS_oldG, *uzS_oldG, *rhoS_oldG;
	double   *fSG, *fShG;
	//int *imageSLG;

	double   *fSBCSend,  *fSBCReceive ,*fSToSend,*fSToReceive;
///////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/  ^^^^^^^^^^^^^^^^^^^^^^^^^^^	
 


	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>-	
	cc = 1;
	c_squ = cc*cc / 3;

	t_k[0] = 1.0 / 3.0; rsq[0] = 0.0;
	for (k = 1; k <= 6; k++)
	{
		t_k[k] = 1.0 / 18.0;      rsq[k] = 1.0;
	}
	for (k = 7; k <= 18; k++)
	{
		t_k[k] = 1.0 / 36.0;      rsq[k] = sqrt(2.0);
	}

	f_body[0] = 0.0;
	f_body[1] = 0.0;
	f_body[2] = 0.00001;




//=====================================================================================================
//== master core 
	if (ID_CPU_CORE == 0)//  allot global parameter  sent subblock to sub processors 
	{
		///////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/  VVVVVVVVVVVVVVVVVVV
		_malloc_host_np(&fS,LDisGlobal);
		_malloc_host_np(&fSToSend,LDisLocal);
		_malloc_host_np(&uxS, &uyS, &uzS, &uxS_old, &uyS_old, &uzS_old,  LRhoGlobal);
		_malloc_host_np(&rhoS, &rhoS_old, LRhoGlobal);
		_malloc_host_np(&imageS,  LRhoGlobal);
	 	
///////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/  ^^^^^^^^^^^^^^^^^^^^^^^

		//initial
		setImageSimpleBlock(image);

	///////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/  VVVVVVVVVVVVVVVVVVV
	    initialSinglePhase_3D(image, uxS,  uyS,  uzS,  rhoS,  fS,  1.0 ,	t_k,ex, ey, ez,  NX, NY,  NZ);
		const int dir_er = mkdir("testSINGLEPHASE", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
///////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/  ^^^^^^^^^^^^^^^^^^^^^^^	


		if (NumCPU_CORE > 1)
		{
			for (i = 1; i < NumCPU_CORE; i++)
			{
				j_startR = numExRho*(i-1) *numOfRowPerPro ;
				L_single_extend = numExRho*(numOfRowPerPro + 2);
				
				L_nineteen_extend = numExDis*(numOfRowPerPro + 2) ;	

				MPI_Send(&rhoS[j_startR], L_single_extend, MPI_DOUBLE, i, TAG_rhoS, MPI_COMM_WORLD);////////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**
				assembleDisToSend(i, fS, fSToSend, NX, NY, NZ, numOfRowPerPro);////////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**
				MPI_Send(fSToSend, L_nineteen_extend, MPI_DOUBLE, i, TAG_fS, MPI_COMM_WORLD);////////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**

			} 
		}
	}
//	//== master core 	
//	//=====================================================================================================

//	//== slave core 
	if (ID_CPU_CORE > 0)    // if (ID_CPU_CORE == 0)// allot local parameter and calculate  // sub process
	{ 
	    

		_malloc_host_np(&imageLocal, LRhoLocal);

		 _malloc_host_np(&fSBCSend, &fSBCReceive,   numExDis);/////////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**
		_malloc_host_np(&fSToReceive , LDisLocal); ////////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**
		_malloc_host_np( &rhoSLocal, &uxSLocal, &uySLocal, &uzSLocal,  LRhoLocal);///////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**
		_malloc_host_np(&fSLocal,   LDisLocal);///////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**

				//GPU local

		_malloc_device_np(&imageG, LRhoLocal, numGPUs);

		_malloc_device_np(&t_kG, &exG, &eyG, &ezG, 19, numGPUs);
		_malloc_device_np(&Ci_rG, &Ci_bG, &BiG, &rsqG, 19, numGPUs);
		_malloc_device_np(&f_bodyG, 3, numGPUs);

		_malloc_device_np(&fSG, &fShG,  LDisLocal, numGPUs);// ////////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**
		_malloc_device_np(&uxSG, &uySG, &uzSG, &uxS_oldG, &uyS_oldG, &uzS_oldG, LRhoLocal, numGPUs);////////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**
		_malloc_device_np(&rhoSG,  &rhoS_oldG, LRhoLocal, numGPUs);////////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**

		// ALLOT ALL VARIABLE
		
		//receiv  		 
		L_single_extend = numExRho*(numOfRowPerPro + 2);		 
		L_nineteen_extend = numExDis*(numOfRowPerPro + 2);

		MPI_Recv(imageLocal, L_single_extend, MPI_INT, 0, TAG_image, MPI_COMM_WORLD, &status);


		MPI_Recv(rhoSLocal, L_single_extend, MPI_DOUBLE, 0, TAG_rhoS, MPI_COMM_WORLD, &status);/////////////////////*/*/*/*/*/*////*/*/*/*/*/*/** 
		MPI_Recv(fSLocal, L_nineteen_extend, MPI_DOUBLE, 0, TAG_fS, MPI_COMM_WORLD, &status);/////////////////////*/*/*/*/*/*////*/*/*/*/*/*/** 


		// copy to GPU

		cudaMemcpy(imageG, imageLocal, LRhoLocal * sizeof(int), cudaMemcpyHostToDevice);


	    cudaMemcpy(fSG, fSLocal, LDisLocal * sizeof(double), cudaMemcpyHostToDevice);		/////////////////////*/*/*/*/*/*////*/*/*/*/*/*/** 
		cudaMemcpy(rhoSG, rhoSLocal, LRhoLocal * sizeof(double), cudaMemcpyHostToDevice);/////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**

		cudaMemcpy(exG, ex, 19 * sizeof(double), cudaMemcpyHostToDevice); //ex live in all core?
		cudaMemcpy(eyG, ey, 19 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(ezG, ez, 19 * sizeof(double), cudaMemcpyHostToDevice);

		cudaMemcpy(t_kG, t_k, 19 * sizeof(double), cudaMemcpyHostToDevice);// from master or slave?   declare directly in slave code.
		cudaMemcpy(rsqG, rsq, 19 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(f_bodyG, f_body, 3 * sizeof(double), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();


	    //outputUVTK_detectDisInRho(0, ID_CPU_CORE, imageLocal, f_bLocal, NX, NY, numOfRowPerPro, "./testRK/detectFRinRho_receivedLocal", " ", 1.0);
		//outputObst4D_View(0, ID_CPU_CORE, f_rLocal, "./testRK/fr_local_2received", NX, NY, numOfRowPerPro);
		//outputObst4D_View(0, ID_CPU_CORE, f_bLocal, "./testRK/fb_local_2received", NX, NY, numOfRowPerPro);
		//outputObst3D_View(ID_CPU_CORE, imageLocal, "./testRK/obsttest", NX, NY, numOfRowPerPro); // output test


	}
//



	for (tStep = 1; tStep <=3000; tStep++)
	{


		if (ID_CPU_CORE == 1)		{          cout << "tstep " << tStep << endl;		}		
		
		if (ID_CPU_CORE > 0)    
		{

		ID_GPU = ID_CPU_CORE - 1;
	//	cudaSetDevice(ID_GPU);
		//	f_body[2] = 0.0000075 + 0.0000005*sin(tStep / 200 * 2 * 3.14159265);
			f_body[2] = 0.00005;
			cudaMemcpy(f_bodyG, f_body, 3 * sizeof(double), cudaMemcpyHostToDevice);

					cudaDeviceSynchronize();

					cudaMemcpy(fSLocal, fSG, LDisLocal *sizeof(double), cudaMemcpyDeviceToHost);////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**
					cudaMemcpy(rhoSLocal, rhoSG, LRhoLocal*sizeof(double), cudaMemcpyDeviceToHost);////////////////////*/*/*/*/*/*////*/*/*/*/*/*/**

					cudaDeviceSynchronize();

			//-----------------------------------^^^^
			if (ID_CPU_CORE>1)
			{
				ID_left = (NumCPU_CORE - 1) - (NumCPU_CORE - ID_CPU_CORE) % (NumCPU_CORE - 1);
				 


			    assembleDisBC_EX(ID_CPU_CORE, fSLocal, fSBCSend, NX, NY, numOfRowPerPro, 1);	//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/			 
	 			MPI_Send(fSBCSend, numExDis, MPI_DOUBLE, ID_left, TAG_fS, MPI_COMM_WORLD);		//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/		
				MPI_Send(&rhoSLocal[numExRho], numExRho, MPI_DOUBLE, ID_left, TAG_rhoS, MPI_COMM_WORLD);//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/





				MPI_Recv(fSBCReceive, numExDis, MPI_DOUBLE, ID_left, TAG_fS, MPI_COMM_WORLD, &status);	//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/				
				assemble_BC_Local_dis(ID_CPU_CORE, fSBCReceive, fSLocal, NX, NY, numOfRowPerPro, 0);  //////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/	
				MPI_Recv(rhoSLocal, numExRho, MPI_DOUBLE, ID_left, TAG_rhoS, MPI_COMM_WORLD, &status);  //////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/	
				//MPI_Send(&imageLocal[numExRho], numExRho, MPI_INT, ID_left, TAG_image, MPI_COMM_WORLD);    //           need to be adjusted.
             }
			////  receive data from "right-hand" neighbor.
			if (ID_CPU_CORE<NumCPU_CORE - 1)
			{
				ID_right = (ID_CPU_CORE) % (NumCPU_CORE - 1) + 1;
				 


				MPI_Recv(fSBCReceive, numExDis, MPI_DOUBLE, ID_right, TAG_fS, MPI_COMM_WORLD, &status);		//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/		 			 
				assemble_BC_Local_dis(ID_CPU_CORE, fSBCReceive, fSLocal, NX, NY, numOfRowPerPro, numOfRowPerPro + 1);	//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
				MPI_Recv(&rhoSLocal[numExRho*(numOfRowPerPro + 1)], numExRho, MPI_DOUBLE, ID_right, TAG_rhoS, MPI_COMM_WORLD, &status);//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
				 



				assembleDisBC_EX(ID_CPU_CORE, fSLocal, fSBCSend, NX, NY, numOfRowPerPro, numOfRowPerPro);	//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/		 
				MPI_Send(fSBCSend, numExDis, MPI_DOUBLE, ID_right, TAG_fS, MPI_COMM_WORLD); //////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/	
				MPI_Send(&rhoSLocal[numExRho*numOfRowPerPro], numExRho, MPI_DOUBLE, ID_right, TAG_rhoS, MPI_COMM_WORLD); //////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/	
				
				//MPI_Recv(&imageLocal[numExRho*(numOfRowPerPro + 1)], numExRho, MPI_INT, ID_right, TAG_image, MPI_COMM_WORLD, &status);
			}				
 
		//	periodic boundary
			if (ID_CPU_CORE == 1)
			{
				ID_left = (NumCPU_CORE - 1) - (NumCPU_CORE - ID_CPU_CORE) % (NumCPU_CORE - 1);



				assembleDisBC_EX(ID_CPU_CORE, fSLocal, fSBCSend, NX, NY, numOfRowPerPro, 1);		//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/			 			 
				MPI_Send(fSBCSend, numExDis, MPI_DOUBLE, ID_left, TAG_fS, MPI_COMM_WORLD);//  	//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/	
				MPI_Send(&rhoSLocal[numExRho], numExRho, MPI_DOUBLE, ID_left, TAG_rhoS, MPI_COMM_WORLD);  //////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/	

	

				MPI_Recv(fSBCReceive, numExDis, MPI_DOUBLE, ID_left, TAG_fS, MPI_COMM_WORLD, &status);	//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/				 
				assemble_BC_Local_dis(ID_CPU_CORE, fSBCReceive, fSLocal, NX, NY, numOfRowPerPro, 0);	//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/	
				MPI_Recv(rhoSLocal, numExRho, MPI_DOUBLE, ID_left, TAG_rhoS, MPI_COMM_WORLD, &status);   //////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/	
				
				//MPI_Recv(imageLocal, numExRho, MPI_INT, ID_left, TAG_image, MPI_COMM_WORLD, &status);
			}

			if (ID_CPU_CORE == NumCPU_CORE - 1)
			{

				ID_right = (ID_CPU_CORE) % (NumCPU_CORE - 1) + 1;


				MPI_Recv(fSBCReceive, numExDis, MPI_DOUBLE, ID_right, TAG_fS, MPI_COMM_WORLD, &status);		//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/					
				assemble_BC_Local_dis(ID_CPU_CORE, fSBCReceive, fSLocal, NX, NY, numOfRowPerPro, numOfRowPerPro + 1);	//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
				MPI_Recv(&rhoSLocal[numExRho*(numOfRowPerPro + 1)], numExRho, MPI_DOUBLE, ID_right, TAG_rhoS, MPI_COMM_WORLD, &status);//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/

				
	

				assembleDisBC_EX(ID_CPU_CORE, fSLocal, fSBCSend, NX, NY, numOfRowPerPro, numOfRowPerPro);	//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/		 
				MPI_Send(fSBCSend, numExDis, MPI_DOUBLE, ID_right, TAG_fS, MPI_COMM_WORLD);	//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
				MPI_Send(&rhoSLocal[numExRho*numOfRowPerPro], numExRho, MPI_DOUBLE, ID_right, TAG_rhoS, MPI_COMM_WORLD);//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
				 
				//MPI_Send(&imageLocal[numExRho*numOfRowPerPro], numExRho, MPI_INT, ID_right, TAG_image, MPI_COMM_WORLD);
			}




				cudaMemcpy(fSG, fSLocal, LDisLocal *sizeof(double), cudaMemcpyHostToDevice);///////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
				cudaMemcpy(rhoSG, rhoSLocal, LRhoLocal*sizeof(double), cudaMemcpyHostToDevice);	///////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/


 
				cudaDeviceSynchronize();

				_Sream_Start << < grid, block >> >(fSG, fShG, imageG, NX, NY, numOfRowPerPro);   //////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
				cudaDeviceSynchronize();
				swap(&fSG, &fShG);//////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
				cudaDeviceSynchronize();
                _GetMacro3D_SF_LocalG<< < grid, block >> >(imageG, fSG,  uxSG,  uySG,  uzSG,  uxS_oldG,  uyS_oldG,  uzS_oldG, rhoSG,     NX,  NY,numOfRowPerPro);
				cudaDeviceSynchronize();		 
                _CollisionSF_localG << < grid, block >> >(imageG, uxSG,  uySG,  uzSG,rhoSG, fSG,  tau,   t_kG, exG, eyG, ezG,   c_squ,  f_bodyG, NX, NY, numOfRowPerPro);

				_Boundary_Wall_LocalG<< < grid, block>> >(fSG, imageG, NX, NY, numOfRowPerPro);    //new
 


			if (tStep % 50== 0)
			{
		 		MPI_Send(&imageLocal[numExRho], numExRho*numOfRowPerPro, MPI_INT, 0, TAG_image, MPI_COMM_WORLD);			
				
				
				cudaMemcpy(uzSLocal, uzSG, LRhoLocal*sizeof(double), cudaMemcpyDeviceToHost);/////////////////////////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
				MPI_Send(&uzSLocal[numExRho], numExRho*numOfRowPerPro, MPI_DOUBLE, 0, TAG_uzS, MPI_COMM_WORLD);
			}

		}//if (ID_CPU_CORE > 0)
		if (ID_CPU_CORE == 0)
		{

			if (tStep %50== 0)
			{
				for (i = 1; i < NumCPU_CORE; i++)
				{
					j_startR = numExRho*((i - 1)*numOfRowPerPro + 1);
					L_single_significant = numExRho*numOfRowPerPro;
	 				

					MPI_Recv(&uzS[j_startR], L_single_significant, MPI_DOUBLE, i, TAG_uzS, MPI_COMM_WORLD, &status);
				}

				

				// 			_outputUVTK_image(imageT, NX, NY, NZ, (char *)"./testRK/image_T_", 1.0);
				outputUVTK_singlevalue(tStep, image, phaseField, NX, NY, NZ, (char *) "./testRK/phaseField_end", (char *) " asdf", 1.0);
				//outputUVTK_singlevalue(tStep, image, rho_b, NX, NY, NZ,(char *)"./testRK/rho_b_end", (char *)" asdf", 1.0);
				//outputUVTK_singlevalue(tStep, image, uz, NX, NY, NZ, (char *)"./testRK/uz_end", (char *)" asdf", 1.0);

				outputUVTK_singlevalue(tStep, image, uzS, NX, NY, NZ, (char *) "./testSINGLEPHASE/UZ", (char *) " asdf", 1.0);

			}

			

		} //if (ID_CPU_CORE == 0)

		}  //______for (tStep = 1; tStep <= tMax; tStep++)

	
	 
	//== slave core 	
	//=====================================================================================================














 //-----------------------------------------
    // Finalize the MPI environment.
    MPI_Finalize();
return 0;
}





void setImageSimpleBlock(int *imageOver)
{
	int  j1 ;
 
	// double inX, inY, inZ, v_sin;
	int jx, jy, jz;
//	int NxD, NyD, NzD;
//	double segL, segR, zCenter, zHalfSpan, Ampt, Abase, jzz;
	 

//	NxD = (double)NX;
//	NyD = (double)NY;
//	NzD = (double)NZ;

//	segL = 160;
//	segR = 40;
//	zCenter = (NzD - segL - segR) / 2.0 + segL;
//	zHalfSpan = (NzD - segL - segR) / 2.0;
//	Ampt = NxD * 0.35 - 0.5;
//	Abase = NxD * 0.15 + 0.1;


	for (jz = 1; jz <= NZ; jz = jz + 1)
	for (jy = 1; jy <= NY; jy = jy + 1)
	for (jx = 1; jx <= NX; jx = jx + 1)
	{		
		j1 = (NX + 2)*(NY + 2)*jz + jy*(NX + 2) + jx;
		//imageOver[j1] = jx*jy*jz;
		imageOver[j1] = 0;
		//if (jz > jy)
		//{
  //        imageOver[j1] = 100;
		//}
		
		if (jx == 1 || jx == NX || jy == 1 || jy == NY)
		{
           imageOver[j1] = 5;
		}	
		//if (jy>jz)
		//{
		//	imageOver[j1] = 5;
		//}
	}

	//int j_pointLG, j_pointL, j_pointR, j_pointRG;
	//for (jy = 1; jy <= NY; jy = jy + 1)
	//for (jx = 1; jx <= NX; jx = jx + 1)
	//{
	//	j_pointLG = 0 * (NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
	//	j_pointL = (1) * (NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
	//	j_pointR = (NZ)* (NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
	//	j_pointRG = (NZ + 1) * (NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
	//	imageOver[j_pointLG] = imageOver[j_pointR];
	//	imageOver[j_pointRG] = imageOver[j_pointL];
	//}


	//fclose(fpin);
}




void initialSinglePhase_3D(int *image, double *ux, double *uy, double *uz, double *rho, double *f,  double rhoo ,
	double *t_k, double *ex, double *ey, double *ez,   int NX, int NY, int NZ)
{
	int jx, jy, jz, i, k, j_1;
	double   u_squ;
	double u_n[19]; 
	double fequ[19];

	int LineNum = (NY + 2)*(NX + 2)*(NZ + 2);

	for (jz = 1; jz <= NZ; jz++)
	for (jy = 1; jy <= NY; jy++)
	for (jx = 1; jx <= NX; jx++)
	{
		j_1 = jz*(NY + 2)*(NX + 2) + jy*(NX + 2) + jx;
		rho[j_1] = rhoo;		
		ux[j_1] = 0.0;
		uy[j_1] = 0.0;
		uz[j_1] = 0.0;

	}


	for (jz = 1; jz <= NZ; jz++)
	for (jy = 1; jy <= NY; jy++)
	for (jx = 1; jx <= NX; jx++)
	{

		j_1 = jz*(NY + 2)*(NX + 2) + jy*(NX + 2) + jx; 

		u_squ = ux[j_1] * ux[j_1] + uy[j_1] * uy[j_1] + uz[j_1] * uz[j_1];
		u_n[0] = 0.0;
		for (k = 1; k < 19; k++)
		{
			u_n[k] = ex[k] * ux[j_1] + ey[k] * uy[j_1] + ez[k] * uz[j_1];
		}
		 
		for (k = 0; k < 19; k++)
		{
			fequ[k] = t_k[k] * rho[j_1] * (1.0 + u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ*1.5) ;
		}

		for (i = 0; i < 19; i++)
		{
			 
			if (image[j_1] != 5)
			{
				f[j_1 + i * LineNum] = fequ[i];				
			}
			else
			{
				f[j_1 + i * LineNum] = 0;
				
			}
		}

	}
}


__global__ void _GetMacro3D_SF_LocalG(int *image, double *fL, double *uxL, double *uyL, double *uzL, double *ux_oldL, double *uy_oldL, double *uz_oldL,
	double *rhoL,    int NX, int NY, int NZ_sub)
{
	int jx, jy, jz, j_p;	 
	int sizeOfCube = (NY + 2)*(NX + 2)*(NZ_sub + 2);

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZ_sub; jz++)	 
	{
		j_p = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx; 

		if (image[j_p] != 5)
		{

			ux_oldL[j_p] = uxL[j_p];
			uy_oldL[j_p] = uyL[j_p];
			uz_oldL[j_p] = uzL[j_p];				

			rhoL[j_p] = fL[j_p]
				+ fL[j_p + 1 * sizeOfCube] + fL[j_p + 2 * sizeOfCube] + fL[j_p + 3 * sizeOfCube] + fL[j_p + 4 * sizeOfCube] + fL[j_p + 5 * sizeOfCube] + fL[j_p + 6 * sizeOfCube]
				+ fL[j_p + 7 * sizeOfCube] + fL[j_p + 8 * sizeOfCube] + fL[j_p + 9 * sizeOfCube] + fL[j_p + 10 * sizeOfCube] + fL[j_p + 11 * sizeOfCube] + fL[j_p + 12 * sizeOfCube]
				+ fL[j_p + 13 * sizeOfCube] + fL[j_p + 14 * sizeOfCube] + fL[j_p + 15 * sizeOfCube] + fL[j_p + 16 * sizeOfCube] + fL[j_p + 17 * sizeOfCube] + fL[j_p + 18 * sizeOfCube];		 


			uxL[j_p] = (fL[j_p + 1 * sizeOfCube] - fL[j_p + 2 * sizeOfCube] + fL[j_p + 7 * sizeOfCube] + fL[j_p + 8 * sizeOfCube] - fL[j_p + 9 * sizeOfCube]
				- fL[j_p + 10 * sizeOfCube] + fL[j_p + 11 * sizeOfCube] - fL[j_p + 12 * sizeOfCube] + fL[j_p + 13 * sizeOfCube] - fL[j_p + 14 * sizeOfCube])	 / rhoL[j_p];   // try for(1-19)

			uyL[j_p] = (fL[j_p + 3 * sizeOfCube] - fL[j_p + 4 * sizeOfCube] + fL[j_p + 7 * sizeOfCube] - fL[j_p + 8 * sizeOfCube] + fL[j_p + 9 * sizeOfCube]
				- fL[j_p + 10 * sizeOfCube] + fL[j_p + 15 * sizeOfCube] + fL[j_p + 16 * sizeOfCube] - fL[j_p + 17 * sizeOfCube] - fL[j_p + 18 * sizeOfCube]) / rhoL[j_p];

			uzL[j_p] = (fL[j_p + 5 * sizeOfCube] - fL[j_p + 6 * sizeOfCube] + fL[j_p + 11 * sizeOfCube] + fL[j_p + 12 * sizeOfCube] - fL[j_p + 13 * sizeOfCube]
				- fL[j_p + 14 * sizeOfCube] + fL[j_p + 15 * sizeOfCube] - fL[j_p + 16 * sizeOfCube] + fL[j_p + 17 * sizeOfCube] - fL[j_p + 18 * sizeOfCube]) / rhoL[j_p];
		}
		else
		{		
			rhoL[j_p] = 1.0;
			uxL[j_p] = 0.0;
			uyL[j_p] = 0.0;
			uzL[j_p] = 0.0;
		}
	}
}

__global__ void _CollisionSF_localG (int *imageL, double *uxLocal, double *uyLocal, double *uzLocal, double *rhoLocal, double *fLocal,  
	double tau,  double *t_k, double *ex, double *ey, double *ez,double c_squ,  double *f_body, int NX, int NY, int NZ_sub)
{
	int jx, jy, jz, j1, i;
	double fequ[19];
	int LineNum = (NY + 2)*(NX + 2)*(NZ_sub + 2);
	 
	 
	double u_n[19], u_squ;
	int k;  

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;		
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZ_sub; jz++)
	{		
		j1 = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
		if (imageL[j1] !=5)
		{
 
			////*******************compute Fequ			 
			u_squ = uxLocal[j1] * uxLocal[j1] + uyLocal[j1] * uyLocal[j1] + uzLocal[j1] * uzLocal[j1];
			u_n[0] = 0.0;
			for (k = 1; k < 19; k++)
			{
				u_n[k] = ex[k] * uxLocal[j1] + ey[k] * uyLocal[j1] + ez[k] * uzLocal[j1];
			}		 
		for (k = 0; k < 19; k++)
		{
			fequ[k] = t_k[k] * rhoLocal[j1] * (1.0 + u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ*1.5) ;
		}
  

			for (i = 0; i < 19; i++)
			{
				fLocal[j1 + i * LineNum] = fequ[i] + (1 - 1 / tau)*(fLocal[j1 + i * LineNum] - fequ[i])
				+ t_k[i] * ex[i] * f_body[0] / c_squ + t_k[i] * ey[i] * f_body[1] / c_squ + t_k[i] * ez[i] * f_body[2] / c_squ; 				  
			}
		}
	} //end for jz
}
