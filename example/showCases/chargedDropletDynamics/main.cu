#include "../../../src/lbm1.h" 
#include "../../../src/FDM.h"

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h> 
#include <mpi.h>
#include <math.h>
#include <cmath> 
#include <ctime>
#include <time.h> 
#include <iostream>
#include <sys/stat.h>  

#include <iomanip>  

using namespace std;  

int NX =64, NY =64, NZ = 60;
#define  blockDim_x       16
#define  blockDim_y       16   

 
double ex[19] = { 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, -1, 1, -1, 0, 0, 0, 0 };             //notice   point----------------------
double ey[19] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1 };
double ez[19] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, -1, 1, -1 };
double   t_k[19], rsq[19], Bi[19];
double *Ci_r, *Ci_b; 
 

void setImageSimpleBlock(int *imageOver);
void initialBlock_3D_RK(int *image, double *ux, double *uy, double *uz, double *rho_b, double *rho_r, double *f_r, double *f_b, double rho_bi, double rho_ri,
	double *t_k, double *ex, double *ey, double *ez, double alfa_r, double alfa_b, int NX, int NY, int NZ);
__global__ void _SORAG  ( int *imageG, double lamda, double *B,double *X_lastIter, double *x_new,    double omiga,int NX,int NY,int NZ    );
__global__ void _SORG_AssembleB(int *imageG,double Lambda,double alpha,double *B,double *Psi,double *ux,double *uy,double *uz,double uMax,double vMax, double wMax,double *detG,int NX,int NY,int NZ);
__global__ void _SORAG_CN  ( int *imageG, double lamda, double *B,double *X_lastIter, double *x_new,    double omiga,int NX,int NY,int NZ    );
__global__ void _SORAG_CN_Phase  ( int *imageG, double lamda, double *B,double *X_lastIter, double *x_new, double*x_old, double *detG,double *detPreG,  double omiga,int NX,int NY,int NZ    );
__global__ void _SORG_AssembleB_CN(int *imageG,double Lambda,double alpha,double *B,double *Psi,double *ux,double *uy,double *uz,double uMax,double vMax, double wMax,double *detG,double *sTerm,double *sTermPre,int NX,int NY,int NZ);

double getAbsMax(double *xMatrix,  int NX,int NY,int NZblock );
void initialPsi(double *phasefield,double *PsiSur, int NX,int NY,int NZblock )
{
	int jx, jy, jz,  j_1; 
	for (jz = 1; jz <= NZblock; jz++)
	for (jy = 1; jy <= NY; jy++)
	for (jx = 1; jx <= NX; jx++)
	{		 
		j_1 = jz*(NY + 2)*(NX + 2) + jy*(NX + 2) + jx;
			PsiSur[j_1] =1.0*( 1.0-fabs(phasefield[j_1]));	
//if  ( 1.0-abs(phasefield[j_1])>0.5)
//PsiSur[j_1] =1.0;   
//PsiSur[j_1] =abs(phasefield[j_1]);      	
	}	
}
void setPhaseField3D_boundaryBlue(int *image, double *phaseField, double *rho_r, double *rho_b, int NX, int NY, int NZ);

__global__ void getStermBlock_temp(double *Sterm, int *image, double *ux, double *uy, double *uz, double *Psi, double *Determinant,   
			double Ds,double alpha,double uMax,double vMax,double wMax, double dx,int NX, int NY, int NZblock);//no weno;//no weno;
__global__ void getStermBlock_temp_Phase(double *Sterm, int *image, double *ux, double *uy, double *uz, double *Psi,  double *detPreG, 
			double Ds,double alpha,double uMax,double vMax,double wMax, double dx,int NX, int NY, int NZblock);//no weno


__global__ void getStermBlock_temp_Phase_weno(double *Sterm, int *imageG, double *ux, double *uy, double *Psi,  double *detPreG, double *fluxZPositiveCap, double *fluxZNegativeCap,
		double *fluxXPositive,double *fluxXNegative,double *fluxYPositive,double *fluxYNegative,double *fluxXPositiveCap,double *fluxXNegativeCap,double *fluxYPositiveCap,double *fluxYNegativeCap, 
		double Ds,double alpha,double uMax,double vMax, double dx,int NX, int NY, int NZblock);
__global__ void getZFlux_weno(double *fluxZPositive,double *fluxZNegative,    double *uz, double *Psi,  double *detPreG,double wMax,  int NX, int NY, int NZblock);
__global__ void getZFluxCap_weno(double *fluxZPositiveCap,double *fluxZNegativeCap, int *imageG,   double *fluxZPositive,double *fluxZNegative  ,int NX, int NY, int NZblock);

void getDETofGradientBlockCPU(double *Determinant, int *image, double *phaseField, int NX, int NY, int NZ) ; //get | \-/ pN |
__global__ void _SORG_AssembleB_CN_Phase(int *imageG,double Lambda,double alpha,double *B,double *Psi,double *ux,double *uy,double *uz,double uMax,double vMax, double wMax,double *detPreG,double *sTerm,double *sTermPre,int NX,int NY,int NZ);

//temporary put here
__global__ void _GetWenoFlux(DOUBLE *FlunxPositive, DOUBLE *FlunxNegative, int *imageG,  double *Psi, double *u, double uMax,double *DET,int NX, int NY, int NZblock);

int main(int argc, char** argv) {   
   // cpu parallel parameter
	int  NumCPU_CORE, ID_CPU_CORE;
	int  numGPUs,ID_GPU ;
	MPI_Status status;
//	cudaError_t cudaStatus;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &ID_CPU_CORE);
	MPI_Comm_size(MPI_COMM_WORLD, &NumCPU_CORE);
    cudaGetDeviceCount(&numGPUs); 
	
    MPICUDAInitial( NumCPU_CORE,ID_CPU_CORE,  numGPUs, &ID_GPU)	;
	dim3  grid(NX / blockDim_x, NY / blockDim_y, 1), block(blockDim_x, blockDim_y, 1);

   string dataFold="temp";
   string imageName="./"+dataFold+"/image";
   string phaseFieldName="./"+dataFold+"/phaseField";
   string PsiName="./"+dataFold+"/Psi";//FDM-TRANSPORT
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////// ^^^^ device allocation

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

//PHYSICAL MODEL PARAMETERS
//	double  Re = 1000.0;
	double  A, beta;
	double  rho_ri, rho_bi;//     rsq_model of direction vector
	double  alfa_r, alfa_b, taur, taub;

	double   cc, c_squ, c_squ_b, c_squ_r;

	//EXTERNAL PARAMETERS
	double   f_body[3];
    double *forceGlobalX,*forceGlobalY,*forceGlobalZ;//FDM-TRANSPORT
    double *forceGlobalXL,*forceGlobalYL,*forceGlobalZL;//FDM-TRANSPORT
    double *forceGlobalXG,*forceGlobalYG,*forceGlobalZG;//FDM-TRANSPORT
    double deltaT=0.1,deltaX=1;//Ds=0.1,
	//PARAMETERS
	//Global
	int *image;
	double   *ux, *uy, *uz, *rho_b, *rho_r, *rho, *phaseField;
	double   *ux_old, *uy_old, *uz_old, *rho_old;
	double   *f_r, *f_b, *ff;

    double *PsiSur,*PsiSurNew,*PsiLastIter;//FDM-TRANSPORT

	//local
    int *imageLocal;
	double   *uxLocal, *uyLocal, *uzLocal, *ux_oldLocal, *uy_oldLocal, *uz_oldLocal;
	double    *rho_bLocal, *rho_rLocal, *rhoLocal, *rho_oldLocal, *phaseFieldLocal;
	double   *f_rLocal, *f_bLocal, *ffLocal, *f_rhLocal, *f_bhLocal;

	double   *f_rToSend, *f_bToSend, *f_rToReceive, *f_bToReceive;
	double   *f_rBCSend, *f_bBCSend, *f_rBCReceive, *f_bBCReceive;
	double *FxLocal, *FyLocal, *FzLocal;

double *PsiSurL,*PsiSurNewL,*PsiLastIterL ;//FDM-TRANSPORT
double uMax,vMax,wMax;
double *fluxZPositiveL, *fluxZNegativeL; 
double *fluxZPositiveCapL, *fluxZNegativeCapL;

	//local GPU
	int *imageG;
	double   *uxG, *uyG, *uzG, *rho_bG, *rho_rG, *rhoG;
	double   *ux_oldG, *uy_oldG, *uz_oldG, *rho_oldG;
	double   *f_rG, *f_bG, *ffG, *f_rhG, *f_bhG;
	double *FxG, *FyG, *FzG;
	double *exG, *eyG, *ezG, *t_kG, *f_bodyG, *Ci_rG, *Ci_bG, *BiG, *rsqG;

double *PsiSurG,*PsiSurNewG,*PsiLastIterG,*bMatrixG; //FDM-TRANSPORT
double *det,*detL,*detG, *detPreG,*sTermG, *sTermPreG;//FDM-TRANSPORT

double *fluxZPositiveG, *fluxZNegativeG; 
double *fluxZPositiveCapG, *fluxZNegativeCapG;
double *fluxXPositiveG, *fluxXNegativeG,  *fluxYPositiveG,  *fluxYNegativeG,  *fluxXPositiveCapG, *fluxXNegativeCapG,  *fluxYPositiveCapG, *fluxYNegativeCapG;
 

_malloc_host_np(&Ci_r, &Ci_b, 19);
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>-	
	cc = 1;
	c_squ = cc*cc / 3;

	t_k[0] = 1.0 / 3.0; Bi[0] = -1.0 / 3.0; rsq[0] = 0.0;
	for (int k = 1; k <= 6; k++){	t_k[k] = 1.0 / 18.0;    Bi[k] = 1.0 / 18.0;   rsq[k] = 1.0;	}
	for (int k = 7; k <= 18; k++){	t_k[k] = 1.0 / 36.0;    Bi[k] = 1.0 / 36.0;   rsq[k] = sqrt(2.0);}
        beta = 0.3;                   //parameter in recoloring process
	A = 0.001;	 
	f_body[0] = 0.0;
	f_body[1] = 0.00000;
	f_body[2] = 0.0000;
	alfa_r = 0.5;
	alfa_b = 0.5;
	rho_ri = 1.0 - alfa_b;
	rho_bi = 1.0 - alfa_r;
	c_squ_r = 3.0*(1.0 - alfa_r) / 5.0;//???????????????????????????????????????
	c_squ_b = 3.0*(1.0 - alfa_b) / 5.0;
	taur = 1.0;
	taub = 0.9;

	Ci_r[0] = alfa_r; Ci_b[0] = alfa_b;
	for (int k = 1; k <= 6; k++)	{		Ci_r[k] = (1.0 - alfa_r) / 12.0;    Ci_b[k] = (1.0 - alfa_b) / 12.0;	}
	for (int k = 7; k <= 18; k++)	{		Ci_r[k] = (1.0 - alfa_r) / 24.0;    Ci_b[k] = (1.0 - alfa_b) / 24.0;	} 

//=====================================================================================================
//== master core 
	if (ID_CPU_CORE == 0)//  allot global parameter  sent subblock to sub processors 
	{
		_malloc_host_np(&f_r, &f_b, &ff, LDisGlobal);
		_malloc_host_np(&f_rToSend, &f_bToSend, LDisLocal);

		_malloc_host_np(&ux, &uy, &uz, &ux_old, &uy_old, &uz_old,  LRhoGlobal);
		_malloc_host_np(&rho_b, &rho_r, &rho, &rho_old, &phaseField,  LRhoGlobal);
		_malloc_host_np(&image,  LRhoGlobal);
    
        _malloc_host_np(&forceGlobalX, &forceGlobalY, &forceGlobalZ, LRhoGlobal);//FDM-TRANSPORT
		_malloc_host_np(&PsiSur,&PsiSurNew,&PsiLastIter, &det, LRhoGlobal);//FDM-TRANSPORT
		 
    

		//initial
		setImageSimpleBlock(image);		
		initialBlock_3D_RK(image, ux, uy, uz, rho_b, rho_r, f_r, f_b, rho_bi, rho_ri, t_k, ex, ey, ez, alfa_r, alfa_b, NX, NY, NZ);
	 //	 initial_3D(PsiSur );//PUT THIS SECTION AFTER PHASEFIELD FORMS  IN THE LOOP AFTER 100
       
        createDirectory( "testRK" ) ; 
        createDirectory( (char*)dataFold.c_str() ) ; 

        writeDataArray_vtk(image, NX, NY, NZ, (char *) "./testRK/image_original_", 1.0);	
        writeDataArray_vtk(image, NX, NY, NZ, (char*)imageName.c_str() , 1.0);			
   

		if (NumCPU_CORE > 1)
		{
			for (int i = 1; i < NumCPU_CORE; i++)
			{
			int	j_startR = numExRho*(i-1) *numOfRowPerPro ;
			//int	L_single_extend = numExRho*(numOfRowPerPro + 2);
           // int	L_nineteen_extend = numExDis*(numOfRowPerPro + 2) ;		
            int	DimBlockEX = numExRho*(numOfRowPerPro + 2);	
            int	DimDisBlockEX = numExDis*(numOfRowPerPro + 2);	  
				
					 

				MPI_Send(&image[j_startR], DimBlockEX, MPI_INT, i, TAG_image, MPI_COMM_WORLD);
				MPI_Send(&rho_r[j_startR], DimBlockEX, MPI_DOUBLE, i, TAG_rho_r, MPI_COMM_WORLD);
				MPI_Send(&rho_b[j_startR], DimBlockEX, MPI_DOUBLE, i, TAG_rho_b, MPI_COMM_WORLD);

			 	assembleDisToSend(i, f_r, f_rToSend, NX, NY, NZ, numOfRowPerPro);
			 	assembleDisToSend(i, f_b, f_bToSend, NX, NY, NZ, numOfRowPerPro);
				MPI_Send(f_rToSend, DimDisBlockEX, MPI_DOUBLE, i, TAG_f_r, MPI_COMM_WORLD);
				MPI_Send(f_bToSend, DimDisBlockEX, MPI_DOUBLE, i, TAG_f_b, MPI_COMM_WORLD);

			} 
		}
	}
//	//== master core 	
//	//=====================================================================================================

//	//== slave core 
	if (ID_CPU_CORE > 0)    // if (ID_CPU_CORE == 0)// allot local parameter and calculate  // sub process
	{     

		_malloc_host_np(&imageLocal, LRhoLocal);
		_malloc_host_np(&f_rLocal, &f_bLocal, &ffLocal, &f_rhLocal, &f_bhLocal, LDisLocal);
		_malloc_host_np(&uxLocal, &uyLocal, &uzLocal, &ux_oldLocal, &uy_oldLocal, &uz_oldLocal, LRhoLocal);
		_malloc_host_np(&rho_bLocal, &rho_rLocal, &rhoLocal, &rho_oldLocal, &phaseFieldLocal, LRhoLocal);
		_malloc_host_np(&FxLocal, &FyLocal, &FzLocal, LRhoLocal);

		_malloc_host_np(&f_rBCSend, &f_bBCSend, &f_rBCReceive, &f_bBCReceive, numExDis);
		_malloc_host_np(&f_rToReceive, &f_bToReceive, LDisLocal);

 		_malloc_host_np(&PsiSurL,&PsiSurNewL,&PsiLastIterL, &detL, LRhoLocal);//FDM-TRANSPORT
		_malloc_host_np(&fluxZPositiveL,&fluxZNegativeL,&fluxZPositiveCapL, &fluxZNegativeCapL, LRhoLocal);

 
		 

				//GPU local
		_malloc_device_np(&imageG, LRhoLocal, numGPUs);

		_malloc_device_np(&f_rG, &f_bG, &ffG, &f_rhG, &f_bhG, LDisLocal, numGPUs);
		_malloc_device_np(&uxG, &uyG, &uzG, &ux_oldG, &uy_oldG, &uz_oldG, LRhoLocal, numGPUs);
		_malloc_device_np(&rho_bG, &rho_rG, &rhoG, &rho_oldG, LRhoLocal, numGPUs);
		_malloc_device_np(&FxG, &FyG, &FzG, LRhoLocal, numGPUs);// number********************************************************

		_malloc_device_np(&t_kG, &exG, &eyG, &ezG, 19, numGPUs);
		_malloc_device_np(&Ci_rG, &Ci_bG, &BiG, &rsqG, 19, numGPUs);
		_malloc_device_np(&f_bodyG, 3, numGPUs);

		_malloc_device_np(&PsiSurG,&PsiSurNewG,&PsiLastIterG,&bMatrixG,&detG,&detPreG,LRhoLocal, numGPUs);
		_malloc_device_np(&sTermG,&sTermPreG,LRhoLocal, numGPUs); 	

_malloc_device_np(&fluxZPositiveG,&fluxZNegativeG,&fluxZPositiveCapG, &fluxZNegativeCapG,LRhoLocal, numGPUs); 
_malloc_device_np(&fluxXPositiveG,&fluxXNegativeG,&fluxYPositiveG, &fluxYNegativeG,LRhoLocal, numGPUs);
_malloc_device_np(&fluxXPositiveCapG,&fluxXNegativeCapG,&fluxYPositiveCapG, &fluxYNegativeCapG,LRhoLocal, numGPUs);		

 
		// ALLOT ALL VARIABLE
		
		//receiv  		 
		int DimBlockEX = numExRho*(numOfRowPerPro + 2);		 
		int DimDisBlockEX  = numExDis*(numOfRowPerPro + 2);

		MPI_Recv(imageLocal, DimBlockEX, MPI_INT, 0, TAG_image, MPI_COMM_WORLD, &status);
		MPI_Recv(rho_rLocal, DimBlockEX, MPI_DOUBLE, 0, TAG_rho_r, MPI_COMM_WORLD, &status);
		MPI_Recv(rho_bLocal, DimBlockEX, MPI_DOUBLE, 0, TAG_rho_b, MPI_COMM_WORLD, &status);
		MPI_Recv(f_rLocal, DimDisBlockEX, MPI_DOUBLE, 0, TAG_f_r, MPI_COMM_WORLD, &status);
		MPI_Recv(f_bLocal, DimDisBlockEX, MPI_DOUBLE, 0, TAG_f_b, MPI_COMM_WORLD, &status);


		// copy to GPU

		cudaMemcpy(imageG, imageLocal, LRhoLocal * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(f_rG, f_rLocal, LDisLocal * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(f_bG, f_bLocal, LDisLocal * sizeof(double), cudaMemcpyHostToDevice);
		//cudaMemcpy(ffG, ff, L_f * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(rho_rG, rho_rLocal, LRhoLocal * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(rho_bG, rho_bLocal, LRhoLocal * sizeof(double), cudaMemcpyHostToDevice);

		cudaMemcpy(exG, ex, 19 * sizeof(double), cudaMemcpyHostToDevice); //ex live in all core?
		cudaMemcpy(eyG, ey, 19 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(ezG, ez, 19 * sizeof(double), cudaMemcpyHostToDevice);

		cudaMemcpy(t_kG, t_k, 19 * sizeof(double), cudaMemcpyHostToDevice);// from master or slave?   declare directly in slave code.
		cudaMemcpy(Ci_rG, Ci_r, 19 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(Ci_bG, Ci_b, 19 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(BiG, Bi, 19 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(rsqG, rsq, 19 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(f_bodyG, f_body, 3 * sizeof(double), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();
	}
//



for (int tStep = 1; tStep <=15000; tStep++)
{ 
 //---------------------------------------------------------------vv
	if (tStep == 100)
	{								//assmeble phasefield  initialize Psi
		if (ID_CPU_CORE > 0)
		{ 
			initialPsi(phaseFieldLocal, PsiSurL,   NX, NY,numOfRowPerPro );
			cudaMemcpy(PsiSurG, PsiSurL, LRhoLocal*sizeof(double), cudaMemcpyHostToDevice);
			
			//MPI_Send(&phaseFieldLocal[numExRho], numExRho*numOfRowPerPro, MPI_DOUBLE, 0, TAG_phaseField, MPI_COMM_WORLD);
			MPI_Send(&PsiSurL[numExRho], numExRho*numOfRowPerPro, MPI_DOUBLE, 0, TAG_PsiSur, MPI_COMM_WORLD);
			getZFlux_weno<< < grid, block>> >(fluxZPositiveG, fluxZNegativeG, uzG, PsiSurG, detPreG, wMax,   NX,  NY, numOfRowPerPro);

		}
		if (ID_CPU_CORE == 0)  // this is to view psi   
		{
			for (int i = 1; i < NumCPU_CORE; i++)
			{
				int j_startR = numExRho*((i - 1)*numOfRowPerPro + 1);
				int DimBlock  = numExRho*numOfRowPerPro;	 
				//MPI_Recv(&phaseField[j_startR], DimBlock, MPI_DOUBLE, i, TAG_phaseField, MPI_COMM_WORLD, &status); 
				MPI_Recv(&PsiSur[j_startR], DimBlock, MPI_DOUBLE, i, TAG_PsiSur, MPI_COMM_WORLD, &status);
			}
		// writeData_ArraybyImage_vtk(tStep, image, phaseField, NX, NY, NZ,   (char *)"./temp/phaseField_test", (char *) " asdf", 1.0);
			writeData_ArraybyImage_vtk(tStep, image, PsiSur, NX, NY, NZ,   (char *)"./temp/PsiSurt", (char *) " asdf", 1.0);
		}
	}//if (tStep == 100)
 //---------------------------------------------------------------^^

		if (ID_CPU_CORE == 1&&tStep%10==0)		{          cout << "tstep " << tStep << endl;		}		
		
		if (ID_CPU_CORE > 0)    
		{		 
		    //f_body[2] = 0.0000075 + 0.0000005*sin(tStep / 200 * 2 * 3.14159265);
			//f_body[2] = 0.0000001;
              f_body[1] =-0.0001;
// f_body[1] =-0.0000;
 getZFlux_weno<< < grid, block>> >(fluxZPositiveG, fluxZNegativeG,     uzG, PsiSurG,   detPreG, wMax,   NX,  NY, numOfRowPerPro);
			cudaMemcpy(f_bodyG, f_body, 3 * sizeof(double), cudaMemcpyHostToDevice);

			cudaDeviceSynchronize();
			cudaMemcpy(f_rLocal, f_rG, LDisLocal*sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(f_bLocal, f_bG, LDisLocal *sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(rho_rLocal, rho_rG, LRhoLocal*sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(rho_bLocal, rho_bG, LRhoLocal*sizeof(double), cudaMemcpyDeviceToHost);

			cudaMemcpy(uxLocal, uxG, LRhoLocal*sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(uyLocal, uyG, LRhoLocal*sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(uzLocal, uzG, LRhoLocal*sizeof(double), cudaMemcpyDeviceToHost);

			if (tStep >= 100)
			{
				cudaMemcpy(PsiSurL, PsiSurG, LRhoLocal*sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(fluxZPositiveL, fluxZPositiveG, LRhoLocal*sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(fluxZNegativeL, fluxZNegativeG, LRhoLocal*sizeof(double), cudaMemcpyDeviceToHost);  
			}
			cudaDeviceSynchronize();


			//setPhaseField3D(imageLocal, phaseFieldLocal, rho_rLocal, rho_bLocal, NX, NY, numOfRowPerPro);
setPhaseField3D_boundaryBlue(imageLocal, phaseFieldLocal, rho_rLocal, rho_bLocal, NX, NY, numOfRowPerPro);
 
// data exchange between blocks//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            int left, right;
			if (ID_CPU_CORE>1) left=ID_CPU_CORE-1;
			else left=MPI_PROC_NULL;			
			if (ID_CPU_CORE < NumCPU_CORE-1) right=ID_CPU_CORE+1;
			else right=MPI_PROC_NULL; 
 // data left-->right
				assembleDisBC_EX(ID_CPU_CORE, f_rLocal, f_rBCSend, NX, NY, numOfRowPerPro, numOfRowPerPro);
				assembleDisBC_EX(ID_CPU_CORE, f_bLocal, f_bBCSend, NX, NY, numOfRowPerPro, numOfRowPerPro);
			MPI_Sendrecv(f_rBCSend,numExDis,MPI_DOUBLE,right,TAG_f_r,   f_rBCReceive,  numExDis,MPI_DOUBLE,left, TAG_f_r,  MPI_COMM_WORLD,&status);
			MPI_Sendrecv(f_bBCSend,numExDis,MPI_DOUBLE,right,TAG_f_b,   f_bBCReceive,  numExDis,MPI_DOUBLE,left, TAG_f_b,  MPI_COMM_WORLD,&status);				 
				assemble_BC_Local_dis(ID_CPU_CORE, f_rBCReceive, f_rLocal, NX, NY, numOfRowPerPro, 0);
				assemble_BC_Local_dis(ID_CPU_CORE, f_bBCReceive, f_bLocal, NX, NY, numOfRowPerPro, 0);

			 MPI_Sendrecv(&rho_rLocal[numExRho*numOfRowPerPro],numExRho,MPI_DOUBLE,right,TAG_rho_r, rho_rLocal, numExRho,MPI_DOUBLE,left, TAG_rho_r,  MPI_COMM_WORLD,&status);
			 MPI_Sendrecv(&rho_bLocal[numExRho*numOfRowPerPro],numExRho,MPI_DOUBLE,right,TAG_rho_b, rho_bLocal, numExRho,MPI_DOUBLE,left, TAG_rho_b,  MPI_COMM_WORLD,&status);

MPI_Sendrecv(&phaseFieldLocal[numExRho*numOfRowPerPro],numExRho,MPI_DOUBLE,right,TAG_phaseField, phaseFieldLocal, numExRho,MPI_DOUBLE,left, TAG_phaseField,  MPI_COMM_WORLD,&status);
if (tStep >= 100){ 
MPI_Sendrecv(&PsiSurL[numExRho*numOfRowPerPro],numExRho,MPI_DOUBLE,right,TAG_PsiSur, PsiSurL, numExRho,MPI_DOUBLE,left, TAG_PsiSur,  MPI_COMM_WORLD,&status);
MPI_Sendrecv(&fluxZPositiveL[numExRho*numOfRowPerPro],numExRho,MPI_DOUBLE,right,TAG_fluxZPositive, fluxZPositiveL, numExRho,MPI_DOUBLE,left, TAG_fluxZPositive,  MPI_COMM_WORLD,&status);
MPI_Sendrecv(&fluxZNegativeL[numExRho*numOfRowPerPro],numExRho,MPI_DOUBLE,right,TAG_fluxZNegative, fluxZNegativeL, numExRho,MPI_DOUBLE,left, TAG_fluxZNegative,  MPI_COMM_WORLD,&status);
}
			 
//data  left<--right
				assembleDisBC_EX(ID_CPU_CORE, f_rLocal, f_rBCSend, NX, NY, numOfRowPerPro, 1);
				assembleDisBC_EX(ID_CPU_CORE, f_bLocal, f_bBCSend, NX, NY, numOfRowPerPro, 1);
			MPI_Sendrecv(f_rBCSend, numExDis,MPI_DOUBLE, left,TAG_f_r, f_rBCReceive,numExDis,MPI_DOUBLE,right,TAG_f_r,  MPI_COMM_WORLD,&status);
			MPI_Sendrecv(f_bBCSend, numExDis,MPI_DOUBLE, left,TAG_f_b, f_bBCReceive,numExDis,MPI_DOUBLE,right,TAG_f_b,  MPI_COMM_WORLD,&status); 				 
				assemble_BC_Local_dis(ID_CPU_CORE, f_rBCReceive, f_rLocal, NX, NY, numOfRowPerPro, numOfRowPerPro + 1);
				assemble_BC_Local_dis(ID_CPU_CORE, f_bBCReceive, f_bLocal, NX, NY, numOfRowPerPro, numOfRowPerPro + 1);

			 MPI_Sendrecv(&rho_rLocal[numExRho], numExRho,MPI_DOUBLE, left,TAG_rho_r, &rho_rLocal[numExRho*(numOfRowPerPro + 1)],numExRho,MPI_DOUBLE,right,TAG_rho_r,  MPI_COMM_WORLD,&status);
			 MPI_Sendrecv(&rho_bLocal[numExRho], numExRho,MPI_DOUBLE, left,TAG_rho_b, &rho_bLocal[numExRho*(numOfRowPerPro + 1)],numExRho,MPI_DOUBLE,right,TAG_rho_b,  MPI_COMM_WORLD,&status);
MPI_Sendrecv(&phaseFieldLocal[numExRho], numExRho,MPI_DOUBLE, left,TAG_phaseField, &phaseFieldLocal[numExRho*(numOfRowPerPro + 1)],numExRho,MPI_DOUBLE,right,TAG_phaseField,  MPI_COMM_WORLD,&status);
if (tStep >= 100){
MPI_Sendrecv(&PsiSurL[numExRho], numExRho,MPI_DOUBLE, left,TAG_PsiSur, &PsiSurL[numExRho*(numOfRowPerPro + 1)],numExRho,MPI_DOUBLE,right,TAG_PsiSur,  MPI_COMM_WORLD,&status);
MPI_Sendrecv(&fluxZPositiveL[numExRho], numExRho,MPI_DOUBLE, left,TAG_fluxZPositive, &fluxZPositiveL[numExRho*(numOfRowPerPro + 1)],numExRho,MPI_DOUBLE,right,TAG_fluxZPositive,  MPI_COMM_WORLD,&status);
MPI_Sendrecv(&fluxZNegativeL[numExRho], numExRho,MPI_DOUBLE, left,TAG_fluxZNegative, &fluxZNegativeL[numExRho*(numOfRowPerPro + 1)],numExRho,MPI_DOUBLE,right,TAG_fluxZNegative,  MPI_COMM_WORLD,&status);
}
// data exchange between blocks//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// boundary condition //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//	periodic boundary
			if (ID_CPU_CORE == 1)
			{	int ID_left = (NumCPU_CORE - 1) - (NumCPU_CORE - ID_CPU_CORE) % (NumCPU_CORE - 1);

				assembleDisBC_EX(ID_CPU_CORE, f_rLocal, f_rBCSend, NX, NY, numOfRowPerPro, 1);
				assembleDisBC_EX(ID_CPU_CORE, f_bLocal, f_bBCSend, NX, NY, numOfRowPerPro, 1);				 
				MPI_Send(f_rBCSend, numExDis, MPI_DOUBLE, ID_left, TAG_f_r, MPI_COMM_WORLD);//  					
				MPI_Send(f_bBCSend, numExDis, MPI_DOUBLE, ID_left, TAG_f_b, MPI_COMM_WORLD);

				MPI_Send(&rho_rLocal[numExRho], numExRho, MPI_DOUBLE, ID_left, TAG_rho_r, MPI_COMM_WORLD);
				MPI_Send(&rho_bLocal[numExRho], numExRho, MPI_DOUBLE, ID_left, TAG_rho_b, MPI_COMM_WORLD);

				MPI_Recv(f_rBCReceive, numExDis, MPI_DOUBLE, ID_left, TAG_f_r, MPI_COMM_WORLD, &status);
				MPI_Recv(f_bBCReceive, numExDis, MPI_DOUBLE, ID_left, TAG_f_b, MPI_COMM_WORLD, &status);
				assemble_BC_Local_dis(ID_CPU_CORE, f_rBCReceive, f_rLocal, NX, NY, numOfRowPerPro, 0);
				assemble_BC_Local_dis(ID_CPU_CORE, f_bBCReceive, f_bLocal, NX, NY, numOfRowPerPro, 0);

				MPI_Recv(rho_rLocal, numExRho, MPI_DOUBLE, ID_left, TAG_rho_r, MPI_COMM_WORLD, &status);
				MPI_Recv(rho_bLocal, numExRho, MPI_DOUBLE, ID_left, TAG_rho_b, MPI_COMM_WORLD, &status);	

						MPI_Send(&phaseFieldLocal[numExRho], numExRho, MPI_DOUBLE, ID_left, TAG_phaseField, MPI_COMM_WORLD);	
						MPI_Recv(phaseFieldLocal, numExRho, MPI_DOUBLE, ID_left, TAG_phaseField, MPI_COMM_WORLD, &status);
					if (tStep >= 100)
					{	MPI_Send(&PsiSurL[numExRho], numExRho, MPI_DOUBLE, ID_left, TAG_PsiSur, MPI_COMM_WORLD);	
						MPI_Recv(PsiSurL, numExRho, MPI_DOUBLE, ID_left, TAG_PsiSur, MPI_COMM_WORLD, &status);	
						MPI_Send(&fluxZPositiveL[numExRho], numExRho, MPI_DOUBLE, ID_left, TAG_fluxZPositive, MPI_COMM_WORLD);	
						MPI_Recv(fluxZPositiveL, numExRho, MPI_DOUBLE, ID_left, TAG_fluxZPositive, MPI_COMM_WORLD, &status);
						MPI_Send(&fluxZNegativeL[numExRho], numExRho, MPI_DOUBLE, ID_left, TAG_fluxZNegative, MPI_COMM_WORLD);	
						MPI_Recv(fluxZNegativeL, numExRho, MPI_DOUBLE, ID_left, TAG_fluxZNegative, MPI_COMM_WORLD, &status);
					}
			}

			if (ID_CPU_CORE == NumCPU_CORE - 1)	{
				int ID_right = (ID_CPU_CORE) % (NumCPU_CORE - 1) + 1;
				MPI_Recv(f_rBCReceive, numExDis, MPI_DOUBLE, ID_right, TAG_f_r, MPI_COMM_WORLD, &status);
				MPI_Recv(f_bBCReceive, numExDis, MPI_DOUBLE, ID_right, TAG_f_b, MPI_COMM_WORLD, &status);				
				assemble_BC_Local_dis(ID_CPU_CORE, f_rBCReceive, f_rLocal, NX, NY, numOfRowPerPro, numOfRowPerPro + 1);
				assemble_BC_Local_dis(ID_CPU_CORE, f_bBCReceive, f_bLocal, NX, NY, numOfRowPerPro, numOfRowPerPro + 1);

				MPI_Recv(&rho_rLocal[numExRho*(numOfRowPerPro + 1)], numExRho, MPI_DOUBLE, ID_right, TAG_rho_r, MPI_COMM_WORLD, &status);
				MPI_Recv(&rho_bLocal[numExRho*(numOfRowPerPro + 1)], numExRho, MPI_DOUBLE, ID_right, TAG_rho_b, MPI_COMM_WORLD, &status);

				assembleDisBC_EX(ID_CPU_CORE, f_rLocal, f_rBCSend, NX, NY, numOfRowPerPro, numOfRowPerPro);
				assembleDisBC_EX(ID_CPU_CORE, f_bLocal, f_bBCSend, NX, NY, numOfRowPerPro, numOfRowPerPro);
				MPI_Send(f_rBCSend, numExDis, MPI_DOUBLE, ID_right, TAG_f_r, MPI_COMM_WORLD);
				MPI_Send(f_bBCSend, numExDis, MPI_DOUBLE, ID_right, TAG_f_b, MPI_COMM_WORLD);

				MPI_Send(&rho_rLocal[numExRho*numOfRowPerPro], numExRho, MPI_DOUBLE, ID_right, TAG_rho_r, MPI_COMM_WORLD);
				MPI_Send(&rho_bLocal[numExRho*numOfRowPerPro], numExRho, MPI_DOUBLE, ID_right, TAG_rho_b, MPI_COMM_WORLD);

						MPI_Recv(&phaseFieldLocal[numExRho*(numOfRowPerPro + 1)], numExRho, MPI_DOUBLE, ID_right, TAG_phaseField, MPI_COMM_WORLD, &status);
						MPI_Send(&phaseFieldLocal[numExRho*numOfRowPerPro], numExRho, MPI_DOUBLE, ID_right, TAG_phaseField, MPI_COMM_WORLD);				
					if (tStep >= 100){
						MPI_Recv(&PsiSurL[numExRho*(numOfRowPerPro + 1)], numExRho, MPI_DOUBLE, ID_right, TAG_PsiSur, MPI_COMM_WORLD, &status);
						MPI_Send(&PsiSurL[numExRho*numOfRowPerPro], numExRho, MPI_DOUBLE, ID_right, TAG_PsiSur, MPI_COMM_WORLD);
						MPI_Recv(&fluxZPositiveL[numExRho*(numOfRowPerPro + 1)], numExRho, MPI_DOUBLE, ID_right, TAG_fluxZPositive, MPI_COMM_WORLD, &status);
						MPI_Send(&fluxZPositiveL[numExRho*numOfRowPerPro], numExRho, MPI_DOUBLE, ID_right, TAG_fluxZPositive, MPI_COMM_WORLD);
						MPI_Recv(&fluxZNegativeL[numExRho*(numOfRowPerPro + 1)], numExRho, MPI_DOUBLE, ID_right, TAG_fluxZNegative, MPI_COMM_WORLD, &status);
						MPI_Send(&fluxZNegativeL[numExRho*numOfRowPerPro], numExRho, MPI_DOUBLE, ID_right, TAG_fluxZNegative, MPI_COMM_WORLD);

					}	
			}
// boundary condition ^^^^//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

cudaMemcpy(fluxZPositiveG,fluxZPositiveL, LRhoLocal*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(fluxZNegativeG,fluxZNegativeL, LRhoLocal*sizeof(double), cudaMemcpyHostToDevice);
getZFluxCap_weno<< < grid, block >> >(fluxZPositiveCapG, fluxZNegativeCapG, imageG,  fluxZPositiveG, fluxZNegativeG  ,NX, NY, numOfRowPerPro);
cudaMemcpy(fluxZPositiveCapL, fluxZPositiveCapG, LRhoLocal*sizeof(double), cudaMemcpyDeviceToHost);
cudaMemcpy(fluxZNegativeCapL, fluxZNegativeCapG, LRhoLocal*sizeof(double), cudaMemcpyDeviceToHost);

getDETofGradientBlockCPU(detL, imageLocal,  phaseFieldLocal, NX, NY, numOfRowPerPro);

 // data left-->rightVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
MPI_Sendrecv(&detL[numExRho*numOfRowPerPro],numExRho,MPI_DOUBLE,right,TAG_det, detL, numExRho,MPI_DOUBLE,left, TAG_det,  MPI_COMM_WORLD,&status);
MPI_Sendrecv(&fluxZPositiveCapL[numExRho*numOfRowPerPro],numExRho,MPI_DOUBLE,right,TAG_fluxZPositiveCap, fluxZPositiveCapL, numExRho,MPI_DOUBLE,left, TAG_fluxZPositiveCap,  MPI_COMM_WORLD,&status);
MPI_Sendrecv(&fluxZNegativeCapL[numExRho*numOfRowPerPro],numExRho,MPI_DOUBLE,right,TAG_fluxZNegativeCap, fluxZNegativeCapL, numExRho,MPI_DOUBLE,left, TAG_fluxZNegativeCap,  MPI_COMM_WORLD,&status);
			 
//data  left<--right 
MPI_Sendrecv(&detL[numExRho], numExRho,MPI_DOUBLE, left,TAG_det, &detL[numExRho*(numOfRowPerPro + 1)],numExRho,MPI_DOUBLE,right,TAG_det,  MPI_COMM_WORLD,&status);
MPI_Sendrecv(&fluxZPositiveCapL[numExRho], numExRho,MPI_DOUBLE, left,TAG_fluxZPositiveCap, &fluxZPositiveCapL[numExRho*(numOfRowPerPro + 1)],numExRho,MPI_DOUBLE,right,TAG_fluxZPositiveCap,  MPI_COMM_WORLD,&status);
MPI_Sendrecv(&fluxZNegativeCapL[numExRho], numExRho,MPI_DOUBLE, left,TAG_fluxZNegativeCap, &fluxZNegativeCapL[numExRho*(numOfRowPerPro + 1)],numExRho,MPI_DOUBLE,right,TAG_fluxZNegativeCap,  MPI_COMM_WORLD,&status);

	if (ID_CPU_CORE == 1)
	{	int ID_left = (NumCPU_CORE - 1) - (NumCPU_CORE - ID_CPU_CORE) % (NumCPU_CORE - 1); 
		MPI_Send(&detL[numExRho], numExRho, MPI_DOUBLE, ID_left, TAG_det, MPI_COMM_WORLD);	
		MPI_Recv(detL, numExRho, MPI_DOUBLE, ID_left, TAG_det, MPI_COMM_WORLD, &status);
						MPI_Send(&fluxZPositiveCapL[numExRho], numExRho, MPI_DOUBLE, ID_left, TAG_fluxZPositiveCap, MPI_COMM_WORLD);	
						MPI_Recv(fluxZPositiveCapL, numExRho, MPI_DOUBLE, ID_left, TAG_fluxZPositiveCap, MPI_COMM_WORLD, &status);
						MPI_Send(&fluxZNegativeCapL[numExRho], numExRho, MPI_DOUBLE, ID_left, TAG_fluxZNegativeCap, MPI_COMM_WORLD);	
						MPI_Recv(fluxZNegativeCapL, numExRho, MPI_DOUBLE, ID_left, TAG_fluxZNegativeCap, MPI_COMM_WORLD, &status);					 
	}

	if (ID_CPU_CORE == NumCPU_CORE - 1)	{
		int ID_right = (ID_CPU_CORE) % (NumCPU_CORE - 1) + 1;
		MPI_Recv(&detL[numExRho*(numOfRowPerPro + 1)], numExRho, MPI_DOUBLE, ID_right, TAG_det, MPI_COMM_WORLD, &status);
		MPI_Send(&detL[numExRho*numOfRowPerPro], numExRho, MPI_DOUBLE, ID_right, TAG_det, MPI_COMM_WORLD);
						MPI_Recv(&fluxZPositiveCapL[numExRho*(numOfRowPerPro + 1)], numExRho, MPI_DOUBLE, ID_right, TAG_fluxZPositiveCap, MPI_COMM_WORLD, &status);
						MPI_Send(&fluxZPositiveCapL[numExRho*numOfRowPerPro], numExRho, MPI_DOUBLE, ID_right, TAG_fluxZPositiveCap, MPI_COMM_WORLD);
						MPI_Recv(&fluxZNegativeCapL[numExRho*(numOfRowPerPro + 1)], numExRho, MPI_DOUBLE, ID_right, TAG_fluxZNegativeCap, MPI_COMM_WORLD, &status);
						MPI_Send(&fluxZNegativeCapL[numExRho*numOfRowPerPro], numExRho, MPI_DOUBLE, ID_right, TAG_fluxZNegativeCap, MPI_COMM_WORLD);							
	}
//extract data^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cudaMemcpy(fluxZPositiveCapG,fluxZPositiveCapL, LRhoLocal*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(fluxZNegativeCapG,fluxZNegativeCapL, LRhoLocal*sizeof(double), cudaMemcpyHostToDevice);
			    
uMax= getAbsMax(uxLocal,  NX, NY, numOfRowPerPro );
vMax= getAbsMax(uyLocal,  NX, NY, numOfRowPerPro );
wMax= getAbsMax(uzLocal,  NX, NY, numOfRowPerPro );

cudaMemcpy(f_rG, f_rLocal, LDisLocal *sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(f_bG, f_bLocal, LDisLocal *sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(rho_rG, rho_rLocal, LRhoLocal*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(rho_bG, rho_bLocal, LRhoLocal*sizeof(double), cudaMemcpyHostToDevice);

cudaMemcpy(PsiSurG, PsiSurL, LRhoLocal*sizeof(double), cudaMemcpyHostToDevice);
swap(detG,detPreG);
cudaMemcpy(detG, detL, LRhoLocal*sizeof(double), cudaMemcpyHostToDevice);

 


if (tStep >= 100){cudaMemcpy(PsiSurG, PsiSurL, LRhoLocal * sizeof(double), cudaMemcpyHostToDevice);}
		
                stream3D_Block << < grid, block >> >(f_rG, f_rhG, imageG, NX, NY, numOfRowPerPro);   //new
				cudaDeviceSynchronize();
				swap(&f_rG, &f_rhG); 
				cudaDeviceSynchronize();	 
                stream3D_Block << < grid, block>> >(f_bG, f_bhG, imageG, NX, NY, numOfRowPerPro);   //new
				cudaDeviceSynchronize();
				swap(&f_bG, &f_bhG);
 		        cudaDeviceSynchronize();
		 		getMacro3DRK_Block<< < grid, block>> >(imageG, f_rG, f_bG, uxG, uyG, uzG, ux_oldG, uy_oldG, uz_oldG, rho_rG, rho_bG, rhoG, rho_ri, rho_bi, NX, NY, numOfRowPerPro);
				//CollisionRK3D_Block<< < grid, block>> >(imageG, uxG, uyG, uzG, rhoG, rho_rG, rho_bG, ffG, f_rG, f_bG, taur, taub, t_kG, exG, eyG, ezG,
				//alfa_r, alfa_b, c_squ, c_squ_r, c_squ_b, f_bodyG, NX, NY, numOfRowPerPro);//  image==0 
                                
CollisionRK3D_Blocktest<< < grid, block>> >(imageG, uxG, uyG, uzG, rhoG, rho_rG, rho_bG, ffG, f_rG, f_bG, taur, taub, t_kG, exG, eyG, ezG,
				alfa_r, alfa_b, c_squ, c_squ_r, c_squ_b, f_bodyG, NX, NY, numOfRowPerPro,1.0);//  image==0 

	            redistributeRK3D_Block<< < grid, block>> >(imageG, rhoG, rho_rG, rho_bG, exG, eyG, ezG, ffG, f_rG, f_bG, t_kG, A, beta, rsqG, BiG, FxG, FyG, FzG,NX, NY, numOfRowPerPro); //  image==0
                cudaDeviceSynchronize();
				boundaryWall3D_Block<< < grid, block>> >(f_rG, imageG, NX, NY, numOfRowPerPro);    //new
				boundaryWall3D_Block<< < grid, block>> >(f_bG, imageG, NX, NY, numOfRowPerPro);    //new
				cudaDeviceSynchronize();

if (tStep >= 100)
{ 
	if (tStep == 100){
	/*1*///getStermBlock_temp<< < grid, block>> >(sTermPreG, imageG, uxG,uyG,uzG,  PsiSurG,  detG, 0.0, deltaT/deltaX,uMax,  vMax,  wMax, deltaX,NX,  NY,numOfRowPerPro); 
	// getStermBlock_temp_Phase<< < grid, block>> >(sTermPreG, imageG, uxG,uyG,uzG,  PsiSurG,  detPreG, 0.0, deltaT/deltaX,uMax,  vMax,  wMax, deltaX,NX,  NY,numOfRowPerPro);

	
	getStermBlock_temp_Phase_weno<< < grid, block>> >(sTermPreG,  imageG, uxG,uyG, PsiSurG,   detPreG, fluxZPositiveCapG, fluxZNegativeCapG, 
	fluxXPositiveG,fluxXNegativeG,fluxYPositiveG, fluxYNegativeG,fluxXPositiveCapG,fluxXNegativeCapG,fluxYPositiveCapG, fluxYNegativeCapG,
	0.0, deltaT/deltaX,uMax,  vMax,deltaX,NX,  NY,numOfRowPerPro);

	
	}
	else 
	{ swap(sTermPreG,sTermG); }

/*1*/	//getStermBlock_temp<< < grid, block>> >(sTermG, imageG, uxG,uyG,uzG,  PsiSurG,  detG, 0.0, deltaT/deltaX,uMax,  vMax,  wMax, deltaX,NX,  NY,numOfRowPerPro);
//getStermBlock_temp_Phase<< < grid, block>> >(sTermG, imageG, uxG,uyG,uzG,  PsiSurG,  detPreG, 0.0, deltaT/deltaX,uMax,  vMax,  wMax, deltaX,NX,  NY,numOfRowPerPro);

getStermBlock_temp_Phase_weno<< < grid, block>> >(sTermG,  imageG, uxG,uyG, PsiSurG,   detPreG, fluxZPositiveCapG, fluxZNegativeCapG, 
fluxXPositiveG,fluxXNegativeG,fluxYPositiveG, fluxYNegativeG,fluxXPositiveCapG,fluxXNegativeCapG,fluxYPositiveCapG, fluxYNegativeCapG,
0.0, deltaT/deltaX,uMax,  vMax,deltaX,NX,  NY,numOfRowPerPro);




	//assemble B matrx
	//_SORG_AssembleB<< < grid, block>> >(imageG,deltaT/deltaX/deltaX,deltaT/deltaX, bMatrixG, PsiSurG,uxG,uyG,uzG,  uMax,  vMax,  wMax, detG, NX,  NY,numOfRowPerPro);

/*1*/	//_SORG_AssembleB_CN<< < grid, block>> >(imageG,0.0*deltaT/deltaX/deltaX,deltaT/deltaX, bMatrixG, PsiSurG,uxG,uyG,uzG,  uMax,  vMax,  wMax, detG,sTermG,sTermPreG, NX,  NY,numOfRowPerPro);
_SORG_AssembleB_CN_Phase<< < grid, block>> >(imageG,0.0*deltaT/deltaX/deltaX,deltaT/deltaX, bMatrixG, PsiSurG,uxG,uyG,uzG,  uMax,  vMax,  wMax, detPreG,sTermG,sTermPreG, NX,  NY,numOfRowPerPro);
 

	//_SORAG << < grid, block>> >( imageG,  0.0*deltaT/deltaX/deltaX, bMatrixG,PsiLastIterG, PsiSurNewG,    0.9,NX, NY, numOfRowPerPro    );

/*1*/ //	_SORAG_CN << < grid, block>> >( imageG,  0.0*deltaT/deltaX/deltaX, bMatrixG,PsiLastIterG, PsiSurNewG,    0.9,NX, NY, numOfRowPerPro    );
_SORAG_CN_Phase << < grid, block>> >( imageG,  0.0*deltaT/deltaX/deltaX, bMatrixG,PsiLastIterG, PsiSurNewG, PsiSurG,detG, detPreG , 0.9,NX, NY, numOfRowPerPro    );

 

	swap(PsiSurG,PsiSurNewG);

 //__global__ void _SORAG_CN_Phase  ( int *imageG, double lamda, double *B,double *X_lastIter, double *x_new, double*x_old, double *detG,  double omiga,int NX,int NY,int NZ    )
}//if (tStep >= 100)

			if (tStep % 500== 0)
			{

				//MPI_Send(&rhoLocal[numExRho], numExRho*numOfRowPerPro, MPI_DOUBLE, 0, TAG_rho, MPI_COMM_WORLD);
				MPI_Send(&rho_rLocal[numExRho], numExRho*numOfRowPerPro, MPI_DOUBLE, 0, TAG_rho_r, MPI_COMM_WORLD);
				MPI_Send(&rho_bLocal[numExRho], numExRho*numOfRowPerPro, MPI_DOUBLE, 0, TAG_rho_b, MPI_COMM_WORLD);
				MPI_Send(&phaseFieldLocal[numExRho], numExRho*numOfRowPerPro, MPI_DOUBLE, 0, TAG_phaseField, MPI_COMM_WORLD);
				//MPI_Send(&uxLocal[numExRho], numExRho*numOfRowPerPro, MPI_DOUBLE, 0, TAG_ux, MPI_COMM_WORLD);
				//MPI_Send(&uyLocal[numExRho], numExRho*numOfRowPerPro, MPI_DOUBLE, 0, TAG_uy, MPI_COMM_WORLD);
				//MPI_Send(&uzLocal[numExRho], numExRho*numOfRowPerPro, MPI_DOUBLE, 0, TAG_uz, MPI_COMM_WORLD); 
MPI_Send(&PsiSurL[numExRho], numExRho*numOfRowPerPro, MPI_DOUBLE, 0, TAG_PsiSur, MPI_COMM_WORLD);	
MPI_Send(&detL[numExRho], numExRho*numOfRowPerPro, MPI_DOUBLE, 0, TAG_det, MPI_COMM_WORLD);			 
			}

		}//if (ID_CPU_CORE > 0)
		if (ID_CPU_CORE == 0)
		{
			if (tStep %500== 0)
			{
				for (int i = 1; i < NumCPU_CORE; i++)
				{
					int j_startR = numExRho*((i - 1)*numOfRowPerPro + 1);
					int DimBlock  = numExRho*numOfRowPerPro;
	 				
					MPI_Recv(&rho_r[j_startR], DimBlock, MPI_DOUBLE, i, TAG_rho_r, MPI_COMM_WORLD, &status);
					MPI_Recv(&rho_b[j_startR], DimBlock, MPI_DOUBLE, i, TAG_rho_b, MPI_COMM_WORLD, &status);
					//MPI_Recv(&rho[j_startR], DimBlock, MPI_DOUBLE, i, TAG_rho, MPI_COMM_WORLD, &status);
					MPI_Recv(&phaseField[j_startR], DimBlock, MPI_DOUBLE, i, TAG_phaseField, MPI_COMM_WORLD, &status);
					//MPI_Recv(&ux[j_startR], DimBlock, MPI_DOUBLE, i, TAG_ux, MPI_COMM_WORLD, &status);
					//MPI_Recv(&uy[j_startR], DimBlock, MPI_DOUBLE, i, TAG_uy, MPI_COMM_WORLD, &status);
					//MPI_Recv(&uz[j_startR], DimBlock, MPI_DOUBLE, i, TAG_uz, MPI_COMM_WORLD, &status);
                   MPI_Recv(&PsiSur[j_startR], DimBlock, MPI_DOUBLE, i,TAG_PsiSur, MPI_COMM_WORLD, &status);
MPI_Recv(&det[j_startR], DimBlock, MPI_DOUBLE, i, TAG_det, MPI_COMM_WORLD, &status);
				}
 
              //writeData_ArraybyImage_vtk(tStep, image, phaseField, NX, NY, NZ,   (char *)"./testRK/phaseField_test", (char *) " asdf", 1.0);
             writeData_ArraybyImage_vtk(tStep, image, phaseField, NX, NY, NZ,   (char*)phaseFieldName.c_str(), (char *) " asdf", 1.0);
writeData_ArraybyImage_vtk(tStep, image, PsiSur, NX, NY, NZ,   (char*)PsiName.c_str(), (char *) " asdf", 1.0);
writeData_ArraybyImage_vtk(tStep, image,det, NX, NY, NZ,   (char *)"./temp/det", (char *) " asdf", 1.0);
			}	
		} //if (ID_CPU_CORE == 0)
}  //______for (tStep = 1; tStep <= tMax; tStep++)
    MPI_Finalize();
    return 0;
}



void initialBlock_3D_RK(int *image, double *ux, double *uy, double *uz, double *rho_b, double *rho_r, double *f_r, double *f_b, double rho_bi, double rho_ri,
	double *t_k, double *ex, double *ey, double *ez, double alfa_r, double alfa_b, int NX, int NY, int NZ)
{
	int jx, jy, jz,   i, k, j_1;
	double   u_squ;
	double u_n[19];
	double fequ_r[19], fequ_b[19];


	int LineNum = (NY + 2)*(NX + 2)*(NZ + 2);

	for (jz = 1; jz <= NZ; jz++)
	for (jy = 1; jy <= NY; jy++)
	for (jx = 1; jx <= NX; jx++)
	{
		j_1 = jz*(NY + 2)*(NX + 2) + jy*(NX + 2) + jx;

		rho_r[j_1] = 0;
		rho_b[j_1] = rho_bi;

		if (pow(jx - NX/2, 2) + pow(jy - 30, 2) + pow((jz - NY/2), 2) <= pow(10, 2))	{
			rho_r[j_1] = rho_ri;
			rho_b[j_1] = 0;
		}
/*
 		if (pow(jx - NX /2, 2) + pow(jy - NY /4, 2) + pow((jz - 40), 2) <= pow(7, 2))	{
			rho_r[j_1] = rho_ri;
			rho_b[j_1] = 0;
		}
		if (pow(jx - 32, 2) + pow(jy - 42, 2) + pow((jz - 34), 2) <= pow(10, 2))	{
			rho_r[j_1] = rho_ri;
			rho_b[j_1] = 0;
		}
 */
		if (image[j_1] == 5)	{
			rho_r[j_1] = rho_ri*0.3;
			rho_b[j_1] = rho_bi*1.0;			
		}
	}


	for (jz = 1; jz <= NZ; jz++)
	for (jy = 1; jy <= NY; jy++)
	for (jx = 1; jx <= NX; jx++)
	{

		j_1 = jz*(NY + 2)*(NX + 2) + jy*(NX + 2) + jx;

		ux[j_1] = 0.0;
		uy[j_1] = 0.0;
		uz[j_1] = 0.0;

		u_squ = ux[j_1] * ux[j_1] + uy[j_1] * uy[j_1] + uz[j_1] * uz[j_1];
		u_n[0] = 0.0;
		for (k = 1; k < 19; k++)
		{
			u_n[k] = ex[k] * ux[j_1] + ey[k] * uy[j_1] + ez[k] * uz[j_1];
		}

		fequ_r[0] = t_k[0] * rho_r[j_1] * (u_n[0] * 3.0 + u_n[0] * u_n[0] * 4.5 - u_squ*1.5) + rho_r[j_1] * alfa_r;
		for (k = 1; k < 7; k++)
		{
			fequ_r[k] = t_k[k] * rho_r[j_1] * (u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ*1.5) + rho_r[j_1] * (1.0 - alfa_r) / 12.0;
		}
		for (k = 7; k < 19; k++)
		{
			fequ_r[k] = t_k[k] * rho_r[j_1] * (u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ*1.5) + rho_r[j_1] * (1.0 - alfa_r) / 24.0;
		}


		fequ_b[0] = t_k[0] * rho_b[j_1] * (u_n[0] * 3.0 + u_n[0] * u_n[0] * 4.5 - u_squ*1.5) + rho_b[j_1] * alfa_b;
		for (k = 1; k < 7; k++)
		{
			fequ_b[k] = t_k[k] * rho_b[j_1] * (u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ*1.5) + rho_b[j_1] * (1.0 - alfa_b) / 12.0;
		}
		for (k = 7; k < 19; k++)
		{
			fequ_b[k] = t_k[k] * rho_b[j_1] * (u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ*1.5) + rho_b[j_1] * (1.0 - alfa_b) / 24.0;
		}


		for (i = 0; i < 19; i++)
		{
			 
			if (image[j_1] != 5)
			{
				f_r[j_1 + i * LineNum] = fequ_r[i];
				f_b[j_1 + i * LineNum] = fequ_b[i];
			}
			else
			{
				f_r[j_1 + i * LineNum] = 0;
				f_b[j_1 + i * LineNum] = 0;
			}
		}

	}
}


void setImageSimpleBlock(int *imageOver)
{	
	int jx, jy, jz, j1 ;
	for (jz = 1; jz <= NZ; jz = jz + 1)
	for (jy = 1; jy <= NY; jy = jy + 1)
	for (jx = 1; jx <= NX; jx = jx + 1)
	{		
		j1 = (NX + 2)*(NY + 2)*jz + jy*(NX + 2) + jx;		
		imageOver[j1] = 0;		
		
		if (jx == 1 || jx == NX || jy == 1 || jy == NY)	{
           imageOver[j1] = 5;
		}		
	}	
}

 __global__ void _SORAG  ( int *imageG, double lamda, double *B,double *X_lastIter, double *x_new,    double omiga,int NX,int NY,int NZ    )
{ 
	int  jx, jy, jz;
	int ip,  ipL, ipR, ipU, ipD, ipF, ipB;//LR x; U  D  y; F B z; 

 	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;
	jx = blockDim.x*blockIdx.x + threadIdx.x + 1; 

		for(jz=0;jz<=NZ+1; jz++){		 
			ip=jz*(NX + 2)*(NY + 2) + jy*(NX + 2) +jx;;
			X_lastIter[ip]=B[ip]; 
            x_new[ip]=B[ip]; 
	    } 
            
	  for (int j = 1; j <=20; j++)
	  {         
           double *tmp = X_lastIter;   X_lastIter = x_new;   x_new = tmp;   //  swapDevice(X_lastIter,x_new);            
           for (jz = 1; jz <= NZ; jz++) {
			ip = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
			if (imageG[ip]==0) {
			ipL = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx - 1;
			ipR = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx + 1;
			ipU = jz*(NX + 2)*(NY + 2) + (jy + 1)*(NX + 2) + jx;
			ipD = jz*(NX + 2)*(NY + 2) + (jy - 1)*(NX + 2) + jx;
			ipF = (jz + 1)*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
			ipB = (jz - 1)*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;          
            x_new[ip] = (B[ip] + lamda*(X_lastIter[ipL]+X_lastIter[ipD]+X_lastIter[ipB] + X_lastIter[ipR]+X_lastIter[ipU]+X_lastIter[ipF] ))/(1+6*lamda);
            }
         }
 			 for (jz = 1; jz <= NZ; jz++) {
				ip = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) +jx;
				if (imageG[ip]==0) {x_new[ip] = (1.0 - omiga) * X_lastIter[ip] + omiga * x_new[ip];		 }
			  }                        
	  }//for iteration step
}
 __global__ void _SORAG_CN  ( int *imageG, double lamda, double *B,double *X_lastIter, double *x_new,    double omiga,int NX,int NY,int NZ    )
{ 
	int  jx, jy, jz;
	int ip,  ipL, ipR, ipU, ipD, ipF, ipB;//LR x; U  D  y; F B z; 

 	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;
	jx = blockDim.x*blockIdx.x + threadIdx.x + 1; 

		for(jz=0;jz<=NZ+1; jz++){		 
			ip=jz*(NX + 2)*(NY + 2) + jy*(NX + 2) +jx;;
			X_lastIter[ip]=B[ip]; 
            x_new[ip]=B[ip]; 
	    } 
            
	  for (int j = 1; j <=20; j++)
	  {         
           double *tmp = X_lastIter;   X_lastIter = x_new;   x_new = tmp;   //  swapDevice(X_lastIter,x_new);            
           for (jz = 1; jz <= NZ; jz++) {
			ip = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
			if (imageG[ip]==0) {
			ipL = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx - 1;
			ipR = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx + 1;
			ipU = jz*(NX + 2)*(NY + 2) + (jy + 1)*(NX + 2) + jx;
			ipD = jz*(NX + 2)*(NY + 2) + (jy - 1)*(NX + 2) + jx;
			ipF = (jz + 1)*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
			ipB = (jz - 1)*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;          
            x_new[ip] = (B[ip] + 0.5*lamda*(X_lastIter[ipL]+X_lastIter[ipD]+X_lastIter[ipB] + X_lastIter[ipR]+X_lastIter[ipU]+X_lastIter[ipF] ))/(1+3.0*lamda);
            }
         }
 			 for (jz = 1; jz <= NZ; jz++) {
				ip = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) +jx;
				if (imageG[ip]==0) {x_new[ip] = (1.0 - omiga) * X_lastIter[ip] + omiga * x_new[ip];		 }
			  }                        
	  }//for iteration step
}
 __global__ void _SORAG_CN_Phase  ( int *imageG, double lamda, double *B,double *X_lastIter, double *x_new, double*x_old, double *detG,double *detPreG,  double omiga,int NX,int NY,int NZ    )
 
{ 
	int  jx, jy, jz;
	int ip,  ipL, ipR, ipU, ipD, ipF, ipB;//LR x; U  D  y; F B z; 

 	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;
	jx = blockDim.x*blockIdx.x + threadIdx.x + 1; 

	for(jz=0;jz<=NZ+1; jz++){		 
		ip=jz*(NX + 2)*(NY + 2) + jy*(NX + 2) +jx;
		X_lastIter[ip]=B[ip]; 
		x_new[ip]=B[ip]; 
	} 
            
	for (int j = 1; j <=20; j++)
	{         
		double *tmp = X_lastIter;   X_lastIter = x_new;   x_new = tmp;   //  swapDevice(X_lastIter,x_new);            
		for (jz = 1; jz <= NZ; jz++) {
			ip = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;

			if (imageG[ip]==0&&detG[ip]>=1e-6) {
				ipL = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx - 1;
				ipR = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx + 1;
				ipU = jz*(NX + 2)*(NY + 2) + (jy + 1)*(NX + 2) + jx;
				ipD = jz*(NX + 2)*(NY + 2) + (jy - 1)*(NX + 2) + jx;
				ipF = (jz + 1)*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
				ipB = (jz - 1)*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;          
			//	x_new[ip] = (B[ip] + 0.5*lamda*(X_lastIter[ipL]+X_lastIter[ipD]+X_lastIter[ipB] + X_lastIter[ipR]+X_lastIter[ipU]+X_lastIter[ipF] ))/(1+3.0*lamda);
				x_new[ip] = (B[ip] + 0.5*lamda*detPreG[ip]*(X_lastIter[ipL]+X_lastIter[ipD]+X_lastIter[ipB] + X_lastIter[ipR]+X_lastIter[ipU]+X_lastIter[ipF] ))/(detG[ip]+3.0*lamda*detPreG[ip]);

				x_new[ip] = (1.0 - omiga) * X_lastIter[ip] + omiga * x_new[ip];
			}
			else if (imageG[ip]==0&&detG[ip]<1e-6) {
				x_new[ip] = x_old[ip];
			}
		}
                
	}//for iteration step
}

__global__ void _SORG_AssembleB(int *imageG,double Lambda,double alpha,double *B,double *Psi,double *ux,double *uy,double *uz,double uMax,double vMax, double wMax,double *detG,int NX,int NY,int NZ)

{
	int  jx, jy, jz;
	int ip,  ipE, ipW, ipN, ipS, ipT, ipB;  //LR x; U  D  y; F B z; 

 	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;
	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
			 
	for (jz = 1; jz <= NZ; jz++)
        {
            ip = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
             if (imageG[ip]==0)
             {
				ipW = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx - 1;
				ipE = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx + 1;
				ipN = jz*(NX + 2)*(NY + 2) + (jy + 1)*(NX + 2) + jx;
				ipS = jz*(NX + 2)*(NY + 2) + (jy - 1)*(NX + 2) + jx;
				ipT = (jz + 1)*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
				ipB = (jz - 1)*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;

     B[ip] =Psi[ip]
            -0.5*alpha*(  (ux[ip]+uMax)*(Psi[ip]-Psi[ipW]) + (uy[ip]+vMax)*(Psi[ip]-Psi[ipS]) + (uz[ip]+wMax)*(Psi[ip]-Psi[ipB])  ) 
            -0.5*alpha*(  (ux[ip]-uMax)*(Psi[ipE]-Psi[ip]) + (uy[ip]-vMax)*(Psi[ipN]-Psi[ip]) + (uz[ip]-wMax)*(Psi[ipT]-Psi[ip])  );     //B column 

	/* 	    B[ip] =Psi[ip]
            - alpha*(  uz[ip] *(Psi[ip]-Psi[ipB])  ) 
             ;     //B column 
*/
/*
 	    B[ip] =Psi[ip]-0.5*LambdaA*6*Psi[ip] + 0.5*LambdaA*(Psi[ipE]+Psi[ipW]+Psi[ipN]+Psi[ipS]+Psi[ipT]+Psi[ipB])
                             -0.5*lambdaB*ux[ip]*(Psi[ipE]-Psi[ipW]) 
                             -0.5*lambdaB*uy[ip]*(Psi[ipN]-Psi[ipS]) 
                             -0.5*lambdaB*uz[ip]*(Psi[ipT]-Psi[ipB])     //B column 
  */                             
                      ;  			
/*		   if (detG[ip]<0.0000001)   
		   {
		      B[ip] =Psi[ip]-0.5*LambdaA*6*Psi[ip] + 0.5*LambdaA*(Psi[ipE]+Psi[ipW]+Psi[ipN]+Psi[ipS]+Psi[ipT]+Psi[ipB])
		                    -0.5*lambdaB*ux[ip]*(Psi[ipE]-Psi[ipW]) 
		                     -0.5*lambdaB*uy[ip]*(Psi[ipN]-Psi[ipS]) 
		                     -0.5*lambdaB*uz[ip]*(Psi[ipT]-Psi[ipB])
		              ;                      
		   }
		 else
		   {
		     B[ip] =detG[ip]*Psi[ip]-detG[ip]*0.5*LambdaA*6*Psi[ip] + detG[ip]*0.5*LambdaA*(Psi[ipE]+Psi[ipW]+Psi[ipN]+Psi[ipS]+Psi[ipT]+Psi[ipB])
		                     -0.5*lambdaB*ux[ip]*(Psi[ipE]*detG[ipE]-Psi[ipW]*detG[ipW]) 
		                     -0.5*lambdaB*uy[ip]*(Psi[ipN]*detG[ipN]-Psi[ipS]*detG[ipS]) 
		                     -0.5*lambdaB*uz[ip]*(Psi[ipT]*detG[ipT]-Psi[ipB]*detG[ipB])     //B column 
		              ;                       
		   }  
           
              */   
             }
         } 
}
__global__ void _SORG_AssembleB_CN(int *imageG,double Lambda,double alpha,double *B,double *Psi,double *ux,double *uy,double *uz,double uMax,double vMax, double wMax,double *detG,double *sTerm,double *sTermPre,int NX,int NY,int NZ)

{
	int  jx, jy, jz;
	int ip,  ipE, ipW, ipN, ipS, ipT, ipB;  //LR x; U  D  y; F B z; 

 	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;
	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
			 
	for (jz = 1; jz <= NZ; jz++)
	{
        ip = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
        if (imageG[ip]==0)
        {
		ipW = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx - 1;
		ipE = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx + 1;
		ipN = jz*(NX + 2)*(NY + 2) + (jy + 1)*(NX + 2) + jx;
		ipS = jz*(NX + 2)*(NY + 2) + (jy - 1)*(NX + 2) + jx;
		ipT = (jz + 1)*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
		ipB = (jz - 1)*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
/*
		B[ip] =(1.0-3.0*Lambda)*Psi[ip]+0.5*Lambda*(Psi[ipE]+Psi[ipW]+Psi[ipN]+Psi[ipS]+Psi[ipT]+Psi[ipB])
            -0.5*alpha*(  (ux[ip]+uMax)*(Psi[ip]-Psi[ipW]) + (uy[ip]+vMax)*(Psi[ip]-Psi[ipS]) + (uz[ip]+wMax)*(Psi[ip]-Psi[ipB])  ) 
            -0.5*alpha*(  (ux[ip]-uMax)*(Psi[ipE]-Psi[ip]) + (uy[ip]-vMax)*(Psi[ipN]-Psi[ip]) + (uz[ip]-wMax)*(Psi[ipT]-Psi[ip])  );     //B column 

*/		B[ip] =(1.0-3.0*Lambda)*Psi[ip]+0.5*Lambda*(Psi[ipE]+Psi[ipW]+Psi[ipN]+Psi[ipS]+Psi[ipT]+Psi[ipB])
            +1.5*sTerm[ip] -0.5*sTermPre[ip] ;     //B column 
		}
	} 
}
__global__ void _SORG_AssembleB_CN_Phase(int *imageG,double Lambda,double alpha,double *B,double *Psi,double *ux,double *uy,double *uz,double uMax,double vMax, double wMax,double *detPreG,double *sTerm,double *sTermPre,int NX,int NY,int NZ)

{
	int  jx, jy, jz;
	int ip,  ipE, ipW, ipN, ipS, ipT, ipB;  //LR x; U  D  y; F B z; 

 	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;
	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
			 
	for (jz = 1; jz <= NZ; jz++)
	{
        ip = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
        if (imageG[ip]==0)
        {
		ipW = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx - 1;
		ipE = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx + 1;
		ipN = jz*(NX + 2)*(NY + 2) + (jy + 1)*(NX + 2) + jx;
		ipS = jz*(NX + 2)*(NY + 2) + (jy - 1)*(NX + 2) + jx;
		ipT = (jz + 1)*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
		ipB = (jz - 1)*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
/*
		B[ip] =(1.0-3.0*Lambda)*Psi[ip]+0.5*Lambda*(Psi[ipE]+Psi[ipW]+Psi[ipN]+Psi[ipS]+Psi[ipT]+Psi[ipB])
            -0.5*alpha*(  (ux[ip]+uMax)*(Psi[ip]-Psi[ipW]) + (uy[ip]+vMax)*(Psi[ip]-Psi[ipS]) + (uz[ip]+wMax)*(Psi[ip]-Psi[ipB])  ) 
            -0.5*alpha*(  (ux[ip]-uMax)*(Psi[ipE]-Psi[ip]) + (uy[ip]-vMax)*(Psi[ipN]-Psi[ip]) + (uz[ip]-wMax)*(Psi[ipT]-Psi[ip])  );     //B column 

*/		B[ip] =detPreG[ip]*(1.0-3.0*Lambda)*Psi[ip]+0.5*Lambda*detPreG[ip]*(Psi[ipE]+Psi[ipW]+Psi[ipN]+Psi[ipS]+Psi[ipT]+Psi[ipB])
            +1.5*sTerm[ip] -0.5*sTermPre[ip] ;     //B column 
		}
	} 
}
double getAbsMax(double *xMatrix,  int NX,int NY,int NZblock )
{
	int jx, jy, jz,  ip; 
    double xMax=0;
	for (jz = 1; jz <= NZblock; jz++)
	for (jy = 1; jy <= NY; jy++)
	for (jx = 1; jx <= NX; jx++)
	{		 
		ip = jz*(NY + 2)*(NX + 2) + jy*(NX + 2) + jx;
        if(xMax < abs(xMatrix[ip])) 
          {xMax = abs(xMatrix[ip]);}      	
	}	
return xMax;
}


__global__ void getStermBlock_temp(double *Sterm, int *image, double *ux, double *uy, double *uz, double *Psi,  double *Determinant, 
double Ds,double alpha,double uMax,double vMax,double wMax, double dx,int NX, int NY, int NZblock)//no weno
//  no weno, pending usage
{
	int jx, jy, jz, ip;
//	double Rx, Ry, Rz;
	int y_n, x_e, y_s, x_w, z_t, z_b;
	//double Px, Py, Pz;
	//double Dx, Dy, Dz;
	//double PDxPositive, PDyPositive, PDzPositive;
	//double PDxNegative, PDyNegative, PDzNegative;
	int ipE, ipW, ipN, ipS, ipT, ipB;

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZblock; jz++) {
ip = jz * (NY + 2)*(NX + 2) + jy * (NX + 2) + jx;
		if (image[ip] == 0) {
			
			x_e = jx % NX + 1;
			x_w = NX - (NX + 1 - jx) % NX;
			y_n = jy % NY + 1;
			y_s = NY - (NY + 1 - jy) % NY;
			z_t = jz + 1;
			z_b = jz - 1;

			//center 6
			ipE = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_e;
			ipW = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_w;
			ipN = jz * (NX + 2)*(NY + 2) + y_n * (NX + 2) + jx;
			ipS = jz * (NX + 2)*(NY + 2) + y_s * (NX + 2) + jx;
			ipT = z_t * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
			ipB = z_b * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
/*
			PDxPositive = (Psi[ip] * Determinant[ip] - Psi[ipW] * Determinant[ipW]) / dx;
			PDyPositive = (Psi[ip] * Determinant[ip] - Psi[ipS] * Determinant[ipS]) / dx;
		    PDzPositive = (Psi[ip] * Determinant[ip] - Psi[ipB] * Determinant[ipB]) / dx;

			PDxNegative = (Psi[ipE] * Determinant[ipE] - Psi[ip] * Determinant[ip]) / dx;
			PDyNegative = (Psi[ipN] * Determinant[ipN] - Psi[ip] * Determinant[ip]) / dx;
			PDzNegative = (Psi[ipT] * Determinant[ipT] - Psi[ip] * Determinant[ip]) / dx;

			Dx = 0.5*(Determinant[ipE] - Determinant[ipW]) / dx;
			Dy = 0.5*(Determinant[ipN] - Determinant[ipS]) / dx;
			Dz = 0.5*(Determinant[ipT] - Determinant[ipB]) / dx;

			Px = 0.5*(Psi[ipE] - Psi[ipW]) / dx;
			Py = 0.5*(Psi[ipN] - Psi[ipS]) / dx;
			Pz = 0.5*(Psi[ipT] - Psi[ipB]) / dx;
*/
		/*	Sterm[j1] = -0.5 * (  (u[j1] + uMax)* PDxPositive+ ( v[j1] + vMax) * PDyPositive + (w[j1] + wMax) * PDzPositive
				                 + (u[j1] - uMax)* PDxNegative + (v[j1] - vMax) * PDyNegative + (w[j1] - wMax) * PDzNegative )
				         + Ds * (Dx * Px+ Dy * Py+ Dz * Pz);
*/


			Sterm[ip] = -0.5*alpha*(  (ux[ip]+uMax)*(Psi[ip]-Psi[ipW]) + (uy[ip]+vMax)*(Psi[ip]-Psi[ipS]) + (uz[ip]+wMax)*(Psi[ip]-Psi[ipB])  ) 
            -0.5*alpha*(  (ux[ip]-uMax)*(Psi[ipE]-Psi[ip]) + (uy[ip]-vMax)*(Psi[ipN]-Psi[ip]) + (uz[ip]-wMax)*(Psi[ipT]-Psi[ip])  );
		}
	}
}
__global__ void getStermBlock_temp_Phase(double *Sterm, int *image, double *ux, double *uy, double *uz, double *Psi,  double *detPreG, 
double Ds,double alpha,double uMax,double vMax,double wMax, double dx,int NX, int NY, int NZblock)//no weno
//  no weno, pending usage
{
	int jx, jy, jz, ip;
 
	int y_n, x_e, y_s, x_w, z_t, z_b;
	double Px, Py, Pz;
	double Dx, Dy, Dz;
 
	int ipE, ipW, ipN, ipS, ipT, ipB;

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZblock; jz++) {
	ip = jz * (NY + 2)*(NX + 2) + jy * (NX + 2) + jx;
		if (image[ip] == 0) {
			
			x_e = jx % NX + 1;
			x_w = NX - (NX + 1 - jx) % NX;
			y_n = jy % NY + 1;
			y_s = NY - (NY + 1 - jy) % NY;
			z_t = jz + 1;
			z_b = jz - 1;

			//center 6
			ipE = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_e;
			ipW = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_w;
			ipN = jz * (NX + 2)*(NY + 2) + y_n * (NX + 2) + jx;
			ipS = jz * (NX + 2)*(NY + 2) + y_s * (NX + 2) + jx;
			ipT = z_t * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
			ipB = z_b * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
/*
			PDxPositive = (Psi[ip] * Determinant[ip] - Psi[ipW] * Determinant[ipW]) / dx;
			PDyPositive = (Psi[ip] * Determinant[ip] - Psi[ipS] * Determinant[ipS]) / dx;
		    PDzPositive = (Psi[ip] * Determinant[ip] - Psi[ipB] * Determinant[ipB]) / dx;

			PDxNegative = (Psi[ipE] * Determinant[ipE] - Psi[ip] * Determinant[ip]) / dx;
			PDyNegative = (Psi[ipN] * Determinant[ipN] - Psi[ip] * Determinant[ip]) / dx;
			PDzNegative = (Psi[ipT] * Determinant[ipT] - Psi[ip] * Determinant[ip]) / dx;
*/
			Dx = 0.5*(detPreG[ipE] - detPreG[ipW]) / dx;
			Dy = 0.5*(detPreG[ipN] - detPreG[ipS]) / dx;
			Dz = 0.5*(detPreG[ipT] - detPreG[ipB]) / dx;

			Px = 0.5*(Psi[ipE] - Psi[ipW]) / dx;
			Py = 0.5*(Psi[ipN] - Psi[ipS]) / dx;
			Pz = 0.5*(Psi[ipT] - Psi[ipB]) / dx;

		/*	Sterm[j1] = -0.5 * (  (u[j1] + uMax)* PDxPositive+ ( v[j1] + vMax) * PDyPositive + (w[j1] + wMax) * PDzPositive
				                 + (u[j1] - uMax)* PDxNegative + (v[j1] - vMax) * PDyNegative + (w[j1] - wMax) * PDzNegative )
				         + Ds * (Dx * Px+ Dy * Py+ Dz * Pz);
*/

/*
			Sterm[ip] = -0.5*alpha*(  
(ux[ip]+uMax)*(Psi[ip]*detPreG[ip]-Psi[ipW]*detPreG[ipW]) + 
(uy[ip]+vMax)*(Psi[ip]*detPreG[ip]-Psi[ipS]*detPreG[ipS]) + 
(uz[ip]+wMax)*(Psi[ip]*detPreG[ip]-Psi[ipB]*detPreG[ipB])  ) 
            -0.5*alpha*( 
(ux[ip]-uMax)*(Psi[ipE]*detPreG[ipE]-Psi[ip]*detPreG[ip]) + 
(uy[ip]-vMax)*(Psi[ipN]*detPreG[ipN]-Psi[ip]*detPreG[ip]) + 
(uz[ip]-wMax)*(Psi[ipT]*detPreG[ipT]-Psi[ip]*detPreG[ip])  )
+ Ds * (Dx * Px+ Dy * Py+ Dz * Pz);
*/
			Sterm[ip] = -0.5*alpha*(  
(ux[ip]+uMax)*(Psi[ip]-Psi[ipW])*detPreG[ip] + 
(uy[ip]+vMax)*(Psi[ip]-Psi[ipS])*detPreG[ip] + 
(uz[ip]+wMax)*(Psi[ip]-Psi[ipB])*detPreG[ip]    ) 
            -0.5*alpha*( 
(ux[ip]-uMax)*(Psi[ipE]-Psi[ip])*detPreG[ip] + 
(uy[ip]-vMax)*(Psi[ipN]-Psi[ip])*detPreG[ip] + 
(uz[ip]-wMax)*(Psi[ipT]-Psi[ip])*detPreG[ip]  )
+ Ds * (Dx * Px+ Dy * Py+ Dz * Pz);// this works for no velocity
		}
	}
}
__global__ void getStermBlock_temp_Phase_weno(double *Sterm, int *imageG, double *ux, double *uy, double *Psi,  double *detPreG, double *fluxZPositiveCap, double *fluxZNegativeCap, 
double *fluxXPositive,double *fluxXNegative,double *fluxYPositive,double *fluxYNegative,double *fluxXPositiveCap,double *fluxXNegativeCap,double *fluxYPositiveCap,double *fluxYNegativeCap,
double Ds,double alpha,double uMax,double vMax, double dx,int NX, int NY, int NZblock)
//  no weno, pending usage
{
	int jx, jy, jz, ip;
	int y_n, x_e, y_s, x_w, z_t, z_b;
	int ipE, ipW, ipN, ipS, ipT, ipB;	 
	double Px, Py, Pz;
	double Dx, Dy, Dz;
	double gama_P1, gama_P2, F_P1, F_P2, beta_P1, beta_P2, omegabar_P1, omegabar_P2, omega_P1, omega_P2 ;
	double gama_N1, gama_N2, F_N1, F_N2, beta_N1, beta_N2, omegabar_N1, omegabar_N2, omega_N1, omega_N2 ;
	double epsilon = 1e-6;

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	gama_P1 = 1.0 / 3.0;
	gama_P2 = 2.0 / 3.0;
	gama_N1 = 2.0 / 3.0;
	gama_N2 = 1.0 / 3.0;

	for (jz = 1; jz <= NZblock; jz++) {
		ip = jz * (NY + 2)*(NX + 2) + jy * (NX + 2) + jx;		 
		fluxXPositive[ip] = 0.5 * ( ux[ip] + uMax ) * Psi[ip]*detPreG[ip];  //single value
		fluxXNegative[ip] = 0.5 * ( ux[ip] - uMax ) * Psi[ip]*detPreG[ip];

		fluxYPositive[ip] = 0.5 * ( uy[ip] + vMax ) * Psi[ip]*detPreG[ip];
		fluxYNegative[ip] = 0.5 * ( uy[ip] - vMax ) * Psi[ip]*detPreG[ip]; 	 
	}

	for (jz = 1; jz <= NZblock; jz++) { 
		ip = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
		if (imageG[ip] == 0)
		{
			x_e = (jx) % (NX)+1;
            x_w = NX - (NX + 1 - jx) % NX;			
			y_n = (jy) % (NY)+1;
			y_s = NY - (NY + 1 - jy) % NY;         

			ipE = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_e;
			ipW = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_w;
			ipN = jz * (NX + 2)*(NY + 2) + y_n * (NX + 2) + jx;
			ipS = jz * (NX + 2)*(NY + 2) + y_s * (NX + 2) + jx; 

			//=====================================================================================================
			// EW direction    POSITIVE flux f^+  j+1/2
			F_P1 = (-fluxXPositive[ipW] + 3.0 * fluxXPositive[ip]) / 2.0;
			F_P2 = (fluxXPositive[ip] + fluxXPositive[ipE]) / 2.0;
			beta_P1 = pow(fluxXPositive[ip] - fluxXPositive[ipW], 2);// need test 2   2.0
			beta_P2 = pow(fluxXPositive[ipE] - fluxXPositive[ip], 2);

			omegabar_P1 = gama_P1 / pow(epsilon + beta_P1, 2);
			omegabar_P2 = gama_P2 / pow(epsilon + beta_P2, 2);
			omega_P1 = omegabar_P1 / (omegabar_P1 + omegabar_P2);
			omega_P2 = omegabar_P2 / (omegabar_P1 + omegabar_P2);

			fluxXPositiveCap[ip] = omega_P1 * F_P1 + omega_P2 * F_P2; 
			  
			//EW direction NEGATIVE flux        f^-  j+1/2
			F_N1 = (fluxXNegative[ipW] + fluxXNegative[ip]) / 2.0;
			F_N2 = (3.0 * fluxXNegative[ip] - fluxXNegative[ipE]) / 2.0;
			beta_N1 = pow(fluxXNegative[ip] - fluxXNegative[ipW], 2);
			beta_N2 = pow(fluxXNegative[ipE] - fluxXNegative[ip], 2);

			omegabar_N1 = gama_N1 / pow(epsilon + beta_N1, 2);
			omegabar_N2 = gama_N2 / pow(epsilon + beta_N2, 2);
			omega_N1 = omegabar_N1 / (omegabar_N1 + omegabar_N2);
			omega_N2 = omegabar_N2 / (omegabar_N1 + omegabar_N2);

			fluxXNegativeCap[ip] = omega_N1 * F_N1 + omega_N2 * F_N2;	
			//=====================================================================================================
			// NS direction    POSITIVE flux f^+  j+1/2
			F_P1 = (-fluxYPositive[ipS] + 3.0 * fluxYPositive[ip]) / 2.0;
			F_P2 = (fluxYPositive[ip] + fluxYPositive[ipN] ) / 2.0;
			beta_P1 = pow(fluxYPositive[ip] - fluxYPositive[ipS] , 2);// need test 2   2.0
			beta_P2 = pow(fluxYPositive[ipN] - fluxYPositive[ip] , 2);

			omegabar_P1 = gama_P1 / pow(epsilon + beta_P1, 2);
			omegabar_P2 = gama_P2 / pow(epsilon + beta_P2, 2);
			omega_P1 = omegabar_P1 / (omegabar_P1 + omegabar_P2);
			omega_P2 = omegabar_P2 / (omegabar_P1 + omegabar_P2);

			fluxYPositiveCap[ip] = omega_P1 * F_P1 + omega_P2 * F_P2; 
			//NS direction NEGATIVE flux        f^-  j+1/2
			F_N1 = (fluxYNegative[ipS] + fluxYNegative[ip] ) / 2.0;
			F_N2 = (3.0 * fluxYNegative[ip] - fluxYNegative[ipN] ) / 2.0;
			beta_N1 = pow(fluxYNegative[ip] - fluxYNegative[ipS] , 2);
			beta_N2 = pow(fluxYNegative[ipN] - fluxYNegative[ip] , 2);

			omegabar_N1 = gama_N1 / pow(epsilon + beta_N1, 2);
			omegabar_N2 = gama_N2 / pow(epsilon + beta_N2, 2);
			omega_N1 = omegabar_N1 / (omegabar_N1 + omegabar_N2);
			omega_N2 = omegabar_N2 / (omegabar_N1 + omegabar_N2);

			fluxYNegativeCap[ip] = omega_N1 * F_N1 + omega_N2 * F_N2;
			//=====================================================================================================
			
		}//if imageG==0
		else
		{
			fluxXPositiveCap[ip] = 0.0; 
			fluxXPositiveCap[ip] = 0.0; 
			fluxYPositiveCap[ip] = 0.0; 
			fluxYNegativeCap[ip] = 0.0; 			
		} 
	}//for (jz = 1; jz <= NZblock; jz++) 
 
	for (jz = 1; jz <= NZblock; jz++) {
			ip = jz * (NY + 2)*(NX + 2) + jy * (NX + 2) + jx;
		if (imageG[ip] == 0) {

			x_e = jx % NX + 1;
			x_w = NX - (NX + 1 - jx) % NX;
			y_n = jy % NY + 1;
			y_s = NY - (NY + 1 - jy) % NY;
			z_t = jz + 1;
			z_b = jz - 1;

			//center 6
			ipE = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_e;
			ipW = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_w;
			ipN = jz * (NX + 2)*(NY + 2) + y_n * (NX + 2) + jx;
			ipS = jz * (NX + 2)*(NY + 2) + y_s * (NX + 2) + jx;
			ipT = z_t * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
			ipB = z_b * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;

			Dx = 0.5*(detPreG[ipE] - detPreG[ipW]) / dx;
			Dy = 0.5*(detPreG[ipN] - detPreG[ipS]) / dx;
			Dz = 0.5*(detPreG[ipT] - detPreG[ipB]) / dx;

			Px = 0.5*(Psi[ipE] - Psi[ipW]) / dx;
			Py = 0.5*(Psi[ipN] - Psi[ipS]) / dx;
			Pz = 0.5*(Psi[ipT] - Psi[ipB]) / dx; 

 
			Sterm[ip] = -alpha * ( fluxXPositiveCap[ip] - fluxXPositiveCap[ipW] +fluxXNegativeCap[ipE] - fluxXNegativeCap[ip] +
									fluxYPositiveCap[ip] - fluxYPositiveCap[ipS] +fluxYNegativeCap[ipN] - fluxYNegativeCap[ip] +
									fluxZPositiveCap[ip] - fluxZPositiveCap[ipB] +fluxZNegativeCap[ipT] - fluxZNegativeCap[ip] )+ Ds * (Dx * Px+ Dy * Py+ Dz * Pz); 
   
 

 
		}
	}

 
 
	 
}

__global__ void getZFlux_weno(double *fluxZPositive,double *fluxZNegative,    double *uz, double *Psi,  double *detPreG,double wMax,  int NX, int NY, int NZblock)
//  no weno, pending usage
{
	int jx, jy, jz, ip;
	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZblock; jz++) 
	{
		ip = jz * (NY + 2)*(NX + 2) + jy * (NX + 2) + jx;
		fluxZPositive[ip] = 0.5 * ( uz[ip] + wMax ) * Psi[ip]*detPreG[ip];
		fluxZNegative[ip] = 0.5 * ( uz[ip] - wMax ) * Psi[ip]*detPreG[ip];	 
	}	 
}

__global__ void getZFluxCap_weno(double *fluxZPositiveCap,double *fluxZNegativeCap, int *imageG,   double *fluxZPositive,double *fluxZNegative  ,int NX, int NY, int NZblock)
//  no weno, pending usage
{
	int jx, jy, jz, ip;
	int   z_t, z_b;
	int   ipT, ipB;	 
	double gama_P1, gama_P2, F_P1, F_P2, beta_P1, beta_P2, omegabar_P1, omegabar_P2, omega_P1, omega_P2 ;
	double gama_N1, gama_N2, F_N1, F_N2, beta_N1, beta_N2, omegabar_N1, omegabar_N2, omega_N1, omega_N2 ;
	double epsilon = 1e-6;

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	gama_P1 = 1.0 / 3.0 ;
	gama_P2 = 2.0  / 3.0 ;
	gama_N1 = 2.0  / 3.0 ;
	gama_N2 = 1.0  / 3.0 ; 

	for (jz = 1; jz <= NZblock; jz++) { 
		ip = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
		if (imageG[ip] == 0)
		{
            z_t = jz + 1 ;
			z_b = jz - 1 ; 
			ipT = z_t * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
			ipB = z_b * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx; 
			//=====================================================================================================
			// TB direction    POSITIVE flux f^+  j+1/2
			F_P1 = (-fluxZPositive[ipB] + 3.0 * fluxZPositive[ip]) / 2.0;
			F_P2 = (fluxZPositive[ip]  + fluxZPositive[ipT] ) / 2.0;
			beta_P1 = pow(fluxZPositive[ip] - fluxZPositive[ipB] , 2);// need test 2   2.0
			beta_P2 = pow(fluxZPositive[ipT] - fluxZPositive[ip] , 2);

			omegabar_P1 = gama_P1 / pow(epsilon + beta_P1, 2);
			omegabar_P2 = gama_P2 / pow(epsilon + beta_P2, 2);
			omega_P1 = omegabar_P1 / (omegabar_P1 + omegabar_P2);
			omega_P2 = omegabar_P2 / (omegabar_P1 + omegabar_P2);

			fluxZPositiveCap[ip] = omega_P1 * F_P1 + omega_P2 * F_P2; 
			//TB direction NEGATIVE flux        f^-  j+1/2
			F_N1 = (fluxZNegative[ipB] + fluxZNegative[ip]) / 2.0;
			F_N2 = (3.0 * fluxZNegative[ip] - fluxZNegative[ipT]) / 2.0;
			beta_N1 = pow(fluxZNegative[ip] - fluxZNegative[ipB], 2);
			beta_N2 = pow(fluxZNegative[ipT] - fluxZNegative[ip], 2);

			omegabar_N1 = gama_N1 / pow(epsilon + beta_N1, 2);
			omegabar_N2 = gama_N2 / pow(epsilon + beta_N2, 2);
			omega_N1 = omegabar_N1 / (omegabar_N1 + omegabar_N2);
			omega_N2 = omegabar_N2 / (omegabar_N1 + omegabar_N2);

			fluxZNegativeCap[ip] = omega_N1 * F_N1 + omega_N2 * F_N2;
		}//if imageG==0
		else
		{ 
			fluxZPositiveCap[ip] = 0.0;  
			fluxZNegativeCap[ip] = 0.0; 
		} 
	}//for (jz = 1; jz <= NZblock; jz++)  
}
 
void getDETofGradientBlockCPU(double *Determinant, int *image, double *phaseField, int NX, int NY, int NZ)  //get | \-/ pN |
{
	int jx, jy, jz, ip; 
	int y_n, x_e, y_s, x_w, z_t, z_b; 
	double Rx, Ry, Rz;
	int ipE, ipW, ipN, ipS, ipT, ipB; 

	for (jz = 1; jz <= NZ ; jz++)
	for (jy = 1; jy <= NY ; jy++)
	for (jx = 1; jx <= NX ; jx++)
	{
		ip = jz * (NY + 2)*(NX + 2) + jy * (NX + 2) + jx;
		if (image[ip] == 0) {
			x_e = jx % NX + 1;
			x_w = NX - (NX + 1 - jx) % NX;
			y_n = jy % NY + 1;
			y_s = NY - (NY + 1 - jy) % NY;
			z_t = jz + 1;
			z_b = jz - 1;

			//center 6
			ipE = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_e;
			ipW = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_w;
			ipN = jz * (NX + 2)*(NY + 2) + y_n * (NX + 2) + jx;
			ipS = jz * (NX + 2)*(NY + 2) + y_s * (NX + 2) + jx;
			ipT = z_t * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
			ipB = z_b * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;

			Rx = (phaseField[ipE] - phaseField[ipW]) / 2;
			Ry = (phaseField[ipN] - phaseField[ipS]) / 2;
			Rz = (phaseField[ipT] - phaseField[ipB]) / 2;

			Determinant[ip] = sqrt(Rx*Rx + Ry * Ry + Rz * Rz);
		}
		else Determinant[ip] = 0.0;
    }	
}

 
void setPhaseField3D_boundaryBlue(int *image, double *phaseField, double *rho_r, double *rho_b, int NX, int NY, int NZ)
{
	int x, y, z, j1;
	for (z = 1; z < NZ + 1; z++)
	for (y = 1; y < NY + 1; y++)
	for (x = 1; x < NX + 1; x++){
		j1 = z * (NX + 2)*(NY + 2) + y * (NX + 2) + x;
		if (image[j1] != 5)	{	phaseField[j1] = (rho_r[j1] - rho_b[j1]) / (rho_r[j1] + rho_b[j1]);	}
		else	{	phaseField[j1] = -1;	}
	}
}	





