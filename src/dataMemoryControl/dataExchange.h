/*file statement



*/

#ifndef dataExchange
#define dataExchange 

 
 
inline void   swap (   double   **f,     double   **fn )  // purpos     :  update the variable fn --> f 
{  double *tmp = *f;   *f = *fn;   *fn = tmp;  }
inline void   swap (   float  **f,     float  **fn )  // purpos     :  update the variable fn --> f 
{  float *tmp = *f;   *f = *fn;   *fn = tmp;  }

 

template <typename T >
void   assembleDisToSend(int IDcore,T *DTota1, T *DLocal, int NX, int NY,int NZ, int NZ_sub){	
	int LineNumTotal = (NY + 2)*(NX + 2)*(NZ + 2);
	int LineNumSub = (NY + 2)*(NX + 2)*(NZ_sub + 2);
	int LineNumSubNet = (NY + 2)*(NX + 2)*(NZ_sub);

	for (int jz = 1; jz <= NZ_sub; jz++)
	for (int jy = 1; jy <= NY; jy++)
	for (int jx = 1; jx <= NX; jx++)	{
		int j_T = LineNumSubNet*(IDcore-1)+ jz*(NY + 2)*(NX + 2) + jy*(NX + 2) + jx;
		int j_L = jz*(NY + 2)*(NX + 2) + jy*(NX + 2) + jx;
		for (int k = 0; k < 19; k++)		{
			DLocal[j_L + k * LineNumSub] = DTota1[j_T + k * LineNumTotal];			 
		}
	}
}

template <typename T >
void   assembleDisBC_EX(int IDcore, T *DataLocal, T *DataBC, int NX, int NY, int NZ_sub, int N){	 
	int LineNumLocal = (NY + 2)*(NX + 2)*(NZ_sub + 2);	 
	int LineNumSubNet = (NY + 2)*(NX + 2);
		 
	for (int jy = 1; jy <= NY; jy++)
	for (int jx = 1; jx <= NX; jx++)	{		 
		int j_L = N * (NY + 2)*(NX + 2) + jy*(NX + 2) + jx;		 
		int j_BC =   jy*(NX + 2) + jx;
		for (int k = 0; k < 19; k++){
			DataBC[j_BC + k * LineNumSubNet] = DataLocal[j_L + k * LineNumLocal];
		}
	}
}
template <typename T >
void   assemble_BC_Local_dis(int IDcore, T *DataBC, T *DataLocal, int NX, int NY, int NZ_sub, int N){	
	int LineNumLocal = (NY + 2)*(NX + 2)*(NZ_sub + 2);	 
	int LineNumSubNet = (NY + 2)*(NX + 2);	
	for (int jy = 1; jy <= NY; jy++)
	for (int jx = 1; jx <= NX; jx++){
		int j_L = N * (NY + 2)*(NX + 2) + jy*(NX + 2) + jx;//N=0,assemble to 0 row;N=N,assemble to N row.
		int j_BC = jy*(NX + 2) + jx;
		for (int k = 0; k < 19; k++){
			DataLocal[j_L + k * LineNumLocal] = DataBC[j_BC + k * LineNumSubNet]  ;
		}
	}
}


#endif
