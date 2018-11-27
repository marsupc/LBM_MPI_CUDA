/*file statement



*/

#ifndef LBMFDM
#define LBMFDM

  
template <typename T>
__global__ void getDETofGradientBlock(int *image, T *phaseField, T *Determinant, int NX, int NY, int NZ)  //get | \-/ pN |
{
	int jx, jy, jz, j1; 
	int y_n, x_e, y_s, x_w, z_t, z_d; 
	T Rx, Ry, Rz;
	int ipE, ipW, ipN, ipS, ipT, ipD;
	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZ; jz++){
		if (image[j1] == 0) {
			j1 = jz * (NY + 2)*(NX + 2) + jy * (NX + 2) + jx;

			x_e = jx % NX + 1;
			x_w = NX - (NX + 1 - jx) % NX;
			y_n = jy % NY + 1;
			y_s = NY - (NY + 1 - jy) % NY;
			z_t = jz + 1;
			z_d = jz - 1;

			//center 6
			ipE = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_e;
			ipW = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_w;
			ipN = jz * (NX + 2)*(NY + 2) + y_n * (NX + 2) + jx;
			ipS = jz * (NX + 2)*(NY + 2) + y_s * (NX + 2) + jx;
			ipT = z_t * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
			ipD = z_d * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;

			Rx = (phaseField[ipE] - phaseField[ipW]) / 2;
			Ry = (phaseField[ipN] - phaseField[ipS]) / 2;
			Rz = (phaseField[ipT] - phaseField[ipD]) / 2;

			Determinant[j1] = sqrt(Rx*Rx + Ry * Ry + Rz * Rz);
		}
    }	
}


template <typename T>
__global__ void getMMatrixBlock(int *image,T *Determinant,T halfLamda , T *MMatrix, int NX, int NY, int NZ)
{
	int jx, jy, jz, j1;	

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZ; jz++) {
		if (image[j1] == 0) {
			j1 = jz * (NY + 2)*(NX + 2) + jy * (NX + 2) + jx;
			MMatrix[j1] = Determinant[j1]* halfLamda;
		}
	}
}

template <typename T>
__global__ void getSStermBlock(int *image, T *u, T *v, T *w, T *Psi, T *Determinant,  T *MMatrix, T Ds, T uMax, T vMax, T wMax, T dx,T *SSterm, int NX, int NY, int NZblock)
//  no weno, pending usage
{
	int jx, jy, jz, ip;
	T Rx, Ry, Rz;
	int y_n, x_e, y_s, x_w, z_t, z_d;
	T Px, Py, Pz;
	T Dx, Dy, Dz;
	T PDxPositive, PDyPositive, PDzPositive;
	T PDxNegative, PDyNegative, PDzNegative;
	int ipE, ipW, ipN, ipS, ipT, ipD;

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZblock; jz++) {
		if (image[ip] == 0) {
			ip = jz * (NY + 2)*(NX + 2) + jy * (NX + 2) + jx;
			x_e = jx % NX + 1;
			x_w = NX - (NX + 1 - jx) % NX;
			y_n = jy % NY + 1;
			y_s = NY - (NY + 1 - jy) % NY;
			z_t = jz + 1;
			z_d = jz - 1;

			//center 6
			ipE = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_e;
			ipW = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_w;
			ipN = jz * (NX + 2)*(NY + 2) + y_n * (NX + 2) + jx;
			ipS = jz * (NX + 2)*(NY + 2) + y_s * (NX + 2) + jx;
			ipT = z_t * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
			ipD = z_d * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;

			PDxPositive = (Psi[ip] * Determinant[ip] - Psi[ipW] * Determinant[ipW]) / dx;
			PDyPositive = (Psi[ip] * Determinant[ip] - Psi[ipS] * Determinant[ipS]) / dx;
		    PDzPositive = (Psi[ip] * Determinant[ip] - Psi[ipD] * Determinant[ipD]) / dx;

			PDxNegative = (Psi[ipE] * Determinant[ipE] - Psi[ip] * Determinant[ip]) / dx;
			PDyNegative = (Psi[ipN] * Determinant[ipN] - Psi[ip] * Determinant[ip]) / dx;
			PDzNegative = (Psi[ipT] * Determinant[ipT] - Psi[ip] * Determinant[ip]) / dx;

			Dx = 0.5*(Determinant[ipE] - Determinant[ipW]) / dx;
			Dy = 0.5*(Determinant[ipN] - Determinant[ipS]) / dx;
			Dz = 0.5*(Determinant[ipT] - Determinant[ipD]) / dx;

			Px = 0.5*(Psi[ipE] - Psi[ipW]) / dx;
			Py = 0.5*(Psi[ipN] - Psi[ipS]) / dx;
			Pz = 0.5*(Psi[ipT] - Psi[ipD]) / dx;

			SSterm[j1] = -0.5 * (  (u[j1] + uMax)* PDxPositive+ ( v[j1] + vMax) * PDyPositive + (w[j1] + wMax) * PDzPositive
				                 + (u[j1] - uMax)* PDxNegative + (v[j1] - vMax) * PDyNegative + (w[j1] - wMax) * PDzNegative )
				         + Ds * (Dx * Px+ Dy * Py+ Dz * Pz);
		}
	}
}

template <typename T>
__global__ void _GetWenoFlux(int *imageG,  T *Psi, T *u, T uMax,T *DET,T *FlunxPositive, T *FlunxNegative, int NX, int NY, int NZblock)

{//u fake =0.1;  need to find out the maximum value.
	int jx, jy, jz;
	int x_e, x_w;  // y_n, y_s, z_t, z_b;
	//int x_ee, x_ww, y_nn, y_ss, z_tt, z_bb;
	int ip, ipE, ipW;
	//ipN, ipS, ipF, ipB;
	//int      ipEE, ipWW, ipNN, ipSS, ipFF, ipBB;
	T PsiR, PsiL;
	T gama_P1, gama_P2, F_P1, F_P2, beta_P1, beta_P2, omegabar_P1, omegabar_P2, omega_P1, omega_P2, PsiWenoZ_P;
	T gama_N1, gama_N2, F_N1, F_N2, beta_N1, beta_N2, omegabar_N1, omegabar_N2, omega_N1, omega_N2, PsiWenoZ_N;
	T epsilon = 1e-6;
	T PsiWenoZ_PR, PsiWenoZ_PL, PsiWenoZ_NR, PsiWenoZ_NL;

	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;
	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;

	gama_P1 = 1 / 3;
	gama_P2 = 2 / 3;
	gama_N1 = 2 / 3;
	gama_N2 = 1 / 3;

	for (jz = 1; jz <= NZblock; jz++)
	{
		ip = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;	 

		if (imageG[ip] == 0)
		{
			x_e = (jx) % (NX)+1;
            x_w = NX - (NX + 1 - jx) % NX;			
			
			//y_n = (jy) % (NY)+1;
			//y_s = NY - (NY + 1 - jy) % NY;
            //z_t = (jz) % (NZ)+1;
			//z_b = NZ - (NZ + 1 - jz) % NZ;

			//x_ee = x_e % NX + 1;
			//x_ww = NX - (NX + 1 - x_w) % NX;
			//y_nn = y_n % NY + 1;
			//y_ss = NY - (NY + 1 - y_s) % NY;
			//z_tt = z_t % NZ + 1;
			//z_bb = NZ - (NZ + 1 - z_b) % NZ;

			ipE = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_e;
			ipW = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_w;
			//ipN = jz * (NX + 2)*(NY + 2) + y_n * (NX + 2) + jx;
			//ipS = jz * (NX + 2)*(NY + 2) + y_s * (NX + 2) + jx;
			//ipF = z_t * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
			//ipB = z_b * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;

			//-------------------------------------------------------------------
			//right flux
			//PsiPositive=0.5*(Psi[ip]+a*w);	
			//PsiNegative=0.5*(Psi[ip]-a*w);
			// f^+  j+1/2
			F_P1 = (-Psi[ipW]* DET[ipW] + 3.0 * Psi[ip]*DET[ip]) / 2.0;
			F_P2 = (Psi[ip] * DET[ip] + Psi[ipE] * DET[ipE]) / 2.0;
			beta_P1 = pow(Psi[ip] * DET[ip] - Psi[ipW] * DET[ipW], 2);// need test 2   2.0
			beta_P2 = pow(Psi[ipE] * DET[ipE] - Psi[ip] * DET[ip], 2);

			omegabar_P1 = gama_P1 / pow(epsilon + beta_P1, 2);
			omegabar_P2 = gama_P2 / pow(epsilon + beta_P2, 2);
			omega_P1 = omegabar_P1 / (omegabar_P1 + omegabar_P2);
			omega_P2 = omegabar_P2 / (omegabar_P1 + omegabar_P2);

			FlunxPositive[ip] = 0.5*(omega_P1 * F_P1 + omega_P2 * F_P2)+0.5*uMax*Psi[ip]; 

			//-------------------------------------------------------------------
			//left flux
			// f^-  j+1/2
			F_N1 = (Psi[ipW] * DET[ipW] + Psi[ip] * u[ip]) / 2;
			F_N2 = (3 * Psi[ip] * u[ip] - Psi[ipE] * u[ipE]) / 2;
			beta_N1 = pow(Psi[ip] * u[ip] - Psi[ipW] * u[ipW], 2);
			beta_N2 = pow(Psi[ipE] * u[ipE] - Psi[ip] * u[ip], 2);

			omegabar_N1 = gama_N1 / pow(epsilon + beta_N1, 2);
			omegabar_N2 = gama_N2 / pow(epsilon + beta_N2, 2);
			omega_N1 = omegabar_N1 / (omegabar_N1 + omegabar_N2);
			omega_N2 = omegabar_N2 / (omegabar_N1 + omegabar_N2);

			FlunxNegative[ip] = 0.5*(omega_N1 * F_N1 + omega_N2 * F_N2)-0.5*uMax*Psi[ip];			 
		}
		else
		{
			FlunxPositive[ip] = 0;
			FlunxNegative[ip] = 0;
		}
	}
}
template <typename T>
__global__ void getSStermWenoBlock(int *image, T *u, T *v, T *w, T *Psi, T *Determinant, T *MMatrix,
	T *FluxPositiveX, T *FluxPositiveY, T *FluxPositiveZ, T *FluxNegativeX, T *FluxNegativeY, T *FluxNegativeZ,
	T Ds, T uMax, T vMax, T wMax, T dx, T *Sterm, int NX, int NY, int NZ)
{
	int jx, jy, jz, ip;	 
	int y_n, x_e, y_s, x_w, z_t, z_d;
	int ipE, ipW, ipN, ipS, ipT, ipD;
	T Px, Py, Pz;
	T Dx, Dy, Dz;
	T PDx, PDy, PDz;

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZ; jz++) {
		if (image[ip] == 0) {
			ip = jz * (NY + 2)*(NX + 2) + jy * (NX + 2) + jx;
			x_e = jx % NX + 1;
			x_w = NX - (NX + 1 - jx) % NX;
			y_n = jy % NY + 1;
			y_s = NY - (NY + 1 - jy) % NY;
			z_t = jz + 1;
			z_d = jz - 1;

			//center 6
			ipE = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_e;
			ipW = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_w;
			ipN = jz * (NX + 2)*(NY + 2) + y_n * (NX + 2) + jx;
			ipS = jz * (NX + 2)*(NY + 2) + y_s * (NX + 2) + jx;
			ipT = z_t * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
			ipD = z_d * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;

			PDx = 0.5*(u[j1] + uMax)*(FluxPositiveX[ip] - FluxPositiveX[ipW]) / dx + 0.5*(u[j1] - uMax)*(FluxNegativeX[ipE] - FluxNegativeX[ip]) / dx;//input variable
			PDy = 0.5*(v[j1] + vMax)*(FluxPositiveY[ip] - FluxPositiveY[ipS]) / dx + 0.5*(v[j1] - vMax)*(FluxNegativeY[ipN] - FluxNegativeY[ip]) / dx;
			PDz = 0.5*(w[j1] + wMax)*(FluxPositiveZ[ip] - FluxPositiveZ[ipD]) / dx + 0.5*(w[j1] - wMax)*(FluxNegativeZ[ipT] - FluxNegativeZ[ip]) / dx;

			Dx = 0.5*(Determinant[ipE] - Determinant[ipW]) / dx;
			Dy = 0.5*(Determinant[ipN] - Determinant[ipS]) / dx;
			Dz = 0.5*(Determinant[ipT] - Determinant[ipD]) / dx;

			Px = 0.5*(Psi[ipE] - Psi[ipW]) / dx;
			Py = 0.5*(Psi[ipN] - Psi[ipS]) / dx;
			Pz = 0.5*(Psi[ipT] - Psi[ipD]) / dx;

			Sterm[j1] = -( PDx +  PDy +  PDz) + Ds * (Dx * Px + Dy * Py + Dz * Pz);
	 
		}
	}
}

template <typename T>
__global__ void assembleBvectorBlock(int *image, int tstep, T *B, T *MMatrix, T *DETCurrent,T *u, T *v, T *w, T *Psi,  T *StermCurrent, T *StermPre, T Ds, T uMax, T vMax, T wMax, T dx, T *SSterm, int NX, int NY, int NZ)
{
	int jx, jy, jz, ip;
	int y_n, x_e, y_s, x_w, z_t, z_d;
	int ipE, ipW, ipN, ipS, ipT, ipD;

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZ; jz++) {
		 
			ip = jz * (NY + 2)*(NX + 2) + jy * (NX + 2) + jx;
			x_e = jx % NX + 1;
			x_w = NX - (NX + 1 - jx) % NX;
			y_n = jy % NY + 1;
			y_s = NY - (NY + 1 - jy) % NY;
			z_t = jz + 1;
			z_d = jz - 1;

			ipE = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_e;
			ipW = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + x_w;
			ipN = jz * (NX + 2)*(NY + 2) + y_n * (NX + 2) + jx;
			ipS = jz * (NX + 2)*(NY + 2) + y_s * (NX + 2) + jx;
			ipT = z_t * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
			ipD = z_d * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;

			if (tstep == 1)
			{
				B[j1] = (DETCurrent[ip] - 6 * MMatrix[ip]) * Psi[ip] + MMatrix[ip] * (Psi[ipE] + Psi[ipW] + Psi[ipN] + Psi[ipS] + Psi[ipT] + Psi[ipD]) + StermCurrent[ip];
			}
			else {
				B[j1] = (DETCurrent[ip] - 6 * MMatrix[ip]) * Psi[ip] + MMatrix[ip] * (Psi[ipE] + Psi[ipW] + Psi[ipN] + Psi[ipS] + Psi[ipT] + Psi[ipD]) +1.5* StermCurrent[ip] -0.5*StermPre[ip];
			}
			

		 
	}
}

template <typename T>
void getElectricForceBlock(int *image,   T *Psi,   T *ForceEX, T *ForceEY, T *ForceEZ, int NX, int NY, int NZ)
{
	int ix, iy, iz, ip, jx, jy, jz, jp;
	int y_n, x_e, y_s, x_w, z_t, z_d;
	int ipE, ipW, ipN, ipS, ipT, ipD;

	for (ix = 1; ix <= NX; ix++)
	for (iy = 1; iy <= NY; iy++)
	for (iz = 1; iz <= NZ; iz++) {
     ip = iz * (NY + 2)*(NX + 2) + iy * (NX + 2) + ix;
		for (jx = 1; jx <= NX; jx++)
			for (jy = 1; jy <= NY; jy++)
				for (jz = 1; jz <= NZ; jz++) {
					jp = jz * (NY + 2)*(NX + 2) + jy * (NX + 2) + jx;

					T px = (T)ix - (T)jx;
					T pyI = (T)iy - (T)jy;
					T pyO = (T)iy + (T)jy;
					T pz = (T)iz - (T)jz;

					T distanceI = sqrt(px * px + pyI * pyI + pz * pz);
					T distanceO = sqrt(px * px + pyO * pyO + pz * pz);  //Y axis
					T forceI = 0.01*Psi[jp] * Psi[jp] / distanceI / distanceI;
					T forceO = 0.01*Psi[jp] * Psi[jp] / distanceO / distanceO;

					if (ip != jp)
					{
                    ForceEX[ip] += forceI * px / distanceI; // force three direction
					ForceEY[ip] += forceI * pyI/ distanceI;
					ForceEZ[ip] += forceI * pz / distanceI;

					ForceEX[ip] += forceO * px / distanceO; // force three direction
					ForceEY[ip] += forceO * pyO/ distanceO;
					ForceEZ[ip] += forceO * pz / distanceO;
					}
				}

	}
}

#endif