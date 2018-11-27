/*file statement



*/

#ifndef latticeBoltzmann
#define latticeBoltzmann

template <typename T>
__global__ void stream3D_Block (T *f,T *fN,int *image, int NX, int NY, int NZ){
    int jx, jy, jz, j1;
	int y_n, x_e, y_s, x_w, z_t, z_b;	
	int LineNum = (NY + 2)*(NX + 2)*(NZ + 2);  
 	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZ; jz++){
		y_n = (jy) % (NY)+1;
		x_e = (jx) % (NX)+1;
        z_t = jz+1;
		y_s = NY - (NY + 1 - jy) % NY;
		x_w = NX - (NX + 1 - jx) % NX;
        z_b = jz-1;		
 
		j1 = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx; 

		fN[j1] = f[j1];
		fN[j1 + 1 *  (LineNum)] = f[1 * (LineNum)+jz*  (NX + 2)*(NY + 2) + jy* (NX + 2) + x_w];
		fN[j1 + 2 *  (LineNum)] = f[2 * (LineNum)+jz*  (NX + 2)*(NY + 2) + jy* (NX + 2) + x_e];
		fN[j1 + 3 *  (LineNum)] = f[3 * (LineNum)+jz*  (NX + 2)*(NY + 2) + y_s*(NX + 2) + jx];
		fN[j1 + 4 *  (LineNum)] = f[4 * (LineNum)+jz*  (NX + 2)*(NY + 2) + y_n*(NX + 2) + jx];
		fN[j1 + 5 *  (LineNum)] = f[5 * (LineNum)+z_b* (NX + 2)*(NY + 2) + jy* (NX + 2) + jx];
		fN[j1 + 6 *  (LineNum)] = f[6 * (LineNum)+z_t* (NX + 2)*(NY + 2) + jy* (NX + 2) + jx];
		fN[j1 + 7 *  (LineNum)] = f[7 * (LineNum)+jz*  (NX + 2)*(NY + 2) + y_s*(NX + 2) + x_w];
		fN[j1 + 8 *  (LineNum)] = f[8 * (LineNum)+jz*  (NX + 2)*(NY + 2) + y_n*(NX + 2) + x_w];
		fN[j1 + 9 *  (LineNum)] = f[9 * (LineNum)+jz*  (NX + 2)*(NY + 2) + y_s*(NX + 2) + x_e];
		fN[j1 + 10 * (LineNum)] = f[10 * (LineNum)+jz* (NX + 2)*(NY + 2) + y_n*(NX + 2) + x_e];
		fN[j1 + 11 * (LineNum)] = f[11 * (LineNum)+z_b*(NX + 2)*(NY + 2) + jy* (NX + 2) + x_w];
		fN[j1 + 12 * (LineNum)] = f[12 * (LineNum)+z_b*(NX + 2)*(NY + 2) + jy* (NX + 2) + x_e];
		fN[j1 + 13 * (LineNum)] = f[13 * (LineNum)+z_t*(NX + 2)*(NY + 2) + jy* (NX + 2) + x_w];
		fN[j1 + 14 * (LineNum)] = f[14 * (LineNum)+z_t*(NX + 2)*(NY + 2) + jy* (NX + 2) + x_e];
		fN[j1 + 15 * (LineNum)] = f[15 * (LineNum)+z_b*(NX + 2)*(NY + 2) + y_s*(NX + 2) + jx];
		fN[j1 + 16 * (LineNum)] = f[16 * (LineNum)+z_t*(NX + 2)*(NY + 2) + y_s*(NX + 2) + jx];
		fN[j1 + 17 * (LineNum)] = f[17 * (LineNum)+z_b*(NX + 2)*(NY + 2) + y_n*(NX + 2) + jx];
		fN[j1 + 18 * (LineNum)] = f[18 * (LineNum)+z_t*(NX + 2)*(NY + 2) + y_n*(NX + 2) + jx];
	} 
}

template <typename T>
__global__ void CollisionRK3D_Block (int *imageL, T *uxLocal, T *uyLocal, T *uzLocal, T *rhoLocal, T *rho_rLocal, T *rho_bLocal,
	T *ffLocal, T *f_rLocal, T *f_bLocal,
	T taur, T taub, T *t_k, T *ex, T *ey, T *ez,
	T alfa_r, T alfa_b, T c_squ, T c_squ_r, T c_squ_b,  T *f_body, int NX, int NY, int NZ_sub){

	int jx, jy, jz, j1, i;
	double fequ_r[19], fequ_b[19];
	int LineNum = (NY + 2)*(NX + 2)*(NZ_sub + 2);
	double   fei, tau;
	double alfa1, beta1, kappa1, delta1, eta1, kxi1;
	double u_n[19], u_squ;
	int k; 

	delta1 = 0.98;
	alfa1 = 2.0 * taur*taub / (taur + taub);
	beta1 = 2.0 * (taur - alfa1) / delta1;
	kappa1 = -beta1 / (2.0 * delta1);
	eta1 = 2.0*(alfa1 - taub) / delta1;
	kxi1 = eta1 / (2.0 * delta1);

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;		
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZ_sub; jz++){		
		j1 = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;
		if (imageL[j1] == 0){					 
			u_squ = uxLocal[j1] * uxLocal[j1] + uyLocal[j1] * uyLocal[j1] + uzLocal[j1] * uzLocal[j1];
			u_n[0] = 0.0;
			for (k = 1; k < 19; k++){u_n[k] = ex[k] * uxLocal[j1] + ey[k] * uyLocal[j1] + ez[k] * uzLocal[j1];}

			fequ_r[0] = t_k[0] * rho_rLocal[j1] * (u_n[0] * 3.0 + u_n[0] * u_n[0] * 4.5 - u_squ*1.5) + rho_rLocal[j1] * alfa_r;
			for (k = 1; k < 7; k++){fequ_r[k] = t_k[k] * rho_rLocal[j1] * (u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ*1.5) + rho_rLocal[j1] * (1.0 - alfa_r) / 12.0;}
			for (k = 7; k < 19; k++){fequ_r[k] = t_k[k] * rho_rLocal[j1] * (u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ*1.5) + rho_rLocal[j1] * (1.0 - alfa_r) / 24.0;}

			fequ_b[0] = t_k[0] * rho_bLocal[j1] * (u_n[0] * 3.0 + u_n[0] * u_n[0] * 4.5 - u_squ*1.5) + rho_bLocal[j1] * alfa_b;
			for (k = 1; k < 7; k++)	{	fequ_b[k] = t_k[k] * rho_bLocal[j1] * (u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ*1.5) + rho_bLocal[j1] * (1.0 - alfa_b) / 12.0;}
			for (k = 7; k < 19; k++){ fequ_b[k] = t_k[k] * rho_bLocal[j1] * (u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ*1.5) + rho_bLocal[j1] * (1.0 - alfa_b) / 24.0;}			
			////*******************compute Fequ end

			fei = (rho_rLocal[j1] - rho_bLocal[j1]) / (rho_rLocal[j1] + rho_bLocal[j1]);

			if (fei > delta1)			         	tau = taur;
			else if (fei > 0 && fei <= delta1)		tau = alfa1 + beta1*fei + kappa1*fei*fei;
			else if (fei > -delta1  && fei <= 0)	tau = alfa1 + eta1*fei + kxi1*fei*fei;
			else if (fei < -delta1)			    	tau = taub;	 

			for (i = 0; i < 19; i++){
				f_rLocal[j1 + i * LineNum] = fequ_r[i] + (1 - 1 / tau)*(f_rLocal[j1 + i * LineNum] - fequ_r[i]);
				f_bLocal[j1 + i * LineNum] = fequ_b[i] + (1 - 1 / tau)*(f_bLocal[j1 + i * LineNum] - fequ_b[i]);
				ffLocal[j1 + i * LineNum] = f_rLocal[j1 + i * LineNum] + f_bLocal[j1 + i * LineNum] + t_k[i] * ex[i] * f_body[0] / c_squ + t_k[i] * ey[i] * f_body[1] / c_squ + t_k[i] * ez[i] * f_body[2] / c_squ;
			}
		}
	} //end for jz
}
template <typename T>
__global__ void redistributeRK3D_Block(int *imageL, T *rhoL, T *rho_rL, T *rho_bL,
	T *xc, T *yc, T *zc, T *ffL, T *f_rL, T *f_bL,
	T *t_k, T A, T beta, T *rsq, T *Bi,
	T *FxL, T *FyL, T *FzL, int NX, int NY, int NZ_sub)
{
	int jx, jy, jz, j1, i, x_e, x_w, y_n, y_s, z_t, z_b;
	T feq[19], cosfai[19], fm, temp;
	int LineNum = (NY + 2)*(NX + 2)*(NZ_sub + 2); 

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZ_sub; jz++){
		y_n = (jy) % (NY)+1;
		x_e = (jx) % (NX)+1;
		z_t = jz+1;
		y_s = NY - (NY + 1 - jy) % NY;
		x_w = NX - (NX + 1 - jx) % NX;
		z_b = jz-1;
 
		j1 = jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx;

		if (imageL[j1] == 0){			
			FxL[j1]
				= xc[1] * (rho_rL[jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_e] - rho_bL[jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_e])
				+ xc[7] * (rho_rL[jz*(NX + 2)*(NY + 2) + y_n*(NX + 2) + x_e] - rho_bL[jz*(NX + 2)*(NY + 2) + y_n*(NX + 2) + x_e])
				+ xc[8] * (rho_rL[jz*(NX + 2)*(NY + 2) + y_s*(NX + 2) + x_e] - rho_bL[jz*(NX + 2)*(NY + 2) + y_s*(NX + 2) + x_e])
				+ xc[11] * (rho_rL[z_t*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_e] - rho_bL[z_t*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_e])
				+ xc[13] * (rho_rL[z_b*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_e] - rho_bL[z_b*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_e])

				+ xc[2] * (rho_rL[jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_w] - rho_bL[jz*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_w])
				+ xc[9] * (rho_rL[jz*(NX + 2)*(NY + 2) + y_n*(NX + 2) + x_w] - rho_bL[jz*(NX + 2)*(NY + 2) + y_n*(NX + 2) + x_w])
				+ xc[10] * (rho_rL[jz*(NX + 2)*(NY + 2) + y_s*(NX + 2) + x_w] - rho_bL[jz*(NX + 2)*(NY + 2) + y_s*(NX + 2) + x_w])
				+ xc[12] * (rho_rL[z_t*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_w] - rho_bL[z_t*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_w])
				+ xc[14] * (rho_rL[z_b*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_w] - rho_bL[z_b*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_w]);

			FyL[j1]
				= yc[3] * (rho_rL[jz*(NX + 2)*(NY + 2) + y_n*(NX + 2) + jx] - rho_bL[jz*(NX + 2)*(NY + 2) + y_n*(NX + 2) + jx])
				+ yc[7] * (rho_rL[jz*(NX + 2)*(NY + 2) + y_n*(NX + 2) + x_e] - rho_bL[jz*(NX + 2)*(NY + 2) + y_n*(NX + 2) + x_e])
				+ yc[9] * (rho_rL[jz*(NX + 2)*(NY + 2) + y_n*(NX + 2) + x_w] - rho_bL[jz*(NX + 2)*(NY + 2) + y_n*(NX + 2) + x_w])
				+ yc[15] * (rho_rL[z_t*(NX + 2)*(NY + 2) + y_n*(NX + 2) + jx] - rho_bL[z_t*(NX + 2)*(NY + 2) + y_n*(NX + 2) + jx])
				+ yc[16] * (rho_rL[z_b*(NX + 2)*(NY + 2) + y_n*(NX + 2) + jx] - rho_bL[z_b*(NX + 2)*(NY + 2) + y_n*(NX + 2) + jx])

				+ yc[4] * (rho_rL[jz*(NX + 2)*(NY + 2) + y_s*(NX + 2) + jx] - rho_bL[jz*(NX + 2)*(NY + 2) + y_s*(NX + 2) + jx])
				+ yc[8] * (rho_rL[jz*(NX + 2)*(NY + 2) + y_s*(NX + 2) + x_e] - rho_bL[jz*(NX + 2)*(NY + 2) + y_s*(NX + 2) + x_e])
				+ yc[10] * (rho_rL[jz*(NX + 2)*(NY + 2) + y_s*(NX + 2) + x_w] - rho_bL[jz*(NX + 2)*(NY + 2) + y_s*(NX + 2) + x_w])
				+ yc[17] * (rho_rL[z_t*(NX + 2)*(NY + 2) + y_s*(NX + 2) + jx] - rho_bL[z_t*(NX + 2)*(NY + 2) + y_s*(NX + 2) + jx])
				+ yc[18] * (rho_rL[z_b*(NX + 2)*(NY + 2) + y_s*(NX + 2) + jx] - rho_bL[z_b*(NX + 2)*(NY + 2) + y_s*(NX + 2) + jx]);

			FzL[j1]
				= zc[5] * (rho_rL[z_t*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx] - rho_bL[z_t*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx])
				+ zc[11] * (rho_rL[z_t*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_e] - rho_bL[z_t*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_e])
				+ zc[12] * (rho_rL[z_t*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_w] - rho_bL[z_t*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_w])
				+ zc[15] * (rho_rL[z_t*(NX + 2)*(NY + 2) + y_n*(NX + 2) + jx] - rho_bL[z_t*(NX + 2)*(NY + 2) + y_n*(NX + 2) + jx])
				+ zc[17] * (rho_rL[z_t*(NX + 2)*(NY + 2) + y_s*(NX + 2) + jx] - rho_bL[z_t*(NX + 2)*(NY + 2) + y_s*(NX + 2) + jx])

				+ zc[6] * (rho_rL[z_b*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx] - rho_bL[z_b*(NX + 2)*(NY + 2) + jy*(NX + 2) + jx])
				+ zc[13] * (rho_rL[z_b*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_e] - rho_bL[z_b*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_e])
				+ zc[14] * (rho_rL[z_b*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_w] - rho_bL[z_b*(NX + 2)*(NY + 2) + jy*(NX + 2) + x_w])
				+ zc[16] * (rho_rL[z_b*(NX + 2)*(NY + 2) + y_n*(NX + 2) + jx] - rho_bL[z_b*(NX + 2)*(NY + 2) + y_n*(NX + 2) + jx])
				+ zc[18] * (rho_rL[z_b*(NX + 2)*(NY + 2) + y_s*(NX + 2) + jx] - rho_bL[z_b*(NX + 2)*(NY + 2) + y_s*(NX + 2) + jx]);

			fm = sqrt(FxL[j1] * FxL[j1] + FyL[j1] * FyL[j1] + FzL[j1] * FzL[j1]);

			if (fm < 1e-8){
				for (i = 0; i < 19; i++){
					f_rL[j1 + i * LineNum] = rho_rL[j1] / rhoL[j1] * ffL[j1 + i * LineNum];
					f_bL[j1 + i * LineNum] = rho_bL[j1] / rhoL[j1] * ffL[j1 + i * LineNum];
				}
			}

			else{
				i = 0;
				ffL[j1 + i*LineNum] = ffL[j1 + i*LineNum] + A*fm*(-Bi[i]);

				for (i = 1; i < 19; i++){
					cosfai[i] = (xc[i] * FxL[j1] + yc[i] * FyL[j1] + zc[i] * FzL[j1]) / rsq[i] / fm;
					ffL[j1 + i*LineNum] = ffL[j1 + i*LineNum] + A*fm*(t_k[i] * cosfai[i] * cosfai[i] * rsq[i] * rsq[i] - Bi[i]);
				}
				temp = rho_bL[j1] * rho_rL[j1] / rhoL[j1] / rhoL[j1];

				f_rL[j1 + 0 * LineNum] = rho_rL[j1] / rhoL[j1] * ffL[j1 + 0 * LineNum];
				f_bL[j1 + 0 * LineNum] = rho_bL[j1] / rhoL[j1] * ffL[j1 + 0 * LineNum];

				for (i = 1; i < 19; i++){
					feq[i] = t_k[i] * rhoL[j1];
					f_rL[j1 + i * LineNum] = rho_rL[j1] / rhoL[j1] * ffL[j1 + i * LineNum] + beta*temp*feq[i] * cosfai[i];
					f_bL[j1 + i * LineNum] = rho_bL[j1] / rhoL[j1] * ffL[j1 + i * LineNum] - beta*temp*feq[i] * cosfai[i];
				}
			}
		}//if (image == 0)		
	}//for loop	
}

template <typename T>
__global__ void boundaryWall3D_Block(T *f, int *image, int NX, int NY, int NZ_sub)
{
	int jx, jy, jz, j1;
	T temp;
	int LineNum = (NY + 2)*(NX + 2)*(NZ_sub + 2);
	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZ_sub; jz++) {
		j1 = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
		if (image[j1] == 5) {
			temp = f[j1 + 1 * (LineNum)];
			f[j1 + 1 * (LineNum)] = f[j1 + 2 * (LineNum)];
			f[j1 + 2 * (LineNum)] = temp;

			temp = f[j1 + 3 * (LineNum)];
			f[j1 + 3 * (LineNum)] = f[j1 + 4 * (LineNum)];
			f[j1 + 4 * (LineNum)] = temp;

			temp = f[j1 + 5 * (LineNum)];
			f[j1 + 5 * (LineNum)] = f[j1 + 6 * (LineNum)];
			f[j1 + 6 * (LineNum)] = temp;

			temp = f[j1 + 7 * (LineNum)];
			f[j1 + 7 * (LineNum)] = f[j1 + 10 * (LineNum)];
			f[j1 + 10 * (LineNum)] = temp;

			temp = f[j1 + 8 * (LineNum)];
			f[j1 + 8 * (LineNum)] = f[j1 + 9 * (LineNum)];
			f[j1 + 9 * (LineNum)] = temp;

			temp = f[j1 + 11 * (LineNum)];
			f[j1 + 11 * (LineNum)] = f[j1 + 14 * (LineNum)];
			f[j1 + 14 * (LineNum)] = temp;

			temp = f[j1 + 12 * (LineNum)];
			f[j1 + 12 * (LineNum)] = f[j1 + 13 * (LineNum)];
			f[j1 + 13 * (LineNum)] = temp;

			temp = f[j1 + 15 * (LineNum)];
			f[j1 + 15 * (LineNum)] = f[j1 + 18 * (LineNum)];
			f[j1 + 18 * (LineNum)] = temp;

			temp = f[j1 + 16 * (LineNum)];
			f[j1 + 16 * (LineNum)] = f[j1 + 17 * (LineNum)];
			f[j1 + 17 * (LineNum)] = temp;
		}
	}
}

template <typename T>
__global__ void getMacro3DRK_Block(int *image, T *f_rL, T *f_bL, T *uxL, T *uyL, T *uzL, T *ux_oldL, T *uy_oldL, T *uz_oldL,T *rho_rL, T *rho_bL, T *rhoL, T rho_ri, T rho_bi, int NX, int NY, int NZ_sub)
{
	int jx, jy, jz, j_p;	
	int sizeOfCube = (NY + 2)*(NX + 2)*(NZ_sub + 2);

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZ_sub; jz++){
		j_p = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
		if (image[j_p] != 5){

			ux_oldL[j_p] = uxL[j_p];
			uy_oldL[j_p] = uyL[j_p];
			uz_oldL[j_p] = uzL[j_p];

			rho_rL[j_p] = f_rL[j_p]
				+ f_rL[j_p + 1 * sizeOfCube] + f_rL[j_p + 2 * sizeOfCube] + f_rL[j_p + 3 * sizeOfCube] + f_rL[j_p + 4 * sizeOfCube] + f_rL[j_p + 5 * sizeOfCube] + f_rL[j_p + 6 * sizeOfCube]
				+ f_rL[j_p + 7 * sizeOfCube] + f_rL[j_p + 8 * sizeOfCube] + f_rL[j_p + 9 * sizeOfCube] + f_rL[j_p + 10 * sizeOfCube] + f_rL[j_p + 11 * sizeOfCube] + f_rL[j_p + 12 * sizeOfCube]
				+ f_rL[j_p + 13 * sizeOfCube] + f_rL[j_p + 14 * sizeOfCube] + f_rL[j_p + 15 * sizeOfCube] + f_rL[j_p + 16 * sizeOfCube] + f_rL[j_p + 17 * sizeOfCube] + f_rL[j_p + 18 * sizeOfCube];

			rho_bL[j_p] = f_bL[j_p]
				+ f_bL[j_p + 1 * sizeOfCube] + f_bL[j_p + 2 * sizeOfCube] + f_bL[j_p + 3 * sizeOfCube] + f_bL[j_p + 4 * sizeOfCube] + f_bL[j_p + 5 * sizeOfCube] + f_bL[j_p + 6 * sizeOfCube]
				+ f_bL[j_p + 7 * sizeOfCube] + f_bL[j_p + 8 * sizeOfCube] + f_bL[j_p + 9 * sizeOfCube] + f_bL[j_p + 10 * sizeOfCube] + f_bL[j_p + 11 * sizeOfCube] + f_bL[j_p + 12 * sizeOfCube]
				+ f_bL[j_p + 13 * sizeOfCube] + f_bL[j_p + 14 * sizeOfCube] + f_bL[j_p + 15 * sizeOfCube] + f_bL[j_p + 16 * sizeOfCube] + f_bL[j_p + 17 * sizeOfCube] + f_bL[j_p + 18 * sizeOfCube];

			rhoL[j_p] = rho_rL[j_p] + rho_bL[j_p];

			uxL[j_p] = ((f_rL[j_p + 1 * sizeOfCube] - f_rL[j_p + 2 * sizeOfCube] + f_rL[j_p + 7 * sizeOfCube] + f_rL[j_p + 8 * sizeOfCube] - f_rL[j_p + 9 * sizeOfCube]
				- f_rL[j_p + 10 * sizeOfCube] + f_rL[j_p + 11 * sizeOfCube] - f_rL[j_p + 12 * sizeOfCube] + f_rL[j_p + 13 * sizeOfCube] - f_rL[j_p + 14 * sizeOfCube])
				+ (f_bL[j_p + 1 * sizeOfCube] - f_bL[j_p + 2 * sizeOfCube] + f_bL[j_p + 7 * sizeOfCube] + f_bL[j_p + 8 * sizeOfCube] - f_bL[j_p + 9 * sizeOfCube]
					- f_bL[j_p + 10 * sizeOfCube] + f_bL[j_p + 11 * sizeOfCube] - f_bL[j_p + 12 * sizeOfCube] + f_bL[j_p + 13 * sizeOfCube] - f_bL[j_p + 14 * sizeOfCube])) / rhoL[j_p];   // try for(1-19)

			uyL[j_p] = ((f_rL[j_p + 3 * sizeOfCube] - f_rL[j_p + 4 * sizeOfCube] + f_rL[j_p + 7 * sizeOfCube] - f_rL[j_p + 8 * sizeOfCube] + f_rL[j_p + 9 * sizeOfCube]
				- f_rL[j_p + 10 * sizeOfCube] + f_rL[j_p + 15 * sizeOfCube] + f_rL[j_p + 16 * sizeOfCube] - f_rL[j_p + 17 * sizeOfCube] - f_rL[j_p + 18 * sizeOfCube])
				+ (f_bL[j_p + 3 * sizeOfCube] - f_bL[j_p + 4 * sizeOfCube] + f_bL[j_p + 7 * sizeOfCube] - f_bL[j_p + 8 * sizeOfCube] + f_bL[j_p + 9 * sizeOfCube]
					- f_bL[j_p + 10 * sizeOfCube] + f_bL[j_p + 15 * sizeOfCube] + f_bL[j_p + 16 * sizeOfCube] - f_bL[j_p + 17 * sizeOfCube] - f_bL[j_p + 18 * sizeOfCube])) / rhoL[j_p];

			uzL[j_p] = ((f_rL[j_p + 5 * sizeOfCube] - f_rL[j_p + 6 * sizeOfCube] + f_rL[j_p + 11 * sizeOfCube] + f_rL[j_p + 12 * sizeOfCube] - f_rL[j_p + 13 * sizeOfCube]
				- f_rL[j_p + 14 * sizeOfCube] + f_rL[j_p + 15 * sizeOfCube] - f_rL[j_p + 16 * sizeOfCube] + f_rL[j_p + 17 * sizeOfCube] - f_rL[j_p + 18 * sizeOfCube])
				+ (f_bL[j_p + 5 * sizeOfCube] - f_bL[j_p + 6 * sizeOfCube] + f_bL[j_p + 11 * sizeOfCube] + f_bL[j_p + 12 * sizeOfCube] - f_bL[j_p + 13 * sizeOfCube]
					- f_bL[j_p + 14 * sizeOfCube] + f_bL[j_p + 15 * sizeOfCube] - f_bL[j_p + 16 * sizeOfCube] + f_bL[j_p + 17 * sizeOfCube] - f_bL[j_p + 18 * sizeOfCube])) / rhoL[j_p];
		}
		else{
			//rho_rL[j_p] = 0.01*rho_ri;
			//rho_bL[j_p] = 1.0*rho_bi;			 
			//rhoL[j_p] = rho_rL[j_p] + rho_bL[j_p];
			uxL[j_p] = 0.0;
			uyL[j_p] = 0.0;
			uzL[j_p] = 0.0;
		}
	}
}

template <typename T>
void setPhaseField3D(int *image, T *phaseField, T *rho_r, T *rho_b, int NX, int NY, int NZ)
{
	int x, y, z, j1;
	for (z = 1; z < NZ + 1; z++)
	for (y = 1; y < NY + 1; y++)
	for (x = 1; x < NX + 1; x++){
		j1 = z * (NX + 2)*(NY + 2) + y * (NX + 2) + x;
		if (image[j1] != 5)	{	phaseField[j1] = (rho_r[j1] - rho_b[j1]) / (rho_r[j1] + rho_b[j1]);	}
		else	{	phaseField[j1] = 0;	}
	}
}

template <typename T>
__global__ void CollisionRK3D_Blocktest(int *imageL, T *uxLocal, T *uyLocal, T *uzLocal, T *rhoLocal, T *rho_rLocal, T *rho_bLocal,
	T *ffLocal, T *f_rLocal, T *f_bLocal,
	T taur, T taub, T *t_k, T *ex, T *ey, T *ez,
	T alfa_r, T alfa_b, T c_squ, T c_squ_r, T c_squ_b, T *f_body, int NX, int NY, int NZ_sub,T Fdex) {

	int jx, jy, jz, j1, i;
	double fequ_r[19], fequ_b[19];
	int LineNum = (NY + 2)*(NX + 2)*(NZ_sub + 2);
	double   fei, tau;
	double alfa1, beta1, kappa1, delta1, eta1, kxi1;
	double u_n[19], u_squ;
	int k;

	delta1 = 0.98;
	alfa1 = 2.0 * taur*taub / (taur + taub);
	beta1 = 2.0 * (taur - alfa1) / delta1;
	kappa1 = -beta1 / (2.0 * delta1);
	eta1 = 2.0*(alfa1 - taub) / delta1;
	kxi1 = eta1 / (2.0 * delta1);

	jx = blockDim.x*blockIdx.x + threadIdx.x + 1;
	jy = blockDim.y*blockIdx.y + threadIdx.y + 1;

	for (jz = 1; jz <= NZ_sub; jz++) {
		j1 = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
		if (imageL[j1] == 0) {
			u_squ = uxLocal[j1] * uxLocal[j1] + uyLocal[j1] * uyLocal[j1] + uzLocal[j1] * uzLocal[j1];
			u_n[0] = 0.0;
			for (k = 1; k < 19; k++) { u_n[k] = ex[k] * uxLocal[j1] + ey[k] * uyLocal[j1] + ez[k] * uzLocal[j1]; }

			fequ_r[0] = t_k[0] * rho_rLocal[j1] * (u_n[0] * 3.0 + u_n[0] * u_n[0] * 4.5 - u_squ * 1.5) + rho_rLocal[j1] * alfa_r;
			for (k = 1; k < 7; k++) { fequ_r[k] = t_k[k] * rho_rLocal[j1] * (u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ * 1.5) + rho_rLocal[j1] * (1.0 - alfa_r) / 12.0; }
			for (k = 7; k < 19; k++) { fequ_r[k] = t_k[k] * rho_rLocal[j1] * (u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ * 1.5) + rho_rLocal[j1] * (1.0 - alfa_r) / 24.0; }

			fequ_b[0] = t_k[0] * rho_bLocal[j1] * (u_n[0] * 3.0 + u_n[0] * u_n[0] * 4.5 - u_squ * 1.5) + rho_bLocal[j1] * alfa_b;
			for (k = 1; k < 7; k++) { fequ_b[k] = t_k[k] * rho_bLocal[j1] * (u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ * 1.5) + rho_bLocal[j1] * (1.0 - alfa_b) / 12.0; }
			for (k = 7; k < 19; k++) { fequ_b[k] = t_k[k] * rho_bLocal[j1] * (u_n[k] * 3.0 + u_n[k] * u_n[k] * 4.5 - u_squ * 1.5) + rho_bLocal[j1] * (1.0 - alfa_b) / 24.0; }
			////*******************compute Fequ end

			fei = (rho_rLocal[j1] - rho_bLocal[j1]) / (rho_rLocal[j1] + rho_bLocal[j1]);

			if (fei > delta1)			         	tau = taur;
			else if (fei > 0 && fei <= delta1)		tau = alfa1 + beta1 * fei + kappa1 * fei*fei;
			else if (fei > -delta1 && fei <= 0)	tau = alfa1 + eta1 * fei + kxi1 * fei*fei;
			else if (fei < -delta1)			    	tau = taub;

            double dex;
			for (i = 0; i < 19; i++) {				
				if (fei> 0.0)
					dex = 1.0*Fdex;
				else
					dex = 0.001*Fdex;
				f_rLocal[j1 + i * LineNum] = fequ_r[i] + (1 - 1 / tau)*(f_rLocal[j1 + i * LineNum] - fequ_r[i]);
				f_bLocal[j1 + i * LineNum] = fequ_b[i] + (1 - 1 / tau)*(f_bLocal[j1 + i * LineNum] - fequ_b[i]);
				ffLocal[j1 + i * LineNum] = f_rLocal[j1 + i * LineNum] + f_bLocal[j1 + i * LineNum] + t_k[i] * ex[i] * f_body[0] / c_squ + dex* t_k[i] * ey[i] * f_body[1] / c_squ + dex * t_k[i] * ez[i] * f_body[2] / c_squ;

			}
		}
	} //end for jz
}


////////////////////////////////////////////
#endif