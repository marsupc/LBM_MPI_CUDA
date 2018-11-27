/*file statement

//writeData_ArraybyImage_vtk(tStep,image,phaseField,NX,NY,NZ, (char *)"./testRK/phaseField_test", (char *) " asdf", 1.0);
//writeData_ArraybyImage_vtk(tStep,image,phaseField,NX,NY,NZ, (char*)phaseFieldName.c_str(), (char *) " asdf", 1.0);

*/

#ifndef writeData
#define writeData

#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <iostream>   
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

template <typename T >
void   writeData_ArraybyImage_vtk(int tStep, 
	int    *image, 
	T    *vt,               
	int      NX,    	int      NY,      	int      NZ,  /*dimension of array*/
	char *filename, char *filetitle, double delta_x)           /* file parameters*/   // this is to replace  outputUVTK_singlevalue
{
	int   jx, jy, jz, j1;
	ostringstream name;
	name << filename << "_" << tStep << "ts" << ".vtk";
	ofstream out(name.str().c_str());

	out << "# vtk DataFile Version 2.0 " << endl;
	out << "LBM by Guo " << endl;
	out << "ASCII " << endl << endl;

	out << "DATASET STRUCTURED_POINTS " << endl;
	out << "DIMENSIONS  " << NX << " " << NY << " " << NZ << endl;
	out << "ORIGIN    " << 0 << " " << 0 << " " << 0 << endl;
	out << "SPACING    " << delta_x << " " << delta_x << " " << delta_x << endl << endl;

	out << "POINT_DATA    " << NX * NY*NZ << endl << endl;

	out << "SCALARS tv double" << endl;
	out << "LOOKUP_TABLE default" << endl;

	for (jz = 1; jz <= NZ; jz++)
	for (jy = 1; jy <= NY; jy++)
	for (jx = 1; jx <= NX; jx++)
		{
			j1 = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
			if (image[j1] != 5)	{	out << vt[j1] << endl;	}
			else				{	out << 0 << endl;		}
		}
	out.close();
}

template <typename T >
void   writeDataArray_vtk(T   *image,
	int      NX,         /* x-dimension size                          */
	int      NY,       /* y-dimension size                          */
	int      NZ,
	char *filename, double delta_x){
	int   jx, jy, jz, j1;
	ostringstream name;
	name << filename << ".vtk";
	ofstream out(name.str().c_str());

	out << "# vtk DataFile Version 2.0 " << endl;
	out << "LBM by Guo " << endl;
	out << "ASCII " << endl << endl;

	out << "DATASET STRUCTURED_POINTS " << endl;
	out << "DIMENSIONS  " << NX << " " << NY << " " << NZ << endl;
	out << "ORIGIN    " << 0 << " " << 0 << " " << 0 << endl;
	out << "SPACING    " << delta_x << " " << delta_x << " " << delta_x << endl << endl;

	out << "POINT_DATA    " << NX * NY*NZ << endl << endl;

	out << "SCALARS image int" << endl;
	out << "LOOKUP_TABLE default" << endl;

	for (jz = 1; jz <= NZ; jz++)
		for (jy = 1; jy <= NY; jy++)
			for (jx = 1; jx <= NX; jx++){
				j1 = jz * (NX + 2)*(NY + 2) + jy * (NX + 2) + jx;
				out << image[j1] << endl;
			}
	out.close();
}




#endif