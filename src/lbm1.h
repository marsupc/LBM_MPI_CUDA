typedef  double  DOUBLE; 
typedef  float   FLOAT;

#define TAG_image     1
#define TAG_f_r     2
#define TAG_f_b     3
#define TAG_rho_r     4 
#define TAG_rho_b     5
#define TAG_phaseField     6
#define TAG_ux     7
#define TAG_uy     8
#define TAG_uz     9
#define TAG_rho     10

 
#define TAG_det     12
#define TAG_fluxZPositive     13
#define TAG_fluxZNegative     14
#define TAG_fluxZPositiveCap     15
#define TAG_fluxZNegativeCap     16


#define TAG_TT     101


#include "dataMemoryControl/directoryManage.h"
#include "dataMemoryControl/writeData.h"
#include "dataMemoryControl/dataExchange.h"
#include "dataMemoryControl/variableAllocate.h"
#include "block/MPICUDAInitialize.h"
#include "latticeBoltzmann/latticeBoltzmannBlock.h"
 



 