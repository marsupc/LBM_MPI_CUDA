/*file statement



*/

#ifndef directoryManage
#define directoryManage 

#include <sys/stat.h> 

template <typename T>
void   createDirectory( T* directName )   {
	mkdir(directName, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);}



#endif