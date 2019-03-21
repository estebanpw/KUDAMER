#include "structs.h"


//__constant__ unsigned long cs_kmers_in_work_item;



__global__ void kernel_index(unsigned long long int * table, const char * sequence);

