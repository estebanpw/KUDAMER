#include "structs.h"


//__constant__ unsigned long cs_kmers_in_work_item;

__global__ void kernel_register_less_synchro(unsigned long long int * table, const char * sequence);

__global__ void kernel_register(unsigned long long int * table, const char * sequence);

__global__ void kernel_register_no_synchro_exp(unsigned long long int * table, const char * sequence);

__global__ void kernel_register_fast_hash_no_synchro_exp(unsigned long long int * table, const char * sequence);

__global__ void kernel_register_fast_hash_no_synchro_exp_64(unsigned long long int * table, const char * sequence);

__global__ void kernel_index32(unsigned long long int * table, const char * sequence);

__global__ void kernel_index64(unsigned long long int * table, const char * sequence);

__global__ void kernel_index_global32(unsigned long long int * table, const char * sequence);

__global__ void kernel_index_global64(unsigned long long int * table, const char * sequence);

__global__ void kernel_index_global_fast_hash(unsigned long long int * table, const char * sequence);

__global__ void kernel_index_global_fast_hash_on_shared(unsigned long long int * table, const char * sequence);

__global__ void kernel_index_global_coalesced(unsigned long long int * table, const char * sequence);

__global__ void kernel_index_global_any_assembled(unsigned long long int * table, const char * sequence);

__global__ void kernel_index_global_any_simplest(unsigned long long int * table, const char * sequence);

__global__ void some_float_experiment(unsigned long long int * table, const char * sequence);
