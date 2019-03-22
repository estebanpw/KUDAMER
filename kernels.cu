#include "kernels.cuh"



__global__ void kernel_index(unsigned long long int * table, const char * sequence) {
		

	unsigned long long int k, j, hash;
	unsigned char multiplier, val;

	__shared__ unsigned long long int seq_shared[20]; // 160 bytes of sequence divided by 8 bytes per uint64_t (should be 156)
	//												// Last thread (31-th) accesses bytes 124 to 159
	
	
	if(threadIdx.x < 20){
		seq_shared[threadIdx.x] = ((unsigned long long int *) sequence)[threadIdx.x + blockIdx.x * 20];
	}
	__syncthreads();

	char * sweet_pointer = (char *) seq_shared;
	
	for(j=0; j<4; j++){ // the indexing makes [0,1,2,3,4...] -> [0,4,8,12,16] so we need to fit in the [1,2,3], etc.
		char c;
		hash = 0;
		for(k=0; k<32; k++){
			//__syncthreads();
			c = sweet_pointer[threadIdx.x * 4 + k + j];
			// This is better because it uses online 10 registers and the branch divergence is SO low (4 ops...)
			if(c == 'A') hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) 0;
			if(c == 'C') hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) 1;
			if(c == 'G') hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) 2;
			if(c == 'T') hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) 3;

			/*
			val = (unsigned char) sweet_pointer[threadIdx.x * 4 + k];
			multiplier = (val & 6) >> 1;
			hash += (1 << (2*k)) * (unsigned long long int) multiplier;
			*/

		}

		table[threadIdx.x + blockDim.x * j + 224 * blockDim.x] = hash;
	}
}