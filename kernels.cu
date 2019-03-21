#include "kernels.cuh"



__global__ void kernel_index(unsigned long long int * table, const char * sequence) {
	// Get the index of the current element to be processed

	// gridDim (x,y,z) = (work items in X, Y, Z)
	// blockDim (x,y,z) = (work group size in X, Y, Z)
	// threadIdx (x,y,z) = (local id of work item in X, Y, Z)
	
	//unsigned long long int global_id = threadIdx.x * blockIdx.x; // + blockIdx.x * blockDim.x;
	unsigned long long int k, hash;
	unsigned char multiplier, val;

	__shared__ unsigned long long int seq_shared[20]; // 160 bytes of sequence divided by 8 bytes per uint64_t (should be 156)
	
	if(threadIdx.x < 20){
		seq_shared[threadIdx.x] = ((unsigned long long int *) sequence)[threadIdx.x + blockIdx.x * 20];
	}
	__syncthreads();

	char * sweet_pointer = (char *) seq_shared;
	

	char c;
	for(k=0; k<32; k++){
		c = sweet_pointer[threadIdx.x * 4 + k];
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

	table[threadIdx.x + blockDim.x * blockIdx.x] = hash;
}