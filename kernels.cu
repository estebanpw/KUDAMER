#include "kernels.cuh"
#define WARP_SIZE 32
#define KMER_SIZE 32
#define KMERS_PER_THREAD 3
#define BYTES_PER_REGISTER 4

// This kernel uses the register memory model

__constant__ uint64_t pow4[33]={1L, 4L, 16L, 64L, 256L, 1024L, 4096L, 16384L, 65536L,
    262144L, 1048576L, 4194304L, 16777216L, 67108864L, 268435456L, 1073741824L, 4294967296L,
    17179869184L, 68719476736L, 274877906944L, 1099511627776L, 4398046511104L, 17592186044416L,
    70368744177664L, 281474976710656L, 1125899906842624L, 4503599627370496L, 18014398509481984L,
	72057594037927936L, 288230376151711744L, 1152921504606846976L, 4611686018427387904L};
	

__constant__ uint64_t pow4_G[33]={2*1L, 2*4L, 2*16L, 2*64L, 2*256L, 2*1024L, 2*4096L, 2*16384L, 2*65536L,
    (uint64_t)2*262144L, (uint64_t)2*1048576L,(uint64_t)2*4194304L, (uint64_t)2*16777216L, (uint64_t)2*67108864L, (uint64_t)2*268435456L, (uint64_t)2*1073741824L, (uint64_t)2*4294967296L,
    (uint64_t)2*17179869184L, (uint64_t)2*68719476736L, (uint64_t)2*274877906944L, (uint64_t)2*1099511627776L, (uint64_t)2*4398046511104L, (uint64_t)2*17592186044416L,
    (uint64_t)2*70368744177664L, (uint64_t)2*281474976710656L, (uint64_t)2*1125899906842624L, (uint64_t)2*4503599627370496L, (uint64_t)2*18014398509481984L,
    (uint64_t)2*72057594037927936L, (uint64_t) 2*288230376151711744L, (uint64_t) 2*1152921504606846976L, (uint64_t) 2*4611686018427387904L};

__constant__ uint64_t pow4_T[33]={3*1L, 3*4L, 3*16L, 3*64L, 3*256L, 3*1024L, 3*4096L, 3*16384L, 3*65536L,
    (uint64_t)3*262144L, (uint64_t) 3*1048576L, (uint64_t)3*4194304L, (uint64_t)3*16777216L, (uint64_t)3*67108864L, (uint64_t)3*268435456L, (uint64_t)3*1073741824L, (uint64_t)3*4294967296L,
    (uint64_t)3*17179869184L, (uint64_t)3*68719476736L, (uint64_t)3*274877906944L, (uint64_t)3*1099511627776L, (uint64_t)3*4398046511104L, (uint64_t)3*17592186044416L,
    (uint64_t)3*70368744177664L, (uint64_t)3*281474976710656L, (uint64_t)3*1125899906842624L, (uint64_t)3*4503599627370496L, (uint64_t)3*18014398509481984L,
    (uint64_t)3*72057594037927936L, (uint64_t) 3*288230376151711744L, (uint64_t) 3*1152921504606846976L, (uint64_t) 3*4611686018427387904L};

typedef struct{
	unsigned char nucl[4];
} four_nucl;

__global__ void kernel_register(unsigned long long int * table, const char * sequence) {
	
	int i, k;
	unsigned long long int hash = 0;

	// Notice you need not to synchronize this load as its intra warp and the number of threads must be 32
	unsigned int value = ((unsigned int *)sequence)[threadIdx.x + blockIdx.x * 24 ]; // 24*4 = 96 are the bytes corresponding to the 96 kmers used from the 128 byte transaction
	unsigned int temp_value; 
	char byte;
	unsigned long long int bad;
	

	for(i=0; i<3; i++){

		int kmer_start = threadIdx.x * KMERS_PER_THREAD + i;
		int int_pos = kmer_start / BYTES_PER_REGISTER;
		hash = 0;
		bad = 0xFFFFFFFFFFFFFFFF;
		
		for(k=0; k<KMER_SIZE; k++){
			
			temp_value = __shfl_sync(0xffffffff, value, int_pos);
			
			//printf("[%d, %d] ACcessing at kmerstart: %d on int %d on byte %d\n", threadIdx.x, blockIdx.x, kmer_start, int_pos, kmer_start % BYTES_PER_REGISTER);
			
			//byte = ((char *) temp_value)[kmer_start % BYTES_PER_REGISTER];
			//if(threadIdx.x == 0 && blockIdx.x == 0) printf("[%d, %d] Value= %d: \n", threadIdx.x, blockIdx.x, temp_value);
			byte = (char) (temp_value >> (kmer_start % BYTES_PER_REGISTER)*8);
			
			//if(threadIdx.x == 0 && blockIdx.x == 0) printf("[th:%d, bl:%d, i:%d, k:%d] My bytes: %c: \n", threadIdx.x, blockIdx.x, i, k, byte);
			
			
			++kmer_start;
			int_pos = kmer_start / BYTES_PER_REGISTER;

			/*
			unsigned char val = (unsigned char) byte;
			unsigned char multiplier = (val & 6) >> 1;
			hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) multiplier;
			//checker = checker | (val & (unsigned char) 8);
			if(byte == 'N') bad = 0;
			*/
			
			
			/*
			if(byte == 'A') hash += 0;
			if(byte == 'C') hash += pow4[k];
			if(byte == 'G') hash += pow4_G[k];
			if(byte == 'T') hash += pow4_T[k];
			if(byte == 'N') bad = 0;
			*/
			
			

			if(byte == 'A') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 0;
			if(byte == 'C') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 1;
			if(byte == 'G') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 2;
			if(byte == 'T') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 3;
			if(byte == 'N') bad = 0;
			

		}
		//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & bad;
		table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & bad;
		//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & (~((checker >> 3) * 0xFFFFFFFFFFFFFFFF));
		//if(checker == 0) table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash;
		
	}
}

__global__ void kernel_register_less_synchro(unsigned long long int * table, const char * sequence) {
	
	int i, k;
	unsigned long long int hash = 0;

	unsigned int value = ((unsigned int *)sequence)[threadIdx.x + blockIdx.x * 24 ]; // 24*4 = 96 are the bytes corresponding to the 96 kmers used from the 128 byte transaction
	unsigned int temp_value;
	char byte;
	unsigned long long int bad;
	

	for(i=0; i<3; i++){

		int kmer_start = threadIdx.x * KMERS_PER_THREAD + i;
		int int_pos = kmer_start / BYTES_PER_REGISTER;
		hash = 0;
		bad = 0xFFFFFFFFFFFFFFFF;
		
		for(k=0; k<KMER_SIZE; k++){
			
			temp_value = __shfl_sync((1 << int_pos) || (1 << threadIdx.x), value, int_pos);
			//temp_value = __shfl(value, int_pos, 32);
			
			//printf("[%d, %d] ACcessing at kmerstart: %d on int %d on byte %d\n", threadIdx.x, blockIdx.x, kmer_start, int_pos, kmer_start % BYTES_PER_REGISTER);
			
			//byte = ((char *) temp_value)[kmer_start % BYTES_PER_REGISTER];
			//if(threadIdx.x == 0 && blockIdx.x == 0) printf("[%d, %d] Value= %d: \n", threadIdx.x, blockIdx.x, temp_value);
			byte = (char) (temp_value >> (kmer_start % BYTES_PER_REGISTER)*8);
			
			//if(threadIdx.x == 0 && blockIdx.x == 0) printf("[th:%d, bl:%d, i:%d, k:%d] My bytes: %c: \n", threadIdx.x, blockIdx.x, i, k, byte);
			
			
			++kmer_start;
			int_pos = kmer_start / BYTES_PER_REGISTER;

			/*
			
			unsigned char val = (unsigned char) byte;
			unsigned char multiplier = (val & 6) >> 1;
			hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) multiplier;
			//checker = checker | (val & (unsigned char) 8);
			if(byte == 'N') bad = 0;
			
			*/

			/*
			if(byte == 'A') hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) 0;
			if(byte == 'C') hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) 1;
			if(byte == 'G') hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) 2;
			if(byte == 'T') hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) 3;
			if(byte == 'N') bad = 0;
			*/

			
			if(byte == 'A') hash += 0;
			if(byte == 'C') hash += pow4[k];
			if(byte == 'G') hash += pow4_G[k];
			if(byte == 'T') hash += pow4_T[k];
			if(byte == 'N') bad = 0;
			
			

		}
		//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & bad;
		table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & bad;
		//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & (~((checker >> 3) * 0xFFFFFFFFFFFFFFFF));
		//if(checker == 0) table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash;
		
	}
}

__global__ void kernel_register_no_synchro_exp(unsigned long long int * table, const char * sequence) {
	
	int i, k;
	unsigned long long int hash = 0;

	unsigned int value = ((unsigned int *)sequence)[threadIdx.x + blockIdx.x * 24 ]; // 24*4 = 96 are the bytes corresponding to the 96 kmers used from the 128 byte transaction
	unsigned int temp_value;
	char byte;
	unsigned long long int bad;
	

	for(i=0; i<3; i++){

		int kmer_start = threadIdx.x * KMERS_PER_THREAD + i;
		int int_pos = kmer_start / BYTES_PER_REGISTER;
		hash = 0;
		bad = 0xFFFFFFFFFFFFFFFF;
		
		for(k=0; k<KMER_SIZE; k++){
			
			temp_value = __shfl_sync((1 << int_pos) || (1 << threadIdx.x), value, int_pos);
			//temp_value = value;
			
			//printf("[%d, %d] ACcessing at kmerstart: %d on int %d on byte %d\n", threadIdx.x, blockIdx.x, kmer_start, int_pos, kmer_start % BYTES_PER_REGISTER);
			
			//byte = ((char *) temp_value)[kmer_start % BYTES_PER_REGISTER];
			//if(threadIdx.x == 0 && blockIdx.x == 0) printf("[%d, %d] Value= %d: \n", threadIdx.x, blockIdx.x, temp_value);
			byte = (char) (temp_value >> (kmer_start % BYTES_PER_REGISTER)*8);
			
			//if(threadIdx.x == 0 && blockIdx.x == 0) printf("[th:%d, bl:%d, i:%d, k:%d] My bytes: %c: \n", threadIdx.x, blockIdx.x, i, k, byte);
			
			
			++kmer_start;
			int_pos = kmer_start / BYTES_PER_REGISTER;

			/*
			unsigned char val = (unsigned char) byte;
			unsigned char multiplier = (val & 6) >> 1;
			hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) multiplier;
			//checker = checker | (val & (unsigned char) 8);
			if(byte == 'N') bad = 0;
			*/
			
			
			/*
			if(byte == 'A') hash += 0;
			if(byte == 'C') hash += pow4[k];
			if(byte == 'G') hash += pow4_G[k];
			if(byte == 'T') hash += pow4_T[k];
			if(byte == 'N') bad = 0;
			*/
			
			
			if(byte == 'A') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 0;
			if(byte == 'C') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 1;
			if(byte == 'G') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 2;
			if(byte == 'T') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 3;
			if(byte == 'N') bad = 0;
			

		}
		//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & bad;
		table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & bad;
		//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & (~((checker >> 3) * 0xFFFFFFFFFFFFFFFF));
		//if(checker == 0) table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash;
		
	}
}


// This kernel is to be executed with 32 threads per block 

__global__ void kernel_index32(unsigned long long int * table, const char * sequence) {
		

	unsigned long long int k, j, hash;
	unsigned char multiplier, val;

	__shared__ unsigned long long int seq_shared[20]; // 160 bytes of sequence divided by 8 bytes per uint64_t (should be 156)
	//												  // Last thread (31-th) accesses bytes 124 to 159
	
	
	
	if(threadIdx.x < 20){
		seq_shared[threadIdx.x] = ((unsigned long long int *) sequence)[threadIdx.x + blockIdx.x * 16]; // I thought it was 20 but it is 16. 20 bytes are needed but the ones between 16 and 20 are only used for the last kmer - so many are missing
	}
	unsigned long long int fixed_pos = threadIdx.x + 128 * blockIdx.x;
	//__syncthreads(); //not needed since this is for 32 threads per block

	char * sweet_pointer = (char *) seq_shared;
	
	
	for(j=0; j<4; j++){ // the indexing makes [0,1,2,3,4...] -> [0,4,8,12,16] so we need to fit in the [1,2,3], etc.
		char c;
		unsigned long long int bad = 0xFFFFFFFFFFFFFFFF;
		unsigned long long int pos = fixed_pos + blockDim.x * j;
		hash = 0;
		for(k=0; k<32; k++){
			
			c = sweet_pointer[threadIdx.x * 4 + k + j];
			// This is better because it uses online 10 registers and the branch divergence is SO low (4 ops...)
			
			/*
			if(c == 'A') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 0;
			if(c == 'C') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 1;
			if(c == 'G') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 2;
			if(c == 'T') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 3;
			if(c == 'N') bad = 0;
			*/
			
			/*
			if(c == 'A') hash += 0;
			if(c == 'C') hash += pow4[k];
			if(c == 'G') hash += pow4_G[k];
			if(c == 'T') hash += pow4_T[k];
			if(c == 'N') bad = 0;
			*/
			
			

			
			val = (unsigned char) sweet_pointer[threadIdx.x * 4 + k];
			multiplier = (val & 6) >> 1;
			hash += (1 << (2*k)) * (unsigned long long int) multiplier;
			if(c == 'N') bad = 0;
			

		}
		//if(bad == 1) hash = 0;
		table[pos] = hash & bad;
		//table[threadIdx.x + blockDim.x * j + 224 * blockIdx.x] = hash;
	}
}



// This kernel is to be executed with 64 threads per block 
// Remember to set the block size to 64 !!!!!!!!

__global__ void kernel_index64(unsigned long long int * table, const char * sequence) {
		

	unsigned long long int k, j, hash;
	unsigned char multiplier, val;

	__shared__ unsigned long long int seq_shared[40]; // 320 bytes of sequence divided by 8 bytes per uint64_t (should be 156)
	//												// Last thread (63-th) accesses bytes 63*4 = 252, 252+3 = 255 - 287. 287/8 = ~36
	
	
	if(threadIdx.x < 40){
		seq_shared[threadIdx.x] = ((unsigned long long int *) sequence)[threadIdx.x + blockIdx.x * 32]; // I thought it was 20 but it is 16. 20 bytes are needed but the ones between 36 and 40 are only used for the last kmer - so many are missing
	}
	__syncthreads();

	char * sweet_pointer = (char *) seq_shared;
	
	
	for(j=0; j<4; j++){ // the indexing makes [0,1,2,3,4...] -> [0,4,8,12,16] so we need to fit in the [1,2,3], etc.
		char c;
		unsigned long long int bad = 0xFFFFFFFFFFFFFFFF;
		unsigned long long int pos = threadIdx.x + blockDim.x * j + 256 * blockIdx.x;
		hash = 0;
		for(k=0; k<32; k++){
			
			c = sweet_pointer[threadIdx.x * 4 + k + j];
			// This is better because it uses online 10 registers and the branch divergence is SO low (4 ops...)
			
			/*
			if(c == 'A') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 0;
			if(c == 'C') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 1;
			if(c == 'G') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 2;
			if(c == 'T') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 3;
			if(c == 'N') bad = 0;
			*/
			
			
			/*
			if(c == 'A') hash += ((unsigned long long int)1 << (2*(31-k))) * (unsigned long long int) 0;
			if(c == 'C') hash += ((unsigned long long int)1 << (2*(31-k))) * (unsigned long long int) 1;
			if(c == 'G') hash += ((unsigned long long int)1 << (2*(31-k))) * (unsigned long long int) 2;
			if(c == 'T') hash += ((unsigned long long int)1 << (2*(31-k))) * (unsigned long long int) 3;
			*/

			/*
			if(c == 'A') hash += 0;
			if(c == 'C') hash += pow4[k];
			if(c == 'G') hash += pow4_G[k];
			if(c == 'T') hash += pow4_T[k];
			if(c == 'N') bad = 0;
			*/
			

			
			val = (unsigned char) c;
			multiplier = (val & 6) >> 1;
			hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) multiplier;
			if(c == 'N') bad = 0;
			
			

		}
		//if(bad == 1) hash = 0;
		table[pos] = hash & bad;
		//table[threadIdx.x + blockDim.x * j + 224 * blockIdx.x] = hash;
	}
}


__global__ void kernel_index_global32(unsigned long long int * table, const char * sequence) {
		

	unsigned long long int k, j, hash = 0;
		
	unsigned long long int bad = 0xFFFFFFFFFFFFFFFF;

	for(k=0; k<32; k++){
	//for(k=0; k<1; k++){

		char c = sequence[threadIdx.x + k + blockIdx.x * blockDim.x];
		/*
		if(c == 'A') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 0;
		if(c == 'C') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 1;
		if(c == 'G') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 2;
		if(c == 'T') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 3;
		if(c == 'N') bad = 0;
		*/
		/*
		if(c == 'A') hash += 0;
		if(c == 'C') hash += pow4[k];
		if(c == 'G') hash += pow4_G[k];
		if(c == 'T') hash += pow4_T[k];
		if(c == 'N') bad = 0;
		*/

		
		unsigned char val = (unsigned char) c;
		unsigned char multiplier = (val & 6) >> 1;
		hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) multiplier;
		//checker = checker | (val & (unsigned char) 8);
		if(c == 'N') bad = 0;
		
	}

	table[threadIdx.x + blockIdx.x * blockDim.x] = hash & bad;
}

__global__ void kernel_index_global64(unsigned long long int * table, const char * sequence) {
		

	unsigned long long int k, j, hash = 0;
		
	unsigned long long int bad = 0xFFFFFFFFFFFFFFFF;

	for(k=0; k<32; k++){

		char c = sequence[threadIdx.x + k + blockIdx.x * blockDim.x];
		
		
		if(c == 'A') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 0;
		if(c == 'C') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 1;
		if(c == 'G') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 2;
		if(c == 'T') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 3;
		if(c == 'N') bad = 0;
		
		
		/*
		if(c == 'A') hash += 0;
		if(c == 'C') hash += pow4[k];
		if(c == 'G') hash += pow4_G[k];
		if(c == 'T') hash += pow4_T[k];
		if(c == 'N') bad = 0;
		*/
		

		/*
		unsigned char val = (unsigned char) c;
		unsigned char multiplier = (val & 6) >> 1;
		hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) multiplier;
		//checker = checker | (val & (unsigned char) 8);
		if(c == 'N') bad = 0;
		*/
	}

	table[threadIdx.x + blockIdx.x * blockDim.x] = hash & bad;
	//table[threadIdx.x] = hash & bad;
}

__global__ void kernel_index_global_fast_hash(unsigned long long int * table, const char * sequence) {

	unsigned long long int k, j, hash = 0, my_byte;
	unsigned long long int bad = 0xFFFFFFFFFFFFFFFF;

	for(k=0; k<32; k++){

		my_byte = threadIdx.x * 8 + k + blockIdx.x * blockDim.x * 8; // the 8 comes from each thread processing 8 kmers
		char c = sequence[my_byte];

		/*
		if(c == 'A') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 0;
		if(c == 'C') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 1;
		if(c == 'G') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 2;
		if(c == 'T') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 3;
		if(c == 'N') bad = 0;
		*/
		

		/*
		if(c == 'A') hash += 0;
		if(c == 'C') hash += pow4[k];
		if(c == 'G') hash += pow4_G[k];
		if(c == 'T') hash += pow4_T[k];
		if(c == 'N') bad = 0;
		*/
		
		unsigned char val = (unsigned char) c;
		unsigned char multiplier = (val & 6) >> 1;
		hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) multiplier;
		//checker = checker | (val & (unsigned char) 8);
		if(c == 'N') bad = 0;
		
		
	}

	table[threadIdx.x + blockIdx.x * blockDim.x*8] = hash & bad;

	for(k=1; k<8; k++){
		char next_nucl = sequence[++my_byte];
		bad = 0xFFFFFFFFFFFFFFFF;

		if(next_nucl == 'A') hash +=  4 * (hash - ((unsigned long long int)1 << (62)) * 0);
		if(next_nucl == 'C') hash +=  4 * ((hash - ((unsigned long long int)1 << (62)))) + 1;
		if(next_nucl == 'G') hash +=  4 * ((hash - ((unsigned long long int)1 << (62)))) + 2;
		if(next_nucl == 'T') hash +=  4 * ((hash - ((unsigned long long int)1 << (62)))) + 3;
		if(next_nucl == 'N') bad = 0;

		table[threadIdx.x + blockIdx.x * blockDim.x*8 + blockDim.x * k] = hash & bad;
	}
}


__global__ void kernel_index_global_fast_hash_on_shared(unsigned long long int * table, const char * sequence) {

	unsigned long long int hash = 0; 
	int my_byte, k;
	unsigned long long int bad = 0xFFFFFFFFFFFFFFFF;

	//if(threadIdx.x == 1 && blockIdx.x == 0) printf("ssize: %d\n", sizeof(unsigned long long int));

	
	//__shared__ unsigned long long int seq_shared[36]; // 288 bytes of sequence divided by 8 bytes per uint64_t () this is for 8 kmers per thread
	__shared__ unsigned long long int seq_shared[21]; // 158 bytes of sequence divided by 8 bytes per uint64_t () this is for 4 kmers per thread

	
	//seq_shared[threadIdx.x] = ((unsigned long long int *) sequence)[threadIdx.x + blockIdx.x * 36]; // I thought it was 20 but it is 16. 20 bytes are needed but the ones between 16 and 20 are only used for the last kmer - so many are missing
	if(threadIdx.x < 21) seq_shared[threadIdx.x] = ((unsigned long long int *) sequence)[threadIdx.x + blockIdx.x * 16]; // 4 kmers per thread is: 
	//seq_shared[threadIdx.x] = ((unsigned long long int *) sequence)[threadIdx.x]; //256
	//unsigned long long int fixed_pos = threadIdx.x + 288 * blockIdx.x;
	char * sweet_pointer = (char *) seq_shared;
	int lost[4];


	my_byte = (threadIdx.x << 2);

	for(k=0; k<32; k++){

		//my_byte = threadIdx.x * 8 + k + blockIdx.x * blockDim.x * 8; // the 8 comes from each thread processing 8 kmers
		//if(threadIdx.x == 31) printf("I am thread %d and going for byte %d which is at bank %d\n", threadIdx.x, my_byte, (my_byte / 4) % 32 );
		char c = sweet_pointer[my_byte]; 
		++my_byte;

		/*
		if(c == 'A') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 0;
		if(c == 'C') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 1;
		if(c == 'G') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 2;
		if(c == 'T') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 3;
		if(c == 'N') bad = 0;
		*/
		

		
		if(c == 'A') { hash += 0; if(k<3) lost[k] = 0; }
		if(c == 'C') { hash += pow4[31-k]; if(k<3) lost[k] = 1; }
		if(c == 'G') { hash += pow4_G[31-k]; if(k<3) lost[k] = 2; }
		if(c == 'T') { hash += pow4_T[31-k]; if(k<3) lost[k] = 3; }
		if(c == 'N') bad = 0;
		
		
		
		/*
		unsigned char val = (unsigned char) c;
		unsigned char multiplier = (val & 6) >> 1;
		hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) multiplier;
		//checker = checker | (val & (unsigned char) 8);
		if(c == 'N') bad = 0;
		*/
		
		
		
	}

	
	//table[threadIdx.x + blockIdx.x * blockDim.x * 8] = hash & bad;
	table[threadIdx.x + blockIdx.x * blockDim.x * 4] = hash & bad;

	
	for(k=1; k<4; k++){
		//char next_nucl = sequence[++my_byte];
		char next_nucl = sweet_pointer[my_byte++];
		bad = 0xFFFFFFFFFFFFFFFF;

		if(next_nucl == 'A') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 0;
		if(next_nucl == 'C') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 1;
		if(next_nucl == 'G') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 2;
		if(next_nucl == 'T') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 3;
		if(next_nucl == 'N') bad = 0;

		//table[threadIdx.x + blockIdx.x * blockDim.x*8 + blockDim.x * k] = hash & bad;
		table[threadIdx.x + blockIdx.x * blockDim.x*4 + blockDim.x * k] = hash & bad;
		
	}
}

__global__ void kernel_index_global_coalesced(unsigned long long int * table, const char * sequence) {
		

	unsigned long long int k, j, hash, my_byte;
	unsigned long long int bad = 0xFFFFFFFFFFFFFFFF;
	for(j=0; j<4; j++){

		k = 0;
		hash = 0;

		int value;
		char c;
		my_byte = (threadIdx.x << 2) + j;
		
		while(k < 32){

			// Which BYTE Do I Want?
			
			// Fetch INT from BYTE position
			if((k+j) % 4 == 0) value = ((int *) sequence)[my_byte >> 2];

			c = (char) (value >> ((my_byte % 4) << 3));
			
			
			if(c == 'A') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 0;
			if(c == 'C') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 1;
			if(c == 'G') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 2;
			if(c == 'T') hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) 3;
			if(c == 'N') bad = 0;
			
			
			
			
			/*
			if(c == 'A') hash += 0;
			if(c == 'C') hash += pow4[k];
			if(c == 'G') hash += pow4_G[k];
			if(c == 'T') hash += pow4_T[k];
			if(c == 'N') bad = 0;
			*/
			/*
			unsigned char val = (unsigned char) c;
			unsigned char multiplier = (val & 6) >> 1;
			hash += ((unsigned long long int)1 << (k << 1)) * (unsigned long long int) multiplier;
			//checker = checker | (val & (unsigned char) 8);
			if(c == 'N') bad = 0;
			*/
			
			++my_byte;
			++k;
			
		}

		table[threadIdx.x + j*32 + blockIdx.x * blockDim.x * 4] = hash & bad;
	}

	
	//table[threadIdx.x] = hash & bad;
}



__global__ void kernel_index_global_any_simplest(unsigned long long int * table, const char * sequence) {
		
	unsigned hash;
	
	hash = ((unsigned *)table)[threadIdx.x + blockIdx.x * blockDim.x];

	table[threadIdx.x + blockIdx.x * blockDim.x] = hash;
}


__global__ void kernel_index_global_any_assembled(unsigned long long int * table, const char * sequence) {
		

	unsigned long long int k, j, hash = 0;
		
	unsigned long long int bad = 0xFFFFFFFFFFFFFFFF;

	char c = sequence[threadIdx.x + 0 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*0)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*0)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*0)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*0)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 1 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*1)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*1)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*1)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*1)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 2 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*2)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*2)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*2)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*2)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 3 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*3)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*3)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*3)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*3)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 4 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*4)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*4)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*4)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*4)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 5 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*5)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*5)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*5)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*5)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 6 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*6)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*6)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*6)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*6)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 7 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*7)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*7)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*7)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*7)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 8 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*8)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*8)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*8)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*8)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 9 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*9)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*9)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*9)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*9)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 10 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*10)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*10)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*10)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*10)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 11 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*11)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*11)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*11)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*11)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 12 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*12)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*12)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*12)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*12)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 13 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*13)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*13)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*13)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*13)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 14 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*14)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*14)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*14)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*14)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 15 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*15)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*15)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*15)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*15)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 16 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*16)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*16)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*16)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*16)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 17 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*17)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*17)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*17)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*17)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 18 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*18)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*18)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*18)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*18)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 19 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*19)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*19)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*19)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*19)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 20 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*20)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*20)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*20)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*20)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 21 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*21)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*21)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*21)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*21)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 22 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*22)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*22)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*22)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*22)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 23 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*23)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*23)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*23)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*23)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 24 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*24)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*24)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*24)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*24)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 25 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*25)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*25)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*25)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*25)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 26 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*26)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*26)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*26)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*26)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 27 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*27)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*27)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*27)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*27)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 28 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*28)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*28)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*28)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*28)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 29 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*29)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*29)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*29)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*29)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 30 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*30)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*30)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*30)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*30)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 31 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((unsigned long long int)1 << (2*31)) * (unsigned long long int) 0;
	if(c == 'C') hash += ((unsigned long long int)1 << (2*31)) * (unsigned long long int) 1;
	if(c == 'G') hash += ((unsigned long long int)1 << (2*31)) * (unsigned long long int) 2;
	if(c == 'T') hash += ((unsigned long long int)1 << (2*31)) * (unsigned long long int) 3;
	if(c == 'N') bad = 0;
		
		

		/*
		unsigned char val = (unsigned char) c;
		unsigned char multiplier = (val & 6) >> 1;
		hash += ((unsigned long long int)1 << (2*k)) * (unsigned long long int) multiplier;
		//checker = checker | (val & (unsigned char) 8);
		if(c == 'N') bad = 0;
		*/
	

	table[threadIdx.x + blockIdx.x * blockDim.x] = hash & bad;
	//table[threadIdx.x] = hash & bad;
}

__global__ void some_float_experiment(unsigned long long int * table, const char * sequence) {
		

	

	//float f = ((float *) sequence)[threadIdx.x + blockIdx.x * blockDim.x]; 
	char f = ((char *) sequence)[threadIdx.x + blockIdx.x * blockDim.x]; 

	f = f * 2.0;
	
	
	table[threadIdx.x + blockIdx.x * blockDim.x] = f; 
	//if(threadIdx.x == 0) table[0] = f; 
	//if(threadIdx.x == 1) table[1] = f; 
	
	
	//table[threadIdx.x] = hash & bad;
}


