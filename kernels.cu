#include "kernels.cuh"
#define WARP_SIZE 32
#define KMER_SIZE 32
#define KMERS_PER_THREAD 3
#define BYTES_PER_REGISTER 4

#define ULLI unsigned long long int

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

__global__ void kernel_register(ULLI * table, const char * sequence) {
	
	int i, k;
	ULLI hash = 0;

	// Notice you need not to synchronize this load as its intra warp and the number of threads must be 32
	unsigned int value = ((unsigned int *)sequence)[threadIdx.x + blockIdx.x * 24 ]; // 24*4 = 96 are the bytes corresponding to the 96 kmers used from the 128 byte transaction
	unsigned int temp_value; 
	char byte;
	ULLI bad;
	

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
			hash += ((ULLI)1 << (k << 1)) * (ULLI) multiplier;
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
			
			
			/*
			if(byte == 'A') hash += ((ULLI)1 << (k << 1)) * (ULLI) 0;
			if(byte == 'C') hash += ((ULLI)1 << (k << 1)) * (ULLI) 1;
			if(byte == 'G') hash += ((ULLI)1 << (k << 1)) * (ULLI) 2;
			if(byte == 'T') hash += ((ULLI)1 << (k << 1)) * (ULLI) 3;
			if(byte == 'N') bad = 0;
			*/

			hash = hash << 2;
			if((char) byte == 'C') hash = hash + 1;
			if((char) byte == 'G') hash = hash + 2;
			if((char) byte == 'T') hash = hash + 3;
			if((char) byte == 'N') bad = 0;

		}
		//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & bad;
		table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & bad;
		//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & (~((checker >> 3) * 0xFFFFFFFFFFFFFFFF));
		//if(checker == 0) table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash;
		
	}
}

__global__ void kernel_register_less_synchro(ULLI * table, const char * sequence) {
	
	int i, k;
	ULLI hash = 0;

	unsigned int value = ((unsigned int *)sequence)[threadIdx.x + blockIdx.x * 24 ]; // 24*4 = 96 are the bytes corresponding to the 96 kmers used from the 128 byte transaction
	unsigned int temp_value;
	char byte;
	ULLI bad;
	

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
			hash += ((ULLI)1 << (2*k)) * (ULLI) multiplier;
			//checker = checker | (val & (unsigned char) 8);
			if(byte == 'N') bad = 0;
			
			*/

			/*
			if(byte == 'A') hash += ((ULLI)1 << (2*k)) * (ULLI) 0;
			if(byte == 'C') hash += ((ULLI)1 << (2*k)) * (ULLI) 1;
			if(byte == 'G') hash += ((ULLI)1 << (2*k)) * (ULLI) 2;
			if(byte == 'T') hash += ((ULLI)1 << (2*k)) * (ULLI) 3;
			if(byte == 'N') bad = 0;
			*/

			/*
			if(byte == 'A') hash += 0;
			if(byte == 'C') hash += pow4[k];
			if(byte == 'G') hash += pow4_G[k];
			if(byte == 'T') hash += pow4_T[k];
			if(byte == 'N') bad = 0;
			*/

			hash = hash << 2;
			if((char) byte == 'C') hash = hash + 1;
			if((char) byte == 'G') hash = hash + 2;
			if((char) byte == 'T') hash = hash + 3;
			if((char) byte == 'N') bad = 0;
			

		}
		//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & bad;
		table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & bad;
		//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & (~((checker >> 3) * 0xFFFFFFFFFFFFFFFF));
		//if(checker == 0) table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash;
		
	}
}

__global__ void kernel_register_no_synchro_exp(ULLI * table, const char * sequence) {
	
	int i, k;
	ULLI hash = 0;

	unsigned int value = ((unsigned int *)sequence)[threadIdx.x + blockIdx.x * 24 ]; // 24*4 = 96 are the bytes corresponding to the 96 kmers used from the 128 byte transaction
	unsigned int temp_value;
	char byte;
	ULLI bad;
	

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
			hash += ((ULLI)1 << (k << 1)) * (ULLI) multiplier;
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
			
			/*
			if(byte == 'A') hash += ((ULLI)1 << (k << 1)) * (ULLI) 0;
			if(byte == 'C') hash += ((ULLI)1 << (k << 1)) * (ULLI) 1;
			if(byte == 'G') hash += ((ULLI)1 << (k << 1)) * (ULLI) 2;
			if(byte == 'T') hash += ((ULLI)1 << (k << 1)) * (ULLI) 3;
			if(byte == 'N') bad = 0;
			*/

			hash = hash << 2;
			if((char) byte == 'C') hash = hash + 1;
			if((char) byte == 'G') hash = hash + 2;
			if((char) byte == 'T') hash = hash + 3;
			if((char) byte == 'N') bad = 0;
			

		}
		//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & bad;
		table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & bad;
		//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & (~((checker >> 3) * 0xFFFFFFFFFFFFFFFF));
		//if(checker == 0) table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash;
		
	}
}

// Use this one with second mode of printing kmers

__global__ void kernel_register_fast_hash_no_synchro_exp(ULLI * table, const char * sequence) {
	


	int i, k;
	ULLI hash = 0;

	// Each will do 6 kmers
	// So 32 threads do 32*6 = 192 kmers, starting at first 192 bytes
	// So last kmer processed starts at position 191
	// 192 / 8 = 24

	ULLI value = ((ULLI *)sequence)[threadIdx.x + blockIdx.x * 24 ]; // 24*8 = 192 are the bytes corresponding to the 96 kmers used from the 128 byte transaction
	ULLI temp_value;
	char byte;
	ULLI bad;
	

	i = 0;

	unsigned kmer_start = threadIdx.x * 6 + i; // 6 because of 6 kmers per thread
	unsigned int_pos = kmer_start >> 3; // 8 because bytes per register
	hash = 0;
	bad = 0xFFFFFFFFFFFFFFFF;
	
	for(k=0; k<KMER_SIZE; k++){
		
		temp_value = __shfl_sync((1 << int_pos) || (1 << threadIdx.x), value, int_pos);
		

		// This used to be:
		// byte = (char) (temp_value >> ((kmer_start % BYTES_PER_REGISTER)*8));
		//
		byte = (char) (temp_value >> ((kmer_start & 7) << 3));
		
		//if(threadIdx.x == 0 && blockIdx.x == 0) printf("[th:%d, bl:%d, i:%d, k:%d] My bytes: %c: \n", threadIdx.x, blockIdx.x, i, k, byte);
		
		
		++kmer_start;
		int_pos = kmer_start >> 3;

		/*
		unsigned char val = (unsigned char) byte;
		unsigned char multiplier = (val & 6) >> 1;
		hash += ((ULLI)1 << (k << 1)) * (ULLI) multiplier;
		//checker = checker | (val & (unsigned char) 8);
		if(byte == 'N') bad = 0;
		*/
		
		
		
		
		if(byte == 'C') hash += pow4[31-k];
		if(byte == 'G') hash += pow4_G[31-k];
		if(byte == 'T') hash += pow4_T[31-k];
		if(byte == 'N') bad = 0;
		
		
		/*
		if(byte == 'A') hash += ((ULLI)1 << (k << 1)) * (ULLI) 0;
		if(byte == 'C') hash += ((ULLI)1 << (k << 1)) * (ULLI) 1;
		if(byte == 'G') hash += ((ULLI)1 << (k << 1)) * (ULLI) 2;
		if(byte == 'T') hash += ((ULLI)1 << (k << 1)) * (ULLI) 3;
		if(byte == 'N') bad = 0;
		*/

		/*
		hash = hash << 2;
		if((char) byte == 'C') hash = hash + 1;
		if((char) byte == 'G') hash = hash + 2;
		if((char) byte == 'T') hash = hash + 3;
		if((char) byte == 'N') bad = 0;
		*/
		

	}
	//table[threadIdx.x + 32*i + 192 * blockIdx.x] = hash & bad;
	table[threadIdx.x + (i << 5) + 192 * blockIdx.x] = hash & bad;
	//table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash & (~((checker >> 3) * 0xFFFFFFFFFFFFFFFF));
	//if(checker == 0) table[threadIdx.x + 32*i + 96 * blockIdx.x] = hash;

	for(i=1; i<6; i++){
		//char next_nucl = sequence[++my_byte];
		
		temp_value = __shfl_sync((1 << int_pos) || (1 << threadIdx.x), value, int_pos);
		byte = (char) (temp_value >> ((kmer_start & 7) << 3));
		++kmer_start;
		int_pos = kmer_start >> 3;

		

		bad = 0xFFFFFFFFFFFFFFFF;

		/*
		if(next_nucl == 'A') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 0;
		if(next_nucl == 'C') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 1;
		if(next_nucl == 'G') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 2;
		if(next_nucl == 'T') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 3;
		if(next_nucl == 'N') bad = 0;
		*/

		
		hash = hash << 2;
		
		
		if((char) byte == 'C') hash = hash + 1;
		if((char) byte == 'G') hash = hash + 2;
		if((char) byte == 'T') hash = hash + 3;
		if((char) byte == 'N') bad = 0;
		
		/*
		if((char) byte == 'C') hash = hash + (ULLI)4611686018427387904L;
		if((char) byte == 'G') hash = hash + 2*(ULLI)4611686018427387904L;
		if((char) byte == 'T') hash = hash + 3*(ULLI)4611686018427387904L;
		if((char) byte == 'N') bad = 0;
		*/
		
		

		/*
		if(byte == 'C') hash += pow4[31];
		if(byte == 'G') hash += pow4_G[31];
		if(byte == 'T') hash += pow4_T[31];
		if(byte == 'N') bad = 0;
		*/

		//table[threadIdx.x + 32*i + 192 * blockIdx.x] = hash & bad;
		table[threadIdx.x + (i << 5) + 192 * blockIdx.x] = hash & bad;
		
	}
}

// Use this one with second mode of printing kmers

__global__ void kernel_register_fast_hash_no_synchro_exp_64(ULLI * table, const char * sequence) {
	
	int i, k;
	ULLI hash = 0;

	// Each will do 6 kmers
	// So 64 threads do 64*6 = 384 kmers, first 384 bytes
	// So last kmer processed starts at position 383
	// 384 / 8 = 48

	// Notice that threads in a warp execute the same if else block so no divergence
	ULLI value;


	// The first warp (up to 32 threads) finishes on kmer starting 191
	// Therefore our second warp bundle (threads 32-63) must start on position byte 192 which is 192/8 = 24 
	// We can make it start at ULLI 24 by subtracting 8 to thread numbers, since they start on 32 -> 32-8 = 24 and so on
	if(threadIdx.x < 32)
	{
		value = ((ULLI *)sequence)[threadIdx.x + blockIdx.x * 48 ]; // 48*8 = 384 are the bytes 
	}
	else
	{
		value = ((ULLI *)sequence)[threadIdx.x + blockIdx.x * 48 - 8]; // This makes it start at 192 bytes which is were thread 32 ends
	}

	ULLI temp_value;
	char byte;
	ULLI bad;
	

	i = 0;

	unsigned kmer_start = (threadIdx.x & 31) * 6 + i; // 6 because of 6 kmers per thread
	unsigned int_pos = kmer_start >> 3; // 8 because bytes per register
	hash = 0;
	bad = 0xFFFFFFFFFFFFFFFF;
	
	for(k=0; k<KMER_SIZE; k++){
		
		//temp_value = __shfl_sync((1 << int_pos) || (1 << (threadIdx.x % WARP_SIZE)), value, int_pos);
		temp_value = __shfl_sync((1 << int_pos) || (1 << (threadIdx.x & 31)), value, int_pos);
		
		
		
		//byte = (char) (temp_value >> (kmer_start % 8)*8);
		byte = (char) (temp_value >> ((kmer_start & 7) << 3));
		
		//if(threadIdx.x == 0 && blockIdx.x == 0) printf("[th:%d, bl:%d, i:%d, k:%d] My bytes: %c: \n", threadIdx.x, blockIdx.x, i, k, byte);
		
		
		++kmer_start;
		int_pos = kmer_start >> 3;

		/*
		unsigned char val = (unsigned char) byte;
		unsigned char multiplier = (val & 6) >> 1;
		hash += ((ULLI)1 << (k << 1)) * (ULLI) multiplier;
		//checker = checker | (val & (unsigned char) 8);
		if(byte == 'N') bad = 0;
		*/
		
		
		
		
		if(byte == 'C') hash += pow4[31-k];
		if(byte == 'G') hash += pow4_G[31-k];
		if(byte == 'T') hash += pow4_T[31-k];
		if(byte == 'N') bad = 0;
		
		
		
		/*
		if(byte == 'A') hash += ((ULLI)1 << (k << 1)) * (ULLI) 0;
		if(byte == 'C') hash += ((ULLI)1 << (k << 1)) * (ULLI) 1;
		if(byte == 'G') hash += ((ULLI)1 << (k << 1)) * (ULLI) 2;
		if(byte == 'T') hash += ((ULLI)1 << (k << 1)) * (ULLI) 3;
		if(byte == 'N') bad = 0;
		*/

		/*
		hash = hash << 2;
		if((char) byte == 'C') hash = hash + 1;
		if((char) byte == 'G') hash = hash + 2;
		if((char) byte == 'T') hash = hash + 3;
		if((char) byte == 'N') bad = 0;
		*/
		

	}
	//table[threadIdx.x + 64*i + 384 * blockIdx.x] = hash & bad;
	table[threadIdx.x + (i << 6) + 384 * blockIdx.x] = hash & bad;
	
	for(i=1; i<6; i++){
		//char next_nucl = sequence[++my_byte];
		
		temp_value = __shfl_sync((1 << int_pos) || (1 << (threadIdx.x & 31)), value, int_pos);
		byte = (char) (temp_value >> ((kmer_start & 7) << 3));
		++kmer_start;
		int_pos = kmer_start >> 3;

		bad = 0xFFFFFFFFFFFFFFFF;

		/*
		if(next_nucl == 'A') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 0;
		if(next_nucl == 'C') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 1;
		if(next_nucl == 'G') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 2;
		if(next_nucl == 'T') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 3;
		if(next_nucl == 'N') bad = 0;
		*/
		hash = hash << 2;
		if((char) byte == 'C') hash = hash + 1;
		if((char) byte == 'G') hash = hash + 2;
		if((char) byte == 'T') hash = hash + 3;
		if((char) byte == 'N') bad = 0;

		//table[threadIdx.x + blockIdx.x * blockDim.x*8 + blockDim.x * k] = hash & bad;
		table[threadIdx.x + (i << 6) + 384 * blockIdx.x] = hash & bad;
		
	}
}


// This kernel is to be executed with 32 threads per block 

__global__ void kernel_index32(ULLI * table, const char * sequence) {
		

	ULLI k, j, hash;
	unsigned char multiplier, val;

	__shared__ ULLI seq_shared[20]; // 160 bytes of sequence divided by 8 bytes per uint64_t (should be 156)
	//												  // Last thread (31-th) accesses bytes 124 to 159
	
	
	
	if(threadIdx.x < 20){
		seq_shared[threadIdx.x] = ((ULLI *) sequence)[threadIdx.x + blockIdx.x * 16]; // I thought it was 20 but it is 16. 20 bytes are needed but the ones between 16 and 20 are only used for the last kmer - so many are missing
	}
	ULLI fixed_pos = threadIdx.x + 128 * blockIdx.x;
	//__syncthreads(); //not needed since this is for 32 threads per block

	char * sweet_pointer = (char *) seq_shared;
	
	
	for(j=0; j<4; j++){ // the indexing makes [0,1,2,3,4...] -> [0,4,8,12,16] so we need to fit in the [1,2,3], etc.
		char c;
		ULLI bad = 0xFFFFFFFFFFFFFFFF;
		ULLI pos = fixed_pos + blockDim.x * j;
		hash = 0;
		for(k=0; k<32; k++){
			
			c = sweet_pointer[threadIdx.x * 4 + k + j];
			// This is better because it uses online 10 registers and the branch divergence is SO low (4 ops...)
			
			/*
			if(c == 'A') hash += ((ULLI)1 << (k << 1)) * (ULLI) 0;
			if(c == 'C') hash += ((ULLI)1 << (k << 1)) * (ULLI) 1;
			if(c == 'G') hash += ((ULLI)1 << (k << 1)) * (ULLI) 2;
			if(c == 'T') hash += ((ULLI)1 << (k << 1)) * (ULLI) 3;
			if(c == 'N') bad = 0;
			*/
			
			/*
			if(c == 'A') hash += 0;
			if(c == 'C') hash += pow4[k];
			if(c == 'G') hash += pow4_G[k];
			if(c == 'T') hash += pow4_T[k];
			if(c == 'N') bad = 0;
			*/
			
			

			/*
			val = (unsigned char) sweet_pointer[threadIdx.x * 4 + k];
			multiplier = (val & 6) >> 1;
			hash += (1 << (2*k)) * (ULLI) multiplier;
			if(c == 'N') bad = 0;
			*/

			hash = hash << 2;
			if((char) c == 'C') hash = hash + 1;
			if((char) c == 'G') hash = hash + 2;
			if((char) c == 'T') hash = hash + 3;
			if((char) c == 'N') bad = 0;
			

		}
		//if(bad == 1) hash = 0;
		table[pos] = hash & bad;
		//table[threadIdx.x + blockDim.x * j + 224 * blockIdx.x] = hash;
	}
}



// This kernel is to be executed with 64 threads per block 
// Remember to set the block size to 64 !!!!!!!!

__global__ void kernel_index64(ULLI * table, const char * sequence) {
		

	ULLI k, j, hash;
	unsigned char multiplier, val;

	__shared__ ULLI seq_shared[40]; // 320 bytes of sequence divided by 8 bytes per uint64_t (should be 156)
	//												// Last thread (63-th) accesses bytes 63*4 = 252, 252+3 = 255 - 287. 287/8 = ~36
	
	
	if(threadIdx.x < 40){
		seq_shared[threadIdx.x] = ((ULLI *) sequence)[threadIdx.x + blockIdx.x * 32]; // I thought it was 20 but it is 16. 20 bytes are needed but the ones between 36 and 40 are only used for the last kmer - so many are missing
	}
	__syncthreads();

	char * sweet_pointer = (char *) seq_shared;
	
	
	for(j=0; j<4; j++){ // the indexing makes [0,1,2,3,4...] -> [0,4,8,12,16] so we need to fit in the [1,2,3], etc.
		char c;
		ULLI bad = 0xFFFFFFFFFFFFFFFF;
		ULLI pos = threadIdx.x + blockDim.x * j + 256 * blockIdx.x;
		hash = 0;
		for(k=0; k<32; k++){
			
			c = sweet_pointer[threadIdx.x * 4 + k + j];
			// This is better because it uses online 10 registers and the branch divergence is SO low (4 ops...)
			
			/*
			if(c == 'A') hash += ((ULLI)1 << (k << 1)) * (ULLI) 0;
			if(c == 'C') hash += ((ULLI)1 << (k << 1)) * (ULLI) 1;
			if(c == 'G') hash += ((ULLI)1 << (k << 1)) * (ULLI) 2;
			if(c == 'T') hash += ((ULLI)1 << (k << 1)) * (ULLI) 3;
			if(c == 'N') bad = 0;
			*/
			
			
			/*
			if(c == 'A') hash += ((ULLI)1 << (2*(31-k))) * (ULLI) 0;
			if(c == 'C') hash += ((ULLI)1 << (2*(31-k))) * (ULLI) 1;
			if(c == 'G') hash += ((ULLI)1 << (2*(31-k))) * (ULLI) 2;
			if(c == 'T') hash += ((ULLI)1 << (2*(31-k))) * (ULLI) 3;
			*/

			/*
			if(c == 'A') hash += 0;
			if(c == 'C') hash += pow4[k];
			if(c == 'G') hash += pow4_G[k];
			if(c == 'T') hash += pow4_T[k];
			if(c == 'N') bad = 0;
			*/
			

			/*
			val = (unsigned char) c;
			multiplier = (val & 6) >> 1;
			hash += ((ULLI)1 << (2*k)) * (ULLI) multiplier;
			if(c == 'N') bad = 0;
			*/


			hash = hash << 2;
			if((char) c == 'C') hash = hash + 1;
			if((char) c == 'G') hash = hash + 2;
			if((char) c == 'T') hash = hash + 3;
			if((char) c == 'N') bad = 0;
			
			

		}
		//if(bad == 1) hash = 0;
		table[pos] = hash & bad;
		//table[threadIdx.x + blockDim.x * j + 224 * blockIdx.x] = hash;
	}
}


__global__ void kernel_index_global32(ULLI * table, const char * sequence) {
		

	ULLI k, j, hash = 0;
		
	ULLI bad = 0xFFFFFFFFFFFFFFFF;

	for(k=0; k<32; k++){
	//for(k=0; k<1; k++){

		char c = sequence[threadIdx.x + k + blockIdx.x * blockDim.x];
		/*
		if(c == 'A') hash += ((ULLI)1 << (k << 1)) * (ULLI) 0;
		if(c == 'C') hash += ((ULLI)1 << (k << 1)) * (ULLI) 1;
		if(c == 'G') hash += ((ULLI)1 << (k << 1)) * (ULLI) 2;
		if(c == 'T') hash += ((ULLI)1 << (k << 1)) * (ULLI) 3;
		if(c == 'N') bad = 0;
		*/
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
		hash += ((ULLI)1 << (2*k)) * (ULLI) multiplier;
		//checker = checker | (val & (unsigned char) 8);
		if(c == 'N') bad = 0;
		*/

		hash = hash << 2;
		if((char) c == 'C') hash = hash + 1;
		if((char) c == 'G') hash = hash + 2;
		if((char) c == 'T') hash = hash + 3;
		if((char) c == 'N') bad = 0;
		
	}

	table[threadIdx.x + blockIdx.x * blockDim.x] = hash & bad;
}

__global__ void kernel_index_global64(ULLI * table, const char * sequence) {
		

	ULLI k, j, hash = 0;
		
	ULLI bad = 0xFFFFFFFFFFFFFFFF;

	for(k=0; k<32; k++){

		char c = sequence[threadIdx.x + k + blockIdx.x * blockDim.x];
		
		/*
		if(c == 'A') hash += ((ULLI)1 << (k << 1)) * (ULLI) 0;
		if(c == 'C') hash += ((ULLI)1 << (k << 1)) * (ULLI) 1;
		if(c == 'G') hash += ((ULLI)1 << (k << 1)) * (ULLI) 2;
		if(c == 'T') hash += ((ULLI)1 << (k << 1)) * (ULLI) 3;
		if(c == 'N') bad = 0;
		*/
		
		
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
		hash += ((ULLI)1 << (2*k)) * (ULLI) multiplier;
		//checker = checker | (val & (unsigned char) 8);
		if(c == 'N') bad = 0;
		*/

		hash = hash << 2;
		if((char) c == 'C') hash = hash + 1;
		if((char) c == 'G') hash = hash + 2;
		if((char) c == 'T') hash = hash + 3;
		if((char) c == 'N') bad = 0;
	}

	table[threadIdx.x + blockIdx.x * blockDim.x] = hash & bad;
	//table[threadIdx.x] = hash & bad;
}

__global__ void kernel_index_global_fast_hash(ULLI * table, const char * sequence) {

	ULLI k, j, hash = 0, my_byte;
	ULLI bad = 0xFFFFFFFFFFFFFFFF;
	int lost[8];

	my_byte = threadIdx.x * 8 + blockIdx.x * blockDim.x * 8;

	for(k=0; k<32; k++){

		//my_byte = threadIdx.x * 8 + k + blockIdx.x * blockDim.x * 8; // the 8 comes from each thread processing 8 kmers
		char c = sequence[my_byte++];

		/*
		if(c == 'A'){ hash += ((ULLI)1 << (k << 1)) * (ULLI) 0; if(k<8) lost[k] = 0; }
		if(c == 'C'){ hash += ((ULLI)1 << (k << 1)) * (ULLI) 1; if(k<8) lost[k] = 1; }
		if(c == 'G'){ hash += ((ULLI)1 << (k << 1)) * (ULLI) 2; if(k<8) lost[k] = 2; }
		if(c == 'T'){ hash += ((ULLI)1 << (k << 1)) * (ULLI) 3; if(k<8) lost[k] = 3; }
		if(c == 'N') bad = 0;
		*/
		

		// this one works perfect
		/*
		if(c == 'A') { hash += 0; if(k<8) lost[k] = 0; }
		if(c == 'C') { hash += pow4[31-k]; if(k<8) lost[k] = 1; }
		if(c == 'G') { hash += pow4_G[31-k]; if(k<8) lost[k] = 2; }
		if(c == 'T') { hash += pow4_T[31-k]; if(k<8) lost[k] = 3; }
		if(c == 'N') bad = 0;
		*/
		
		/*
		unsigned char val = (unsigned char) c;
		unsigned char multiplier = (val & 6) >> 1;
		hash += ((ULLI)1 << (k << 1)) * (ULLI) multiplier;
		if(k<8) lost[k] = (int) multiplier;
		//checker = checker | (val & (unsigned char) 8);
		if(c == 'N') bad = 0;
		*/
		
		hash = hash << 2;
		if((char) c == 'C') hash = hash + 1;
		if((char) c == 'G') hash = hash + 2;
		if((char) c == 'T') hash = hash + 3;
		if((char) c == 'N') bad = 0;
		
		
	}

	table[threadIdx.x + blockIdx.x * blockDim.x*8] = hash & bad;

	for(k=1; k<8; k++){
		char next_nucl = sequence[my_byte++];
		bad = 0xFFFFFFFFFFFFFFFF;

		/*
		if(next_nucl == 'A') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 0;
		if(next_nucl == 'C') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 1;
		if(next_nucl == 'G') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 2;
		if(next_nucl == 'T') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 3;
		if(next_nucl == 'N') bad = 0;
		*/

		hash = hash << 2;
		if((char) next_nucl == 'C') hash = hash + 1;
		if((char) next_nucl == 'G') hash = hash + 2;
		if((char) next_nucl == 'T') hash = hash + 3;
		if((char) next_nucl == 'N') bad = 0;

		table[threadIdx.x + blockIdx.x * blockDim.x*8 + blockDim.x * k] = hash & bad;
	}
}


__global__ void kernel_index_global_fast_hash_on_shared(ULLI * table, const char * sequence) {

	ULLI hash = 0; 
	int my_byte, k;
	ULLI bad = 0xFFFFFFFFFFFFFFFF;

	//if(threadIdx.x == 1 && blockIdx.x == 0) printf("ssize: %d\n", sizeof(ULLI));

	
	//__shared__ ULLI seq_shared[36]; // 288 bytes of sequence divided by 8 bytes per uint64_t () this is for 8 kmers per thread
	__shared__ ULLI seq_shared[21]; // 158 bytes of sequence divided by 8 bytes per uint64_t () this is for 4 kmers per thread

	
	
	//if(threadIdx.x < 21) seq_shared[threadIdx.x] = ((ULLI *) sequence)[threadIdx.x + blockIdx.x * 16]; // 4 kmers per thread is: 
	if(threadIdx.x < 21) seq_shared[threadIdx.x] = ((ULLI *) sequence)[threadIdx.x + blockIdx.x << 4]; // 4 kmers per thread is: 
	char * sweet_pointer = (char *) seq_shared;
	


	my_byte = (threadIdx.x << 2);

	for(k=0; k<32; k++){

		//my_byte = threadIdx.x * 8 + k + blockIdx.x * blockDim.x * 8; // the 8 comes from each thread processing 8 kmers
		//if(threadIdx.x == 31) printf("I am thread %d and going for byte %d which is at bank %d\n", threadIdx.x, my_byte, (my_byte / 4) % 32 );
		char c = sweet_pointer[my_byte]; 
		++my_byte;

		/*
		if(c == 'A'){ hash += ((ULLI)1 << (k << 1)) * (ULLI) 0; }
		if(c == 'C'){ hash += ((ULLI)1 << (k << 1)) * (ULLI) 1; }
		if(c == 'G'){ hash += ((ULLI)1 << (k << 1)) * (ULLI) 2; }
		if(c == 'T'){ hash += ((ULLI)1 << (k << 1)) * (ULLI) 3; }
		if(c == 'N') bad = 0;
		*/
		
		
		

		
		// this one works perfect
		/*
		if(c == 'A') hash += 0; 
		if(c == 'C') hash += pow4[31-k]; 
		if(c == 'G') hash += pow4_G[31-k]; 
		if(c == 'T') hash += pow4_T[31-k]; 
		if(c == 'N') bad = 0;
		
		
		
		
		/*
		unsigned char val = (unsigned char) c;
		unsigned char multiplier = (val & 6) >> 1;
		hash += ((ULLI)1 << (k << 1)) * (ULLI) multiplier;
		//if(k<3) lost[k] = (int) multiplier;
		//checker = checker | (val & (unsigned char) 8);
		if(c == 'N') bad = 0;
		*/

		
		hash = hash << 2;
		if((char) c == 'C') hash = hash + 1;
		if((char) c == 'G') hash = hash + 2;
		if((char) c == 'T') hash = hash + 3;
		if((char) c == 'N') bad = 0;
		
	}

	
	//table[threadIdx.x + blockIdx.x * blockDim.x * 4] = hash & bad;
	table[threadIdx.x + blockIdx.x * blockDim.x << 2] = hash & bad;

	
	for(k=1; k<4; k++){
		//char next_nucl = sequence[++my_byte];
		char next_nucl = sweet_pointer[my_byte++];
		bad = 0xFFFFFFFFFFFFFFFF;

		/*
		if(next_nucl == 'A') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 0;
		if(next_nucl == 'C') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 1;
		if(next_nucl == 'G') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 2;
		if(next_nucl == 'T') hash =  4 * (hash - 4611686018427387904L * lost[k-1]) + 3;
		if(next_nucl == 'N') bad = 0;
		*/
		hash = hash << 2;
		if((char) next_nucl == 'C') hash = hash + 1;
		if((char) next_nucl == 'G') hash = hash + 2;
		if((char) next_nucl == 'T') hash = hash + 3;
		if((char) next_nucl == 'N') bad = 0;

		//table[threadIdx.x + blockIdx.x * blockDim.x*4 + blockDim.x * k] = hash & bad;
		table[threadIdx.x + blockIdx.x * blockDim.x << 2 + blockDim.x * k] = hash & bad;
		
	}
}

__global__ void kernel_index_global_coalesced(ULLI * table, const char * sequence) {
		

	ULLI k, j, hash, my_byte;
	ULLI bad = 0xFFFFFFFFFFFFFFFF;
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
			
			
			if(c == 'A') hash += ((ULLI)1 << (k << 1)) * (ULLI) 0;
			if(c == 'C') hash += ((ULLI)1 << (k << 1)) * (ULLI) 1;
			if(c == 'G') hash += ((ULLI)1 << (k << 1)) * (ULLI) 2;
			if(c == 'T') hash += ((ULLI)1 << (k << 1)) * (ULLI) 3;
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
			hash += ((ULLI)1 << (k << 1)) * (ULLI) multiplier;
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



__global__ void kernel_index_global_any_simplest(ULLI * table, const char * sequence) {
		
	unsigned hash;
	
	hash = ((unsigned *)table)[threadIdx.x + blockIdx.x * blockDim.x];

	table[threadIdx.x + blockIdx.x * blockDim.x] = hash;
}


__global__ void kernel_index_global_any_assembled(ULLI * table, const char * sequence) {
		

	ULLI k, j, hash = 0;
		
	ULLI bad = 0xFFFFFFFFFFFFFFFF;

	char c = sequence[threadIdx.x + 0 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*0)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*0)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*0)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*0)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 1 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*1)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*1)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*1)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*1)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 2 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*2)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*2)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*2)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*2)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 3 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*3)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*3)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*3)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*3)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 4 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*4)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*4)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*4)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*4)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 5 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*5)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*5)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*5)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*5)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 6 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*6)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*6)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*6)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*6)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 7 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*7)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*7)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*7)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*7)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 8 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*8)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*8)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*8)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*8)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 9 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*9)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*9)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*9)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*9)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 10 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*10)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*10)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*10)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*10)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 11 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*11)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*11)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*11)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*11)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 12 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*12)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*12)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*12)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*12)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 13 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*13)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*13)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*13)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*13)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 14 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*14)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*14)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*14)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*14)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 15 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*15)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*15)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*15)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*15)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 16 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*16)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*16)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*16)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*16)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 17 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*17)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*17)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*17)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*17)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 18 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*18)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*18)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*18)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*18)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 19 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*19)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*19)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*19)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*19)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 20 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*20)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*20)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*20)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*20)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 21 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*21)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*21)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*21)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*21)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 22 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*22)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*22)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*22)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*22)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 23 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*23)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*23)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*23)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*23)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 24 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*24)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*24)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*24)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*24)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 25 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*25)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*25)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*25)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*25)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 26 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*26)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*26)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*26)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*26)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 27 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*27)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*27)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*27)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*27)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 28 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*28)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*28)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*28)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*28)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 29 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*29)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*29)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*29)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*29)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 30 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*30)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*30)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*30)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*30)) * (ULLI) 3;
	if(c == 'N') bad = 0;

	c = sequence[threadIdx.x + 31 + blockIdx.x * blockDim.x];
	if(c == 'A') hash += ((ULLI)1 << (2*31)) * (ULLI) 0;
	if(c == 'C') hash += ((ULLI)1 << (2*31)) * (ULLI) 1;
	if(c == 'G') hash += ((ULLI)1 << (2*31)) * (ULLI) 2;
	if(c == 'T') hash += ((ULLI)1 << (2*31)) * (ULLI) 3;
	if(c == 'N') bad = 0;
		
		

		/*
		unsigned char val = (unsigned char) c;
		unsigned char multiplier = (val & 6) >> 1;
		hash += ((ULLI)1 << (2*k)) * (ULLI) multiplier;
		//checker = checker | (val & (unsigned char) 8);
		if(c == 'N') bad = 0;
		*/
	

	table[threadIdx.x + blockIdx.x * blockDim.x] = hash & bad;
	//table[threadIdx.x] = hash & bad;
}

__global__ void some_float_experiment(ULLI * table, const char * sequence) {
		

	

	//float f = ((float *) sequence)[threadIdx.x + blockIdx.x * blockDim.x]; 
	char f = ((char *) sequence)[threadIdx.x + blockIdx.x * blockDim.x]; 

	f = f * 2.0;
	
	
	table[threadIdx.x + blockIdx.x * blockDim.x] = f; 
	//if(threadIdx.x == 0) table[0] = f; 
	//if(threadIdx.x == 1) table[1] = f; 
	
	
	//table[threadIdx.x] = hash & bad;
}


