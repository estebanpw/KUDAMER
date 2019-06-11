
CC=gcc
CXX=g++ -mavx2 -fabi-version=0
NVIDIAC=nvcc
OPENCL=/usr/local/cuda-8.0/include/
CFLAGS=-O3 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE #-DVERBOSE
#NVIDIAFLAGS=-Xptxas -O3 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE #-DVERBOSE
#NVIDIAFLAGS=-rdc=true -Xptxas="-v" -O3 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE #-DVERBOSE
NVIDIAFLAGS=-rdc=true -O3 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE -DREDUCEDMATH #-DVERBOSE
#NVIDIAFLAGS=-rdc=true -maxrregcount 20 -Xptxas="-v" -O3 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE #-DVERBOSE
#NVIDIAFLAGS=-g -lineinfo -Xptxas="-v" -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE #-DVERBOSE

# To make L1 cache behave as a cache actually
#NVIDIAFLAGS=-rdc=true -Xptxas -dlcm=ca -O3 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE #-DVERBOSE

BIN=.

all: index_kmers_split_dyn_mat cpu_index cpu_index_parallel

index_kmers_split_dyn_mat: index_kmers_split_dyn_mat.cu
	$(NVIDIAC) $(NVIDIAFLAGS) index_kmers_split_dyn_mat.cu kernels.cu -o $(BIN)/index_kmers_split_dyn_mat

cpu_index: cpu_index.c
	$(CC) $(CFLAGS) cpu_index.c -lm -o $(BIN)/cpu_index

cpu_index_vectorized: cpu_index_vectorized.c
	$(CXX) $(CFLAGS) cpu_index_vectorized.c -lm -o $(BIN)/cpu_index_vectorized

cpu_index_parallel: cpu_index_parallel.c
	$(CC) $(CFLAGS) cpu_index_parallel.c -lpthread -lm -o $(BIN)/cpu_index_parallel

clean:
	rm -rf $(BIN)/cpu_index $(BIN)/index_kmers_split_dyn_mat $(BIN)/cpu_index_parallel 