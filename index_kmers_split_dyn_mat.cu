
// Standard utilities and common systems includes
#include "kernels.cuh"
#include "common.h"

#define BUFFER_SIZE 2048
#define CORES_PER_COMPUTE_UNIT 32
#define KMER_SIZE 32
//#define DIMENSION 1000


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void init_args(int argc, char ** av, FILE ** query, unsigned * selected_device, FILE ** ref, FILE ** out, unsigned * write);
void perfect_hash_to_word(char * word, uint64_t hash, uint64_t k);
void print_kmers_to_file(uint64_t * table_mem, uint64_t table_size, FILE * fout);
void print_kmers_to_file_paused(uint64_t * table_mem, uint64_t table_size);
char * get_dirname(char * path);
char * get_basename(char * path);

int main(int argc, char ** argv)
{
    uint64_t i;
    unsigned selected_device = 0, write = 0;
    FILE * query = NULL, * ref = NULL, * out = NULL;
    init_args(argc, argv, &query, &selected_device, &ref, &out, &write);

    ////////////////////////////////////////////////////////////////////////////////
    // Get info of devices
    ////////////////////////////////////////////////////////////////////////////////

    int ret_num_devices;
    unsigned ret_num_platforms;
    unsigned ret_address_bits;
    unsigned compute_units;
    char device_info[BUFFER_SIZE]; device_info[0] = '\0';
    uint64_t local_device_RAM, global_device_RAM;
    int work_group_dimensions[3], work_group_size_local;
    int ret;
    
    // Query how many devices there are
    if(cudaSuccess != (ret = cudaGetDeviceCount(&ret_num_devices))){ fprintf(stderr, "Failed to query number of devices\n"); exit(-1); }

    cudaDeviceProp device;

    for(i=0; i<ret_num_devices; i++){
        if( cudaSuccess != (ret = cudaGetDeviceProperties(&device, i))){ fprintf(stderr, "Failed to get cuda device property: %d\n", ret); exit(-1); }

        fprintf(stdout, "\tDevice [%"PRIu64"]: %s\n", i, device.name);
        global_device_RAM = device.totalGlobalMem;
        fprintf(stdout, "\t\tGlobal mem   : %"PRIu64" (%"PRIu64" MB)\n", (uint64_t) global_device_RAM, (uint64_t) global_device_RAM / (1024*1024));
        local_device_RAM = device.sharedMemPerBlock;
        fprintf(stdout, "\t\tLocal mem    : %"PRIu64" (%"PRIu64" KB)\n", (uint64_t) local_device_RAM, (uint64_t) local_device_RAM / (1024));
        compute_units = device.multiProcessorCount;
        fprintf(stdout, "\t\tCompute units: %"PRIu64"\n", (uint64_t) compute_units);
        work_group_size_local = device.maxThreadsPerBlock;
        fprintf(stdout, "\t\tMax work group size: %d\n", work_group_size_local);
        memcpy(work_group_dimensions, device.maxThreadsDim, 3*sizeof(int));
        fprintf(stdout, "\t\tWork size dimensions: (%d, %d, %d)\n", work_group_dimensions[0], work_group_dimensions[1], work_group_dimensions[2]);
        fprintf(stdout, "\t\tWarp size: %d\n", device.warpSize);
        fprintf(stdout, "\t\tGrid dimensions: (%d, %d, %d)\n", device.maxGridSize[0], device.maxGridSize[1], device.maxGridSize[2]);
    }
    //selected_device = 3; // REMOVE --- ONLY FOR TESTING $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    if( cudaSuccess != (ret = cudaSetDevice(selected_device))){ fprintf(stderr, "Failed to get cuda device property: %d\n", ret); exit(-1); }
    fprintf(stdout, "[INFO] Using device %d\n", selected_device);

    if( cudaSuccess != (ret = cudaGetDeviceProperties(&device, selected_device))){ fprintf(stderr, "Failed to get cuda device property: %d\n", ret); exit(-1); }
    global_device_RAM = device.totalGlobalMem;

    
    
    ////////////////////////////////////////////////////////////////////////////////
    // Make index dictionary
    ////////////////////////////////////////////////////////////////////////////////

    

    // Calculate how much ram we can use for every chunk
    uint64_t effective_global_ram =  (global_device_RAM - 100*1024*1024);
    uint64_t ram_to_be_used = effective_global_ram / (sizeof(unsigned char) + sizeof(uint64_t)); //Minus 100 MBs
    uint64_t table_size = (effective_global_ram - ram_to_be_used) / sizeof(uint64_t);
    uint64_t query_len_bytes = 0;


    // Allocate hash table
    uint64_t * table_mem = NULL;
    ret = cudaMalloc(&table_mem, table_size * sizeof(uint64_t));
    if(ret != cudaSuccess){ fprintf(stderr, "Could not allocate memory for table in device. Error: %d\n", ret); exit(-1); }
    fprintf(stdout, "[INFO] Allocated %"PRIu64" bytes for hash %"PRIu64" entries\n", table_size * sizeof(uint64_t), table_size);

    // Initialize table
    ret = cudaMemset(table_mem, 0x0, table_size * sizeof(uint64_t));
    if(ret != cudaSuccess){ fprintf(stderr, "Could not initialize k-mer table. Error: %d\n", ret); exit(-1); }

    // Allocate memory in host for sequence chunk
    char * query_mem_host = (char *) malloc(ram_to_be_used * sizeof(char));
    if(query_mem_host == NULL){ fprintf(stderr, "Could not allocate host memory for query sequence\n"); exit(-1); }

    
    // Set working size
    size_t threads_number = 32;
    size_t number_of_blocks;
    //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte); // NOTICE: MAXWELL ignores this--

    

    
    cudaSharedMemConfig shared_mem_conf;
    ret = cudaDeviceGetSharedMemConfig(&shared_mem_conf);
    if(ret != cudaSuccess){ fprintf(stdout, "[WARNING] Could not get shared memory configuration. Error: %d\n", ret); }
    else { fprintf(stdout, "[INFO] Shared memory configuration is: %s\n", (shared_mem_conf == cudaSharedMemBankSizeFourByte) ? ("4 bytes") : ("8 bytes")); }

    
    
    clock_t begin;
    // Pointer to device memory allocating the query sequence
    char * query_mem = NULL;

    

    // Read the input query in chunks
    int split = 0;
    uint64_t items_read = 0;
    while(!feof(query)){

        // Load sequence chunk into ram
        items_read = fread(query_mem_host, sizeof(char), ram_to_be_used, &query[0]);
        query_len_bytes += items_read;

        // Allocate memory in device for sequence chunk
        ret = cudaMalloc(&query_mem, items_read * sizeof(char));
        if(ret != cudaSuccess){ fprintf(stderr, "Could not allocate memory for query sequence in device (Attempted %"PRIu64" bytes). Error: %d\n", items_read * sizeof(char), ret); exit(-1); }

        ret = cudaMemcpy(query_mem, query_mem_host, items_read, cudaMemcpyHostToDevice);
        if(ret != cudaSuccess){ fprintf(stderr, "Could not copy query sequence to device. Error: %d\n", ret); exit(-1); }

        // Set global working sizes
        fprintf(stdout, "[INFO] Split #%d: %"PRIu64"\n", split, items_read);
        
        ////////////////////////////////////////////////////////////////////////////////
        // Execution configuration
        ////////////////////////////////////////////////////////////////////////////////

        // If shared memory index32
        //number_of_blocks = (((items_read - KMER_SIZE + 1)/4) / threads_number); // Blocks

        // If shared memory index64
        //number_of_blocks = (((items_read - KMER_SIZE + 1)/4) / threads_number); // Blocks, each block takes 32 ulongs, 8 bytes each

        // If global memory 32
        //number_of_blocks = (((items_read - KMER_SIZE + 1)) / threads_number); // Blocks

        // If global memory 64
        //number_of_blocks = (((items_read - KMER_SIZE + 1)) / threads_number); // Blocks

        // If index global coalesced
        //number_of_blocks = (((items_read - KMER_SIZE + 1)) / (threads_number*4)); // Blocks

        // If index global fast from previous hash
        //number_of_blocks = (((items_read - KMER_SIZE + 1)) / (threads_number*8)); // Blocks

        // If index global fast from previous hash using shared memory
        //number_of_blocks = (((items_read - KMER_SIZE + 1)) / (threads_number*4)); // Blocks

        // If register mode 32
        //number_of_blocks = (((items_read - KMER_SIZE + 1)/3) / threads_number); // Blocks

        // If register from fast hash 32
        //number_of_blocks = (((items_read - KMER_SIZE + 1)/6) / threads_number); // Blocks

        // If register from fast hash 64
        number_of_blocks = (((items_read - KMER_SIZE + 1)/6) / threads_number); // Blocks

        // If register mode 32 less synchro
        //number_of_blocks = (((items_read - KMER_SIZE + 1)/3) / threads_number); // Blocks

        // If float experiment
        //number_of_blocks = (((items_read)/4) / threads_number); // Blocks

        


        // For all
        //number_of_blocks = number_of_blocks - (number_of_blocks % threads_number); // Make it evenly divisable



        fprintf(stdout, "[INFO] Blocks: %"PRIu64". Threads: %"PRIu64".\n", (uint64_t) number_of_blocks, threads_number);
        fprintf(stdout, "[INFO] Executing the kernel on split %d\n", split++);
        begin = clock();
        

        // For index32 and 64 the translation to kmers is CORRECT
        //kernel_index32<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index64<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_register<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_register_no_synchro_exp<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index_global_fast_hash_on_shared<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_register_less_synchro<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index_global32<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index_global64<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index_global_fast_hash<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_register_fast_hash_no_synchro_exp<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        kernel_register_fast_hash_no_synchro_exp_64<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index_global_coalesced<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index_global_any_simplest<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index_global_any_assembled<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        
        //some_float_experiment<<<32, 32>>>((unsigned long long int *) table_mem, query_mem);

        
        ret = cudaGetLastError();
        if(ret != cudaSuccess){ fprintf(stderr, "Error enqueuing indexing kernel: %d : %s\n", ret, cudaGetErrorString(cudaGetLastError())); exit(-1); }

        

        // Wait for kernel to finish
        ret = cudaDeviceSynchronize();
        if(ret != cudaSuccess){ fprintf(stderr, "Bad finish of indexing kernel: %d : %s\n", ret, cudaGetErrorString(cudaGetLastError())); exit(-1); }

    	fprintf(stdout, "{WARM UP}[EXPERIMENTAL TIME]        Indexing: t=%e\n", (double) (clock()-begin) / CLOCKS_PER_SEC);
        fprintf(stdout, "{WARM UP}[EXPERIMENTAL BANDWIDTH]   Achieved %e GB/s\n", ((double)items_read/((double)1000*1000*1000)) /((double)(clock()-begin)/CLOCKS_PER_SEC));

        begin = clock();
        //kernel_index_global32<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index_global64<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index_global_coalesced<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index64<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_register<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_register_less_synchro<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_register_no_synchro_exp<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_register_fast_hash_no_synchro_exp<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index_global_fast_hash_on_shared<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        kernel_register_fast_hash_no_synchro_exp_64<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index32<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        //kernel_index_global_fast_hash<<<number_of_blocks, threads_number>>>((unsigned long long int *) table_mem, query_mem);
        ret = cudaGetLastError();
        if(ret != cudaSuccess){ fprintf(stderr, "Error enqueuing indexing kernel: %d : %s\n", ret, cudaGetErrorString(cudaGetLastError())); exit(-1); }

        

        // Wait for kernel to finish
        ret = cudaDeviceSynchronize();
        if(ret != cudaSuccess){ fprintf(stderr, "Bad finish of indexing kernel: %d : %s\n", ret, cudaGetErrorString(cudaGetLastError())); exit(-1); }

    	fprintf(stdout, "[EXPERIMENTAL TIME]        Indexing: t=%e\n", (double) (clock()-begin) / CLOCKS_PER_SEC);
        fprintf(stdout, "[EXPERIMENTAL BANDWIDTH]   Achieved %e GB/s\n", ((double)items_read/((double)1000*1000*1000)) /((double)(clock()-begin)/CLOCKS_PER_SEC));


        // Deallocation & cleanup for next round

        ret = cudaFree(query_mem);
        if(ret != cudaSuccess){ fprintf(stderr, "Bad free of query memory in indexing: %d\n", ret); exit(-1); }
    }

    if(write == 1)
    {
        print_kmers_to_file(table_mem, query_len_bytes, out);
        //print_kmers_to_file_paused(table_mem, query_len_bytes);
        fprintf(stdout, "[EXPERIMENTAL TIME (Including download)]        Indexing: t=%e\n", (double) (clock()-begin) / CLOCKS_PER_SEC);   
    }
    else
    {
        uint64_t * debug = (uint64_t *) malloc(query_len_bytes*sizeof(uint64_t));
        int ret = cudaMemcpy(debug, table_mem, query_len_bytes*sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if(ret != cudaSuccess){ fprintf(stderr, "DEBUG. Error: %d\n", ret); exit(-1); }
        fprintf(stdout, "[EXPERIMENTAL TIME (Including download)]        Indexing: t=%e\n", (double) (clock()-begin) / CLOCKS_PER_SEC);
    } 

    /*
    uint64_t * debug = (uint64_t *) malloc(table_size*sizeof(uint64_t));
    ret = cudaMemcpy(debug, table_mem, table_size*sizeof(uint64_t), cudaMemcpyDeviceToHost); 
    if(ret != cudaSuccess){ fprintf(stderr, "DEBUG. Error: %d\n", ret); exit(-1); }
    
    for(i=0;i<12;i++){
        fprintf(stdout, "#%"PRIu64": %"PRIu64"\n", i, debug[i]);
    } 
    */
    
    
    
    
    

    

    fclose(query);
    free(query_mem_host);

    /*


    // Wait for kernel to finish
    ret = clFlush(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad flush of event: %d\n", ret); exit(-1); }
    ret = clFinish(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad finish of event: %d\n", ret); exit(-1); }
    ret = clReleaseKernel(kernel); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (3)\n"); exit(-1); }
    ret = clReleaseProgram(program); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (4)\n"); exit(-1); }



    fprintf(stdout, "[INFO] Completed processing query splits\n");
    
    


    // Hash_item * h = (Hash_item *) malloc(hash_table_size*sizeof(Hash_item));
    // if(h == NULL) { fprintf(stderr, "Could not allocate resulting hash table\n"); exit(-1); }
    // ret = clEnqueueReadBuffer(command_queue, hash_table_mem, CL_TRUE, 0, hash_table_size*sizeof(Hash_item), h, 0, NULL, NULL);
    // if(ret != CL_SUCCESS){ fprintf(stderr, "Could not read from buffer: %d\n", ret); exit(-1); }
    // print_hash_table(h);
    

    ////////////////////////////////////////////////////////////////////////////////
    // Match hits
    ////////////////////////////////////////////////////////////////////////////////

    // Allocate memory in host
    char * ref_mem_host = (char *) malloc(ram_to_be_used * sizeof(char));
    if(ref_mem_host == NULL){ fprintf(stderr, "Could not allocate host memory for reference sequence\n"); exit(-1); }

    // Load new kernel
    kernel_temp_path[0] = '\0';
    strcat(kernel_temp_path, path_kernels);

    switch(z_value){
        case 1: { strcat(kernel_temp_path, "/kernel_match1.cl") ; read_kernel = fopen(kernel_temp_path, "r"); }
        break;
        case 4: { strcat(kernel_temp_path, "/kernel_match2.cl") ; read_kernel = fopen(kernel_temp_path, "r"); }
        break;
        case 8: { strcat(kernel_temp_path, "/kernel_match3.cl") ; read_kernel = fopen(kernel_temp_path, "r"); }
        break;
        default: { fprintf(stderr, "Could not find kernel for z=%lu.\n", z_value); exit(-1); }
        break;
    }

    if(!read_kernel){ fprintf(stderr, "Failed to load kernel (2).\n"); exit(-1); }
    source_str[0] = '\0';
    source_size = fread(source_str, 1, MAX_KERNEL_SIZE, read_kernel);
    fclose(read_kernel);

    // Create a program from the kernel source
    program = clCreateProgramWithSource(context, 1, (const char **) &source_str, (const size_t *) &source_size, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error creating program (2): %d\n", ret); exit(-1); }

    // Build the program
    ret = clBuildProgram(program, 1, &devices[selected_device], NULL, NULL, NULL);
    if(ret != CL_SUCCESS){ 
        fprintf(stderr, "Error building program (2): %d\n", ret); 
        if (ret == CL_BUILD_PROGRAM_FAILURE) {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program, devices[selected_device], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            // Allocate memory for the log
            char *log = (char *) malloc(log_size);
            // Get the log
            clGetProgramBuildInfo(program, devices[selected_device], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            // Print the log
            fprintf(stdout, "%s\n", log);
        }
        exit(-1); 
    }

    // Create the OpenCL kernel
    kernel = clCreateKernel(program, "kernel_match", &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error creating kernel (2): %d\n", ret); exit(-1); }

    begin = clock();

    // Read the reference sequence in chunks
    split = 0;
    items_read = 0;
    unsigned long ref_len_bytes = 0;
    while(!feof(ref)){

        // Load sequence chunk into ram
        items_read = fread(ref_mem_host, sizeof(char), ram_to_be_used, &ref[0]);
        ref_len_bytes += items_read;

        fprintf(stdout, "[INFO] Split #%d: %"PRIu64"\n", split, items_read);

        // Allocate ref chunk for device
        cl_mem ref_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, items_read * sizeof(char), ref_mem_host, &ret);
        if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for reference sequence in device. Error: %d\n", ret); exit(-1); }

        // Set working sizes
        global_item_size = (items_read - kmer_size + 1) / kmers_per_work_item ; // Each work item corresponds to several kmers
        global_item_size = global_item_size - (global_item_size % local_item_size); // Make it evenly divisable (yes, this makes some kmers forgotten)
	// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ HERE


	//global_item_size = 16777216;


	// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ HERE

        fprintf(stdout, "[INFO] Work items: %"PRIu64". Work groups: %"PRIu64". Total K-mers to be computed %"PRIu64"\n", (uint64_t) global_item_size, (uint64_t)(global_item_size/local_item_size), global_item_size * kmers_per_work_item);

        // Set new parameters
        Parameters params = {z_value, kmer_size, items_read, (unsigned long) global_item_size, (unsigned long) kmers_per_work_item, ref_len_bytes - items_read};    
        cl_mem params_mem_ref = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Parameters), &params, &ret);
        if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for kmer sizes variable in device. Error: %d\n", ret); exit(-1); }


        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&hash_table_mem);
        if(ret != CL_SUCCESS){ fprintf(stderr, "Bad setting of param (1): %d\n", ret); exit(-1); }
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&params_mem_ref);
        if(ret != CL_SUCCESS){ fprintf(stderr, "Bad setting of param (2): %d\n", ret); exit(-1); }
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&ref_mem);
        if(ret != CL_SUCCESS){ fprintf(stderr, "Bad setting of param (3): %d\n", ret); exit(-1); }


        fprintf(stdout, "[INFO] Executing the kernel on split %d\n", split++);

        // Execute the OpenCL kernel on the lists
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
                &global_item_size, &local_item_size, 0, NULL, NULL);
        if(ret != CL_SUCCESS){ fprintf(stderr, "Error enqueuing kernel (2): %d\n", ret); exit(-1); }


        // Wait for kernel to finish
        ret = clFlush(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad flush of event (flush): %d\n", ret); exit(-1); }
        ret = clFinish(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad finish of event (finish): %d\n", ret); exit(-1); }

        // Deallocation & cleanup for next round

        ret = clReleaseMemObject(ref_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (5)\n"); exit(-1); }
        ret = clReleaseMemObject(params_mem_ref); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (6)\n"); exit(-1); }

    }

    free(ref_mem_host);

    


    // Wait for kernel to finish
    ret = clReleaseKernel(kernel); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (2.3)\n"); exit(-1); }
    ret = clReleaseProgram(program); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (2.4)\n"); exit(-1); }



    fprintf(stdout, "[TIME] Matching: t=%e\n", (double) (clock()-begin) / CLOCKS_PER_SEC);


    ////////////////////////////////////////////////////////////////////////////////
    // Plot hits
    ////////////////////////////////////////////////////////////////////////////////


    fprintf(stdout, "[INFO] Kernel execution finished. Code = %d\n", ret);

    fprintf(stdout, "[INFO] Retrieving hash table. \n");
    Hash_item * h = (Hash_item *) malloc(hash_table_size*sizeof(Hash_item));
    if(h == NULL) { fprintf(stderr, "Could not allocate resulting hash table\n"); exit(-1); }
    ret = clEnqueueReadBuffer(command_queue, hash_table_mem, CL_TRUE, 0, hash_table_size*sizeof(Hash_item), h, 0, NULL, NULL);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not read from buffer: %d\n", ret); exit(-1); }

    */
    ret = cudaFree(table_mem);
    if(ret != cudaSuccess){ fprintf(stderr, "Bad free of k-mer table: %d\n", ret); exit(-1); }


    /*

    // Scan hits table
    fprintf(stdout, "[INFO] Scanning hits table\n");

    // Print hash table to verify it
    //FILE * right_here = fopen("tablehash", "wt");
    //FILE * right_here = fopen("tablehash", "wb");
    //FILE * right_here = fopen("tablehash", "rb");
    //print_hash_table(h, right_here);
    //fwrite(h, sizeof(hash_item), pow(4, FIXED_K), right_here);
    //fread(h, sizeof(hash_item), hash_table_size, right_here);
    //fclose(right_here);

    unsigned long idx;
    uint64_t ** representation = (uint64_t **) calloc(DIMENSION, sizeof(uint64_t *));
    //unsigned char ** m_in = (unsigned char **) calloc(DIMENSION, sizeof(unsigned char *));
    int * m_in = (int *) calloc(DIMENSION*DIMENSION, sizeof(int));
    int * m_out = (int *) calloc(DIMENSION*DIMENSION, sizeof(int));
    if(representation == NULL || m_in == NULL || m_out == NULL){ fprintf(stderr, "Could not allocate representation"); exit(-1); }
    for(idx=0; idx<DIMENSION; idx++){
        representation[idx] = (uint64_t *) calloc(DIMENSION, sizeof(uint64_t));
        if(representation[idx] == NULL){ fprintf(stderr, "Could not allocate second loop representation"); exit(-1); }
    }

    double ratio_query = (double) query_len_bytes / (double) DIMENSION;
    double ratio_ref = (double) ref_len_bytes / (double) DIMENSION;
    double pixel_size_query = (double) DIMENSION / (double) query_len_bytes;
    double pixel_size_ref = (double) DIMENSION / (double) ref_len_bytes;
    double i_r_fix = MAX(1.0, kmer_size * pixel_size_query);
    double j_r_fix = MAX(1.0, kmer_size * pixel_size_ref);

    // Output sequence lengths
    fprintf(out, "%"PRIu64"\n", query_len_bytes);
    fprintf(out, "%"PRIu64"\n", ref_len_bytes);

    for(idx=0; idx<hash_table_size; idx++){
        if(h[idx].repeat == 2 && h[idx].pos_in_y < 0xFFFFFFFFFFFFFFFF && h[idx].pos_in_x > 0){
            // Plot it  
            // Convert scale to representation
            //printf("With PX: %"PRIu64" and PY:%"PRIu64" (RX, RY) %e %e\n", h[idx].pos_in_x, h[idx].pos_in_y, ratio_ref, ratio_query);
            uint64_t redir_ref = (uint64_t) ((double)h[idx].pos_in_y / (ratio_ref));
            uint64_t redir_query = (uint64_t) ((double)h[idx].pos_in_x / (ratio_query));
            //printf("Writing at %"PRIu64", %"PRIu64"\n", redir_query, redir_ref);
            //getchar();
            double i_r = i_r_fix; double j_r = j_r_fix;
            while((uint64_t) i_r >= 1 && (uint64_t) j_r >= 1){
                if((int64_t) redir_query - (int64_t) i_r > 0 && (int64_t) redir_ref - (int64_t) j_r > 0){
                    representation[(int64_t) redir_query - (int64_t) i_r][(int64_t) redir_ref - (int64_t) j_r]++;
                }else{
                    if(redir_query > (double) DIMENSION || redir_ref > (double) DIMENSION) fprintf(stderr, "Exceeded dimension: %"PRIu64", %"PRIu64"\n", redir_query, redir_ref);
                    representation[redir_query][redir_ref]++;
                    break;
                }
                i_r -= MIN(1.0, pixel_size_query);
                j_r -= MIN(1.0, pixel_size_ref);
            }                                                     
        }
    }

    begin = clock();

    // Find number of unique diffuse hits
    // and keep only maximums
    uint64_t unique_diffuse = 0;
    unsigned long j, value = representation[0][0], pos = 0;
    for(i=0; i<DIMENSION; i++){

        for(j=0; j<DIMENSION; j++){
	        unique_diffuse += representation[i][j];

            // Find max
            if(representation[i][j] > value){
                value = representation[i][j];
                pos = j;
            }
        }

        if(value > 0){ 
            // Replace all points that are not the max
            for(j=0; j<DIMENSION; j++){
                representation[i][j] = 0;
            }
            // Set the max only
            representation[i][pos] = 1;
            m_in[i*DIMENSION+pos] = 1;
            value = 0;
        }

    }
    fprintf(stdout, "[INFO] Found %"PRIu64" unique hits for z = %"PRIu64".\n", unique_diffuse, z_value);
    //fprintf(stdout, "; %"PRIu64" %"PRIu64"\n", z_value, unique_diffuse);

    


    // Repeat for the other coordinate
    value = representation[0][0], pos = 0;
    for(i=0; i<DIMENSION; i++){

        for(j=0; j<DIMENSION; j++){


            // Find max
            if(representation[j][i] > value){
                value = representation[j][i];
                pos = j;
            }
        }
        if(value > 0){
            // Replace all points that are not the max
            for(j=0; j<DIMENSION; j++){
                representation[j][i] = 0;
            }
            // Set the max only
            representation[pos][i] = 1;
            m_in[pos*DIMENSION+i] = 1;
            value = 0;
        }

    }

    // Apply filtering kernel on m_in

    // This was to verify that there are no undesired executions resulting from concurrency
    //FILE * right_here = fopen("matrix", "wt");
    //FILE * right_here = fopen("matrix", "rb");
    //print_hash_table(h, right_here);
    //fwrite(h, sizeof(hash_item), pow(4, FIXED_K), right_here);
    //unsigned long r1;
    //for(r1=0; r1<DIMENSION*DIMENSION; r1++){
    //    fprintf(right_here, "%d\n", m_in[r1]);
    //}
    //fclose(right_here);


    // Create matrix memory object
    cl_mem m_in_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (DIMENSION)*(DIMENSION) * sizeof(int), m_in, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for image matrix. Error: %d\n", ret); exit(-1); }

    // Allocate output matrix
    cl_mem m_out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, (DIMENSION)*(DIMENSION) * sizeof(int), NULL, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for output image matrix in device. Error: %d\n", ret); exit(-1); }

    cl_mem m_dimension = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned long), &DIMENSION, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for resolution value in device. Error: %d\n", ret); exit(-1); }

    // Initialize output matrix
    int empty_int = 0;
    ret = clEnqueueFillBuffer(command_queue, m_out_mem, (const void *) &empty_int, sizeof(int), 0, (DIMENSION)*(DIMENSION) * sizeof(int), 0, NULL, NULL); 
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not initialize output matrix. Error: %d\n", ret); exit(-1); }

    // Read new kernel
    kernel_temp_path[0] = '\0';
    strcat(kernel_temp_path, path_kernels);
    strcat(kernel_temp_path, "/kernel_filter.cl");

    read_kernel = fopen(kernel_temp_path, "r");
    if(!read_kernel){ fprintf(stderr, "Failed to load kernel (3).\n"); exit(-1); }
    source_str[0] = '\0';
    source_size = fread(source_str, 1, MAX_KERNEL_SIZE, read_kernel);
    fclose(read_kernel);

    // Create a program from the kernel source
    program = clCreateProgramWithSource(context, 1, (const char **) &source_str, (const size_t *) &source_size, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error creating program (3): %d\n", ret); exit(-1); }
    
    // Build the program
    ret = clBuildProgram(program, 1, &devices[selected_device], NULL, NULL, NULL);
    if(ret != CL_SUCCESS){ 
        fprintf(stderr, "Error building program (2): %d\n", ret); 
        if (ret == CL_BUILD_PROGRAM_FAILURE) {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program, devices[selected_device], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            // Allocate memory for the log
            char *log = (char *) malloc(log_size);
            // Get the log
            clGetProgramBuildInfo(program, devices[selected_device], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            // Print the log
            fprintf(stdout, "%s\n", log);
        }
        exit(-1); 
    }

    // Create the OpenCL kernel
    kernel = clCreateKernel(program, "kernel_filter", &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error creating kernel (3): %d\n", ret); exit(-1); }

    // Set parameters

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&m_in_mem);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Bad setting of param at image filter (1): %d\n", ret); exit(-1); }
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&m_out_mem);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Bad setting of param at image filter (2): %d\n", ret); exit(-1); }
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&m_dimension);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Bad setting of param at image filter (2): %d\n", ret); exit(-1); }


    // Set working sizes
    size_t global_item_size2d[2] = {DIMENSION, DIMENSION}; 
    size_t local_item_size2d[2] = {10, 10}; 

    fprintf(stdout, "[INFO] Filtering step: Work items: %"PRIu64"x%"PRIu64". Work group size: %"PRIu64"x%"PRIu64"\n", (uint64_t) global_item_size2d[0], (uint64_t) global_item_size2d[1], (uint64_t)local_item_size2d[0], (uint64_t)local_item_size2d[1]);

    begin = clock();

    fprintf(stdout, "[INFO] Executing the kernel\n");
    // Execute the OpenCL kernel on the lists
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
            global_item_size2d, local_item_size2d, 0, NULL, NULL);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error enqueuing kernel (3): %d\n", ret); exit(-1); }

    // Wait for kernel to finish
    ret = clFlush(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Command execution went wrong (1): %d\n", ret); exit(-1); }
    ret = clFinish(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Command execution went wrong (2): %d\n", ret); exit(-1); }

    // Read resulting output matrix
    ret = clEnqueueReadBuffer(command_queue, m_out_mem, CL_TRUE, 0, (DIMENSION)*(DIMENSION) * sizeof(int), m_out, 0, NULL, NULL);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not read output matrix from buffer: %d\n", ret); exit(-1); }


    // Write resulting matrix
    for(i=0; i<DIMENSION; i++){
        for(j=0; j<DIMENSION; j++){
            fprintf(out, "%u ", (m_out[i*DIMENSION+j] > 0) ? (1) : (0) );
        }
        fprintf(out, "\n");
    }

    fprintf(stdout, "[TIME] Filtering: t=%e\n", (double) (clock()-begin) / CLOCKS_PER_SEC);

    fclose(out);

    free(h);
    
    for(j=0;j<DIMENSION;j++){
        free(representation[j]);
        //free(m_in[j]);
        //free(m_out[j]);
    }
    free(representation);
    free(m_in);
    free(m_out);


    // print_hash_table(h);
    
    // Close and deallocate everything
    
    ret = clReleaseKernel(kernel); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (3)\n"); exit(-1); }
    ret = clReleaseProgram(program); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (4)\n"); exit(-1); }
    //ret = clReleaseMemObject(ref_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (5)\n"); exit(-1); }
    //ret = clReleaseMemObject(hash_table_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (6)\n"); exit(-1); }
    //ret = clReleaseMemObject(params_mem_ref); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (7)\n"); exit(-1); }
    ret = clReleaseMemObject(m_in_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (8)\n"); exit(-1); }
    ret = clReleaseMemObject(m_out_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (8)\n"); exit(-1); }
    ret = clReleaseCommandQueue(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (8)\n"); exit(-1); }
    ret = clReleaseContext(context); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (9)\n"); exit(-1); }
    
    free(path_kernels);

    */
    

    return 0;
}


void init_args(int argc, char ** av, FILE ** query, unsigned * selected_device, FILE ** ref, FILE ** out, unsigned * write){
    
    int pNum = 0;
    char * p1 = NULL, * p2 = NULL;
    char outname[2048]; outname[0] = '\0';
    while(pNum < argc){
        if(strcmp(av[pNum], "--help") == 0){
            fprintf(stdout, "USAGE:\n");
            fprintf(stdout, "           CUDAKMER -query [file] -ref [file] -device [device]\n");
            fprintf(stdout, "OPTIONAL:\n");
            fprintf(stdout, "           --write     Enables writing output\n");
            fprintf(stdout, "           --help      Shows help for program usage\n");
            fprintf(stdout, "\n");
            exit(1);
        }
        if(strcmp(av[pNum], "--write") == 0){
            *write = 1;
        }

        if(strcmp(av[pNum], "-query") == 0){
            *query = fopen(av[pNum+1], "rt");
            if(*query==NULL){ fprintf(stderr, "Could not open query file\n"); exit(-1); }
            p1 = get_basename(av[pNum+1]);
        }
        
        if(strcmp(av[pNum], "-ref") == 0){
            *ref = fopen(av[pNum+1], "rt");
            if(*ref==NULL){ fprintf(stderr, "Could not open reference file\n"); exit(-1); }
            p2 = get_basename(av[pNum+1]);
        }

        if(strcmp(av[pNum], "-dev") == 0){
            *selected_device = (unsigned) atoi(av[pNum+1]);
            if(atoi(av[pNum+1]) < 0) { fprintf(stderr, "Device must be >0\n"); exit(-1); }
        }

        pNum++;

    }   
    
    if(*query==NULL || *ref==NULL){ fprintf(stderr, "You have to include a query and a reference sequence!\n"); exit(-1); }
    strcat(outname, p1);
    strcat(outname, "-");
    strcat(outname, p2);
    strcat(outname, ".kmers");
    *out = fopen(outname, "wt");
    if(*out == NULL){ fprintf(stderr, "Could not open output file\n"); exit(-1); }
    if(p1 != NULL) free(p1);
    if(p2 != NULL) free(p2);   
}

void perfect_hash_to_word(char * word, uint64_t hash, uint64_t k){
    /*
    int64_t jIdx = (int64_t) (k-1), upIdx = 31;
    uint64_t v;
    while(jIdx >= 0){
        v = (uint64_t) floor(hash / (pow(4, jIdx)));
        if(v == 0){ word[upIdx--] = (char) 'A'; hash -= ((uint64_t) pow(4, (uint64_t) jIdx) * 0); }
        if(v == 1){ word[upIdx--] = (char) 'C'; hash -= ((uint64_t) pow(4, (uint64_t) jIdx) * 1); }
        if(v == 2){ word[upIdx--] = (char) 'G'; hash -= ((uint64_t) pow(4, (uint64_t) jIdx) * 2); }
        if(v == 3){ word[upIdx--] = (char) 'T'; hash -= ((uint64_t) pow(4, (uint64_t) jIdx) * 3); }
        
        if(jIdx == 0) break;
        --jIdx;
    }
    */
    
    int64_t jIdx = (int64_t) (k-1), upIdx = 0;
    uint64_t v;
    while(jIdx >= 0){
        v = (uint64_t) floor(hash / (pow(4, jIdx)));
        if(v == 0){ word[upIdx++] = (char) 'A'; hash -= ((uint64_t) pow(4, (uint64_t) jIdx) * 0); }
        if(v == 1){ word[upIdx++] = (char) 'C'; hash -= ((uint64_t) pow(4, (uint64_t) jIdx) * 1); }
        if(v == 2){ word[upIdx++] = (char) 'G'; hash -= ((uint64_t) pow(4, (uint64_t) jIdx) * 2); }
        if(v == 3){ word[upIdx++] = (char) 'T'; hash -= ((uint64_t) pow(4, (uint64_t) jIdx) * 3); }
        
        if(jIdx == 0) break;
        --jIdx;
    }
    
    
}

void print_kmers_to_file(uint64_t * table_mem, uint64_t table_size, FILE * fout){
    
    uint64_t * debug = (uint64_t *) malloc(table_size*sizeof(uint64_t));
    int ret = cudaMemcpy(debug, table_mem, table_size*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if(ret != cudaSuccess){ fprintf(stderr, "DEBUG. Error: %d\n", ret); exit(-1); }
    
    uint64_t i;
    char word[KMER_SIZE+1];
    for(i=0;i<table_size;i++){
        perfect_hash_to_word(word, debug[i], KMER_SIZE);
        word[KMER_SIZE] = '\0';
        fprintf(fout, "#%"PRIu64", %s\n", i, word);
        fprintf(fout, "#%"PRIu64", %"PRIu64"\n", i, debug[i]);
    } 

    fclose(fout);
}

void print_kmers_to_file_paused(uint64_t * table_mem, uint64_t table_size){
    
    uint64_t * debug = (uint64_t *) malloc(table_size*sizeof(uint64_t));
    int ret = cudaMemcpy(debug, table_mem, table_size*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if(ret != cudaSuccess){ fprintf(stderr, "DEBUG. Error: %d\n", ret); exit(-1); }
    
    uint64_t i;
    char word[KMER_SIZE+1];
    for(i=3060000;i<table_size;i++){
        perfect_hash_to_word(word, debug[i], KMER_SIZE);
        word[KMER_SIZE] = '\0';
        fprintf(stdout, "#%"PRIu64", %s\n", i, word);
        fprintf(stdout, "#%"PRIu64", %"PRIu64"\n", i, debug[i]);
        if(i % 20 == 0) getchar();
    } 

}



char * get_dirname(char * path){
    int pos_last = 0, i = 0;
    while(path[i] != '\0'){
        if(path[i] == '/'){
            pos_last = i;
        }
        ++i;
    }
    char * dirname = (char *) malloc(BUFFER_SIZE * sizeof(char));
    if(dirname == NULL){ fprintf(stderr, "Could not allocate dirname char\n"); exit(-1); }

    memcpy(&dirname[0], &path[0], pos_last);
    dirname[pos_last] = '\0';

    return dirname;
}

char * get_basename(char * path){
    char * s = strrchr(path, '/');
    if (!s) return strdup(path); else return strdup(s + 1);
}
