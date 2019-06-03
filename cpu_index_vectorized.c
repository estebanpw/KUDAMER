
// Standard utilities and common systems includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <ctype.h>

#define MAX_VECTOR_SIZE 512

#include "vcl/vectorclass.h"

#define BUFFER_SIZE 2048
#define KMER_SIZE 32
#define TABLE_SIZE 200*1000*1000
//#define DIMENSION 1000


static uint64_t pow4[33]={1L, 4L, 16L, 64L, 256L, 1024L, 4096L, 16384L, 65536L,
    262144L, 1048576L, 4194304L, 16777216L, 67108864L, 268435456L, 1073741824L, 4294967296L,
    17179869184L, 68719476736L, 274877906944L, 1099511627776L, 4398046511104L, 17592186044416L,
    70368744177664L, 281474976710656L, 1125899906842624L, 4503599627370496L, 18014398509481984L,
    72057594037927936L, 288230376151711744L, 1152921504606846976L, 4611686018427387904L};

static uint64_t pow4_G[33]={2*1L, 2*4L, 2*16L, 2*64L, 2*256L, 2*1024L, 2*4096L, 2*16384L, 2*65536L,
    (uint64_t)2*262144L, (uint64_t)2*1048576L,(uint64_t)2*4194304L, (uint64_t)2*16777216L, (uint64_t)2*67108864L, (uint64_t)2*268435456L, (uint64_t)2*1073741824L, (uint64_t)2*4294967296L,
    (uint64_t)2*17179869184L, (uint64_t)2*68719476736L, (uint64_t)2*274877906944L, (uint64_t)2*1099511627776L, (uint64_t)2*4398046511104L, (uint64_t)2*17592186044416L,
    (uint64_t)2*70368744177664L, (uint64_t)2*281474976710656L, (uint64_t)2*1125899906842624L, (uint64_t)2*4503599627370496L, (uint64_t)2*18014398509481984L,
    (uint64_t)2*72057594037927936L, (uint64_t) 2*288230376151711744L, (uint64_t) 2*1152921504606846976L, (uint64_t) 2*4611686018427387904L};

static uint64_t pow4_T[33]={3*1L, 3*4L, 3*16L, 3*64L, 3*256L, 3*1024L, 3*4096L, 3*16384L, 3*65536L,
    (uint64_t)3*262144L, (uint64_t) 3*1048576L, (uint64_t)3*4194304L, (uint64_t)3*16777216L, (uint64_t)3*67108864L, (uint64_t)3*268435456L, (uint64_t)3*1073741824L, (uint64_t)3*4294967296L,
    (uint64_t)3*17179869184L, (uint64_t)3*68719476736L, (uint64_t)3*274877906944L, (uint64_t)3*1099511627776L, (uint64_t)3*4398046511104L, (uint64_t)3*17592186044416L,
    (uint64_t)3*70368744177664L, (uint64_t)3*281474976710656L, (uint64_t)3*1125899906842624L, (uint64_t)3*4503599627370496L, (uint64_t)3*18014398509481984L,
    (uint64_t)3*72057594037927936L, (uint64_t) 3*288230376151711744L, (uint64_t) 3*1152921504606846976L, (uint64_t) 3*4611686018427387904L};

Vec4uq powers[8];

/*powers[1] = Vec4uq(256L, 1024L, 4096L, 16384L);
powers[2] = Vec4uq(65536L, 262144L, 1048576L, 4194304L);
powers[3] = Vec4uq(16777216L, 67108864L, 268435456L, 1073741824L);
powers[4] = Vec4uq(4294967296L, 17179869184L, 68719476736L, 274877906944L);
powers[5] = Vec4uq(1099511627776L, 4398046511104L, 17592186044416L, 70368744177664L);
powers[6] = Vec4uq(281474976710656L, 1125899906842624L, 4503599627370496L, 18014398509481984L);
powers[7] = Vec4uq(72057594037927936L, 288230376151711744L, 1152921504606846976L, 4611686018427387904L);*/




////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void init_args(int argc, char ** av, FILE ** query, FILE ** ref, FILE ** out, unsigned * write);
void perfect_hash_to_word(char * word, uint64_t hash, uint64_t k);
uint64_t quick_pow4byLetter(uint64_t n, const char c);
uint64_t fast_hash_from_previous(const unsigned char * word, uint64_t k, unsigned char next_nucl, uint64_t previous_hash);
void print_kmers_to_file(uint64_t * table_mem, uint64_t table_size, FILE * fout);
char * get_dirname(char * path);
char * get_basename(char * path);
uint64_t hash_of_word_the_old_way(uint64_t kmer_size, unsigned char * word);
uint64_t hash_of_word_the_oldest_way_possible(uint64_t kmer_size, unsigned char * word);
void compute_kmers(char * sequence, uint64_t * table_mem, uint64_t len);

int main(int argc, char ** argv)
{

    powers[0].load(pow4);
    powers[1].load(pow4+4);
    powers[2].load(pow4+8);
    powers[3].load(pow4+12);
    powers[4].load(pow4+16);
    powers[5].load(pow4+20);
    powers[6].load(pow4+24);
    powers[7].load(pow4+28);

    uint64_t i;
    unsigned write = 0;
    FILE * query = NULL, * ref = NULL, * out = NULL;
    init_args(argc, argv, &query, &ref, &out, &write);

    int ret;
    
    ////////////////////////////////////////////////////////////////////////////////
    // Make index dictionary
    ////////////////////////////////////////////////////////////////////////////////

    uint64_t query_len_bytes = 0;


    // Allocate hash table
    uint64_t * table_mem = NULL;
    table_mem = (uint64_t *) calloc(TABLE_SIZE, sizeof(uint64_t));
    if(table_mem == NULL) { fprintf(stderr, "Could not allocate hash table\n"); exit(-1); }    
    
    // Allocate memory in host for sequence chunk
    char * query_mem_host = (char *) malloc(TABLE_SIZE * sizeof(char));
    if(query_mem_host == NULL){ fprintf(stderr, "Could not allocate host memory for query sequence\n"); exit(-1); }
   
    
    clock_t begin;
    // Pointer to device memory allocating the query sequence
    char * query_mem = NULL;

    

    // Read the input query in chunks
    int split = 0;
    uint64_t items_read = 0;
    while(!feof(query)){

        // Load sequence chunk into ram
        items_read = fread(query_mem_host, sizeof(char), TABLE_SIZE, &query[0]);
        query_len_bytes += items_read;

        
        // Set global working sizes
        fprintf(stdout, "[INFO] Split #%d: %"PRIu64"\n", split, items_read);
        
        fprintf(stdout, "[INFO] Executing the kernel on split %d\n", split++);

        // Execute the CUDA kernel on the data
        begin = clock();
        

        // Algorithm goes here
        compute_kmers(query_mem_host, table_mem, items_read);


    	fprintf(stdout, "[EXPERIMENTAL TIME]        Indexing: t=%e\n", (double) (clock()-begin) / CLOCKS_PER_SEC);
        fprintf(stdout, "[EXPERIMENTAL BANDWIDTH]   Achieved %e GB/s\n", ((double)items_read/((double)1000*1000*1000)) /((double)(clock()-begin)/CLOCKS_PER_SEC));
    }

    if(write == 1) print_kmers_to_file(table_mem, query_len_bytes, out);

    
    
    
    

    

    fclose(query);
    free(query_mem_host);
    free(table_mem);    
    

    return 0;
}


void init_args(int argc, char ** av, FILE ** query, FILE ** ref, FILE ** out, unsigned * write){
    
    int pNum = 0;
    char * p1 = NULL, * p2 = NULL;
    char outname[2048]; outname[0] = '\0';
    while(pNum < argc){
        if(strcmp(av[pNum], "--help") == 0){
            fprintf(stdout, "USAGE:\n");
            fprintf(stdout, "           CUDAKMER -query [file] -ref [file]\n");
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
    /*
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
    */
    
}

void print_kmers_to_file(uint64_t * table_mem, uint64_t table_size, FILE * fout){
    
        
    uint64_t i;
    char word[KMER_SIZE+1];
    for(i=0;i<table_size;i++){
        perfect_hash_to_word(word, table_mem[i], KMER_SIZE);
        word[KMER_SIZE] = '\0';
        fprintf(fout, "#%"PRIu64", %s\n", i, word);
        fprintf(fout, "#%"PRIu64", %"PRIu64"\n", i, table_mem[i]);
    } 

    fclose(fout);
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

uint64_t quick_pow4byLetter(uint64_t n, const char c){
    
    if(c == 'C') return pow4[n];
    if(c == 'G') return pow4_G[n];
    if(c == 'T') return pow4_T[n];
    return 0;
}

uint64_t fast_hash_from_previous(const unsigned char * word, uint64_t k, unsigned char next_nucl, uint64_t previous_hash){
    
    if((char) next_nucl == 'A') return 4 * (previous_hash - quick_pow4byLetter(k-1, (char) next_nucl));
    if((char) next_nucl == 'C') return 4 * (previous_hash - quick_pow4byLetter(k-1, (char) next_nucl)) + 1;
    if((char) next_nucl == 'G') return 4 * (previous_hash - quick_pow4byLetter(k-1, (char) next_nucl)) + 2;
    return 4 * (previous_hash - quick_pow4byLetter(k-1, (char) next_nucl)) + 3;
    
    /*
    if((char) next_nucl == 'A') return 4 * (previous_hash - quick_pow4byLetter(k, (char) next_nucl));
    if((char) next_nucl == 'C') return 4 * (previous_hash - quick_pow4byLetter(k, (char) next_nucl)) + 1;
    if((char) next_nucl == 'G') return 4 * (previous_hash - quick_pow4byLetter(k, (char) next_nucl)) + 2;
    return 4 * (previous_hash - quick_pow4byLetter(k, (char) next_nucl)) + 3;
    */
}

uint64_t fast_hash_from_previous_natural_order(const unsigned char * word, uint64_t k, unsigned char next_nucl, uint64_t previous_hash){
    if((char) next_nucl == 'A') return ((previous_hash - 0)/4 + quick_pow4byLetter(k-1, (char) next_nucl));
    if((char) next_nucl == 'C') return ((previous_hash - 1)/4 + quick_pow4byLetter(k-1, (char) next_nucl));
    if((char) next_nucl == 'G') return ((previous_hash - 2)/4 + quick_pow4byLetter(k-1, (char) next_nucl));
    return ((previous_hash - 3)/4 + quick_pow4byLetter(k-1, (char) next_nucl));
}

uint64_t hash_of_word_the_old_way(uint64_t kmer_size, unsigned char * word){
    uint64_t i;
    uint64_t hash = 0;
    for(i=0; i<kmer_size; i++){
        hash = hash + quick_pow4byLetter((kmer_size-1)-i, word[i]);
    }
    return hash;
}

uint64_t hash_of_word_the_oldest_way_possible(uint64_t kmer_size, unsigned char * word){
    uint64_t i;
    uint64_t hash = 0;
    for(i=0; i<kmer_size; i++){
        if(word[i] == 'A') hash = hash + powl(4, (kmer_size-1)-i) * 0;
        if(word[i] == 'C') hash = hash + powl(4, (kmer_size-1)-i) * 1;
        if(word[i] == 'G') hash = hash + powl(4, (kmer_size-1)-i) * 2;
        if(word[i] == 'T') hash = hash + powl(4, (kmer_size-1)-i) * 3;
    }
    return hash;
}

uint64_t hash_old_way_but_vectorized(uint64_t kmer_size, unsigned char * word, uint64_t * char_values){
    Vec4uq values[8];
    Vec4uq hashes[8];

    uint64_t h = 0;
    
    for(uint64_t i=0; i<8; i++)
    {
        values[i].load(char_values+(i*4));
        hashes[i] = values[i] * powers[i];
        h += horizontal_add(hashes[i]);
    }

    return h;
}

void compute_kmers(char * sequence, uint64_t * table_mem, uint64_t len){

    char c; //Char to read character
    uint64_t pos = 0, word_size = 0, hash = 0;
    unsigned char curr_kmer[KMER_SIZE], first_time = 1;
    uint64_t char_values[KMER_SIZE]; 
    memset(char_values, 0x0, KMER_SIZE * sizeof(uint64_t));
    
    while (pos < len)
    {

        c = sequence[pos];
        c = toupper(c);

        if (c != 'N')
        {
            curr_kmer[word_size] = (unsigned char) c;
            if(c == 'A') char_values[word_size] = 0;
            if(c == 'C') char_values[word_size] = 1;
            if(c == 'G') char_values[word_size] = 2;
            if(c == 'T') char_values[word_size] = 3;
            
            if (word_size < KMER_SIZE-1 || first_time == 1)
            {
                //hash = hash + quick_pow4byLetter((KMER_SIZE-1)-word_size, c);
                ++word_size;
            }
            else
            {
                
                //hash = fast_hash_from_previous(curr_kmer, KMER_SIZE, c, hash);
                ++word_size;
            }
            
            
            if (word_size == KMER_SIZE)
            {
                
                //table_mem[pos - (KMER_SIZE-1)] = hash_of_word_the_oldest_way_possible(KMER_SIZE, curr_kmer);
                table_mem[pos - (KMER_SIZE-1)] = hash_old_way_but_vectorized(KMER_SIZE, curr_kmer, char_values);
                //memmove(&curr_kmer[0], &curr_kmer[1], KMER_SIZE-1);
                //memmove(&char_values[0], &char_values[1], (KMER_SIZE-1) * sizeof(uint64_t));
                //--word_size;

                word_size = 0;
                

                /*
                table_mem[pos - (KMER_SIZE-1)] = hash; 
                --word_size;
                first_time = 0;
                */
                
                
            }
            
        }
        else
        { 
            word_size = 0;
            first_time = 1;
            hash = 0;
        }

        ++pos;
        
    }
}