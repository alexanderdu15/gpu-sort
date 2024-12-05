#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <type_traits>
#include "single_thread_merge.cuh"
#include "parallel_merge_naive.cuh"
#include "parallel_merge_shared_mem.cuh"

enum class SortMethod {
    SINGLE_THREAD,
    PARALLEL_NAIVE,
    PARALLEL_SHARED_MEM
};

template<typename T>
class GPUSorter {
private:
    T* h_arr;               // Host array
    T* d_arr;               // Device array
    T* d_temp;              // Temporary device array for merging
    size_t size;            // Size of the array
    float total_time;       // Total time including transfers
    float kernel_time;      // Kernel execution time only
    SortMethod method;      // Sorting method

public:
    GPUSorter(size_t array_size, SortMethod sort_method = SortMethod::PARALLEL_NAIVE) 
        : size(array_size), method(sort_method) {
        h_arr = new T[size];
        cudaMalloc(&d_arr, size * sizeof(T));
        cudaMalloc(&d_temp, size * sizeof(T));
    }

    ~GPUSorter() {
        delete[] h_arr;
        cudaFree(d_arr);
        cudaFree(d_temp);
    }

    void generateRandomData() {
        generateRandomDataImpl(std::is_integral<T>());
    }

    void sort() {
        cudaEvent_t start, stop, kernel_start, kernel_stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&kernel_start);
        cudaEventCreate(&kernel_stop);

        cudaEventRecord(start);
        cudaMemcpy(d_arr, h_arr, size * sizeof(T), cudaMemcpyHostToDevice);

        cudaEventRecord(kernel_start);
        if (method == SortMethod::PARALLEL_NAIVE) {
            launchParallelSortNaive();
        }else if (method == SortMethod::PARALLEL_SHARED_MEM) {
            launchParallelSortSharedMem();
        }else {
            launchSingleThreadSort();
        } // TODO: add other methods
        cudaEventRecord(kernel_stop);

        cudaMemcpy(h_arr, d_arr, size * sizeof(T), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&total_time, start, stop);
        cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_stop);
    }

private:
    void launchParallelSortNaive() {
        int threads_per_block = 256; // TODO: try different block sizes?
        int num_blocks = (size + threads_per_block - 1) / threads_per_block; // round up so all elements are sorted
        
        // Local sort within blocks
        parallelMergeSortNaiveKernel<T><<<num_blocks, threads_per_block>>>(d_arr, d_temp, size);
        
        // Merge across blocks, need this becuase we can't sync threads across blocks
        for (uint32_t stride = threads_per_block; stride < size; stride *= 2) {
            uint32_t merge_blocks = (size + (2 * stride) - 1) / (2 * stride); // number of blocks needed to merge
            if (merge_blocks > 0) {
                mergeBlocksNaiveKernel<T><<<merge_blocks, threads_per_block>>>(d_arr, d_temp, size, stride);
            }
        }
    }

    void launchParallelSortSharedMem() {
        // Used shared memory for intra-block merge
        // Used naive version for inter-block merge
        int threads_per_block = 256;
        int num_blocks = (size + threads_per_block - 1) / threads_per_block; // round up so all elements are sorted
        
        // Local sort within blocks
        parallelMergeSortSharedMemKernel<T><<<num_blocks, threads_per_block, 2*sizeof(T)*(size+num_blocks-1)/num_blocks>>>(d_arr, d_temp, size);
        
        // Merge across blocks, need this becuase we can't sync threads across blocks
        for (uint32_t stride = threads_per_block; stride < size; stride *= 2) {
            uint32_t merge_blocks = (size + (2 * stride) - 1) / (2 * stride); // number of blocks needed to merge
            if (merge_blocks > 0) {
                mergeBlocksNaiveKernel<T><<<merge_blocks, threads_per_block>>>(d_arr, d_temp, size, stride);
            }
        }
    }

    void launchSingleThreadSort() {
        // Launch with 1 thread and 1 block
        singleThreadSortKernel<T><<<1, 1>>>(d_arr, d_temp, size);
    }

    // Helper functions for different types
    void generateRandomDataImpl(std::true_type) {
        // For integer types
        for (uint32_t i = 0; i < size; i++) {
            h_arr[i] = rand() % 1000;
        }
    }

    void generateRandomDataImpl(std::false_type) {
        // For floating point types
        for (uint32_t i = 0; i < size; i++) {
            h_arr[i] = static_cast<T>(rand()) / RAND_MAX * 1000.0;
        }
    }

public:
    bool checkResult() const {
        bool is_sorted = true;
        for (uint32_t i = 0; i < size - 1; i++) {
            if (h_arr[i] > h_arr[i+1]) {
                is_sorted = false;
                break;
            }
        }
        return is_sorted;
    }

    const T* getSortedArray() const {
        return h_arr;
    }

    float getTotalTime() const { return total_time; }
    float getKernelTime() const { return kernel_time; }

    T* getData() { return h_arr; }
};