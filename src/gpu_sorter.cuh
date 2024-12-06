#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <type_traits>
#include "single_thread_merge.cuh"
#include "parallel_merge_naive.cuh"
#include "parallel_merge_shared.cuh"
#include <cuda_runtime.h>

enum class SortMethod {
    SINGLE_THREAD,
    PARALLEL_NAIVE,
    PARALLEL_SHARED
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
    int threads_per_block;  // Store the block size

public:
    GPUSorter(size_t array_size, SortMethod sort_method = SortMethod::PARALLEL_NAIVE, int block_size = 256) 
        : size(array_size), method(sort_method), threads_per_block(block_size) {
        h_arr = new T[size];
        cudaMalloc(&d_arr, size * sizeof(T));
        cudaMalloc(&d_temp, size * sizeof(T));
    }

    ~GPUSorter() {
        delete[] h_arr;
        cudaFree(d_arr);
        cudaFree(d_temp);
    }

    void checkMemoryUsage(const char *message) {
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);

        float freeMemMB = freeMem / (1024.0 * 1024.0);
        float totalMemMB = totalMem / (1024.0 * 1024.0);
        float usedMemMB = totalMemMB - freeMemMB;

        std::cout << message << "\n";
        std::cout << "Used Memory: " << usedMemMB << " MB\n";
        std::cout << "Free Memory: " << freeMemMB << " MB\n";
        std::cout << "Total Memory: " << totalMemMB << " MB\n\n";
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
        checkMemoryUsage("Before kernel execution");
        if (method == SortMethod::SINGLE_THREAD) {
            launchSingleThreadSort();
        } else if (method == SortMethod::PARALLEL_NAIVE) {
            launchParallelSortNaive(threads_per_block);
        } else if (method == SortMethod::PARALLEL_SHARED) {
            launchParallelSortShared(threads_per_block);
        }
        checkMemoryUsage("After kernel execution");
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
    void launchSingleThreadSort() {
        // Launch with 1 thread and 1 block
        singleThreadSortKernel<T><<<1, 1>>>(d_arr, d_temp, size);
    }

    void launchParallelSortNaive(int threads_per_block) {
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

    void launchParallelSortShared(int threads_per_block) {
        int num_blocks = (size + threads_per_block - 1) / threads_per_block; // round up so all elements are sorted
        int elements_per_block = (size + num_blocks - 1) / num_blocks;

        // Local sort within blocks
        size_t shared_mem_size = 2 * elements_per_block * sizeof(T);
        parallelMergeSortSharedKernel<T><<<num_blocks, threads_per_block, shared_mem_size>>>(d_arr, d_temp, size);

        // Merge across blocks, need this becuase we can't sync threads across blocks
        for (uint32_t stride = threads_per_block; stride < size; stride *= 2) {
            uint32_t merge_blocks = (size + (2 * stride) - 1) / (2 * stride); // number of blocks needed to merge
            if (merge_blocks > 0) {
                mergeBlocksSharedKernel<T><<<merge_blocks, threads_per_block>>>(d_arr, d_temp, size, stride);
            }
        }
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