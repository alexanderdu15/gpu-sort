// Merge two sorted sequences within a thread block
template<typename T>
__device__ void blockMergeShared(T *shared_arr, T *shared_temp, int left, int mid, int right) {
    // Convert to local indices (relative to shared memory)
    int local_left = left;
    int local_mid = mid;
    int local_right = right;
    
    int tid = threadIdx.x;
    int total_elements = local_right - local_left + 1;
    
    int elements_per_thread = (total_elements + blockDim.x - 1) / blockDim.x;
    int start = local_left + tid * elements_per_thread;
    int end = min(start + elements_per_thread, local_right + 1);
    
    int i = start;
    int j = local_mid + 1;
    int k = start;
    
    // Merge the portions this thread is responsible for
    while (i <= local_mid && j <= local_right && k < end) {
        if (shared_arr[i] <= shared_arr[j]) {
            shared_temp[k++] = shared_arr[i++];
        } else {
            shared_temp[k++] = shared_arr[j++];
        }
    }
    
    while (i <= local_mid && k < end) {
        shared_temp[k++] = shared_arr[i++];
    }
    
    while (j <= local_right && k < end) {
        shared_temp[k++] = shared_arr[j++];
    }
    
    __syncthreads();
    
    // Copy back from temp to shared array
    #pragma unroll
    for (int p = start; p < end; p++) {
        shared_arr[p] = shared_temp[p];
    }
    __syncthreads();
}

template<typename T>
__global__ void parallelMergeSortSharedKernel(T *arr, T *temp, int size) {
    // Calculate the portion of the array this block will handle
    int elements_per_block = (size + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * elements_per_block;
    int block_end = min(block_start + elements_per_block - 1, size - 1);

    extern __shared__ T shared_mem[];
    T* shared_arr = (T*)shared_mem;
    T* shared_temp = shared_arr + elements_per_block;

    if (threadIdx.x < elements_per_block) {
        // Cooperatively load global data into shared memory
        shared_arr[threadIdx.x] = arr[block_start + threadIdx.x];
        __syncthreads();

        // Local sort within block
        #pragma unroll 8
        for (int curr_size = 1; curr_size <= (block_end - block_start + 1); curr_size *= 2) {
            #pragma unroll 4
            for (int left_start = 0; left_start < elements_per_block; left_start += 2*curr_size) {
                int mid = min(left_start + curr_size - 1, elements_per_block - 1);
                int right_end = min(left_start + 2*curr_size - 1, elements_per_block - 1);
                
                // Use local indices for shared memory operations
                blockMergeShared(shared_arr, shared_temp, left_start, mid, right_end);
            }
            __syncthreads();
        }

        // Write back to global memory
        __syncthreads();
        arr[block_start + threadIdx.x] = shared_arr[threadIdx.x];
    }
}

// Merge across blocks
template<typename T>
__global__ void mergeBlocksSharedKernel(T *arr, T *temp, int size, int stride) {
    int block_offset = blockIdx.x * 2 * stride;
    int left = block_offset;
    int mid = min(left + stride - 1, size - 1);
    int right = min(left + 2 * stride - 1, size - 1);
    
    if (mid < right) {
        blockMergeShared(arr, temp, left, mid, right);
    }
} 