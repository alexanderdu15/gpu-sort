// Merge two sorted sequences within a thread block
template<typename T>
__device__ void blockMergeNaive(T *arr, T *temp, int left, int mid, int right) {
    int tid = threadIdx.x;
    int total_elements = right - left + 1;
    
    int elements_per_thread = (total_elements + blockDim.x - 1) / blockDim.x;
    int start = left + tid * elements_per_thread;
    int end = min(start + elements_per_thread, right + 1);
    
    int i = start;
    int j = mid + 1;
    int k = start;
    
    while (i <= mid && j <= right && k < end) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    while (i <= mid && k < end) {
        temp[k++] = arr[i++];
    }
    
    while (j <= right && k < end) {
        temp[k++] = arr[j++];
    }
    
    __syncthreads();
    
    for (int p = start; p < end; p++) {
        arr[p] = temp[p];
    }
}

template<typename T>
__global__ void parallelMergeSortNaiveKernel(T *arr, T *temp, int size) {
    // Calculate the portion of the array this block will handle
    int elements_per_block = (size + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * elements_per_block;
    int block_end = min(block_start + elements_per_block - 1, size - 1);

    // Local sort within block
    for (int curr_size = 1; curr_size < (block_end - block_start + 1); curr_size *= 2) {
        for (int left_start = block_start; left_start < block_end; left_start += 2*curr_size) {
            int mid = min(left_start + curr_size - 1, block_end);
            int right_end = min(left_start + 2*curr_size - 1, block_end);
            blockMergeNaive(arr, temp, left_start, mid, right_end);
        }
        
        __syncthreads();
    }
}

// Merge across blocks
template<typename T>
__global__ void mergeBlocksNaiveKernel(T *arr, T *temp, int size, int stride) {
    int block_offset = blockIdx.x * 2 * stride;
    int left = block_offset;
    int mid = min(left + stride - 1, size - 1);
    int right = min(left + 2 * stride - 1, size - 1);
    
    if (mid < right) {
        blockMergeNaive(arr, temp, left, mid, right);
    }
} 