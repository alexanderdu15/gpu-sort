template<typename T>
__device__ void singleThreadMerge(T *arr, T *temp, int left, int mid, int right) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    while (i <= mid) {
        temp[k++] = arr[i++];
    }
    while (j <= right) {
        temp[k++] = arr[j++];
    }
    for (int p = left; p <= right; p++) {
        arr[p] = temp[p];
    }
}

template<typename T>
__global__ void singleThreadSortKernel(T *arr, T *temp, int size) {
    for (int curr_size = 1; curr_size < size; curr_size *= 2) {
        for (int left_start = 0; left_start < size - 1; left_start += 2*curr_size) {
            int mid = min(left_start + curr_size - 1, size - 1);
            int right_end = min(left_start + 2*curr_size - 1, size - 1);
            singleThreadMerge<T>(arr, temp, left_start, mid, right_end);
        }
        
        T *swap_temp = arr;
        arr = temp;
        temp = swap_temp;
    }
}
