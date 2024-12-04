# GPU Merge Sort

## Usage
```
make
make run
```
Results are saved to `results/`

## Experiments performed with the following system:
- Intel(R) Core(TM) i9-14900KF (32) @ 6.00 GHz
- NVIDIA GeForce RTX 4090 24GB VRAM
- Ubuntu jammy 22.04 x86_64

## Merge sort
- Repeatedly merge smaller sorted subarrays into larger sorted subarrays
- Outer loop:
    - Doubles the size of the subarrays to merge each iteration (1 -> 2 -> 4 -> 8 -> ...)
- Inner loop:
    - Iterates over pairs of subarrays and merges them using

## Additional things to try
- Use shared memory
- Cooperative groups
- Warp level primitives
- Memory coalescing
- Tensor cores
- Different block sizes
- Try more array sizes
- Bitonic sort

