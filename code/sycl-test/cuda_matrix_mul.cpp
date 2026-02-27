#include <iostream>
#include <cuda_runtime.h>

// 1. THE KERNEL (Device Code)
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int K, int N) {
    // Calculate global row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            // A is M x K, B is K x N
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 2. THE HOST CODE (CPU)
int main() {
    int M = 512, K = 512, N = 512;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Pointers for Host memory
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // Initialize matrices with dummy data
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 2.0f;

    // Pointers for Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy data from Host to Device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Define Block and Grid dimensions
    // We use 16x16 blocks (256 threads per block)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    // Copy result back to Host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "Done! Element [0,0] is: " << h_C[0] << " (Expected: " << K * 2.0f << ")" << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}