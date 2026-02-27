#include <sycl/sycl.hpp>
#include <iostream>

// This struct replaces the lambda
struct MatrixMulFunctor {
    float *A, *B, *C;
    int M, K, N;

    // Constructor to "capture" the data
    MatrixMulFunctor(float* a, float* b, float* c, int m, int k, int n)
        : A(a), B(b), C(c), M(m), K(k), N(n) {}

    // The operator() is the actual kernel code
    void operator()(sycl::id<2> index) const {
        int row = index[0];
        int col = index[1];

        if (row < M && col < N) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
};


int main() {
    int M = 512, K = 512, N = 512;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float* h_A = static_cast<float*>(malloc(sizeA));
    float* h_B = static_cast<float*>(malloc(sizeB));
    float* h_C_1 = static_cast<float*>(malloc(sizeC));
    float* h_C_2 = static_cast<float*>(malloc(sizeC));

    // Initialize matrices with dummy data
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 2.0f;
    
    // Create a SYCL queue
    sycl::queue q;
    std::cout << "Running on: " 
              << q.get_device().get_info<sycl::info::device::name>() 
              << std::endl;
    
    // Allocat device memory and copy data to device
    float* d_A = sycl::malloc_device<float>(M * K, q);
    float* d_B = sycl::malloc_device<float>(K * N, q);
    float* d_C_1 = sycl::malloc_device<float>(M * N, q);
    float* d_C_2 = sycl::malloc_device<float>(M * N, q);

    // Initialize device memory
    q.memcpy(d_A, h_A, sizeA);
    q.memcpy(d_B, h_B, sizeB).wait();

    // Launch the kernel with a lambda
    q.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> index) {
        int row = index[0];
        int col = index[1];
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += d_A[row * K + i] * d_B[i * N + col];
        }
        d_C_1[row * N + col] = sum;
    });

    // Launch the kernel with a functor
    q.submit([&](sycl::handler& h) {
        MatrixMulFunctor functor(d_A, d_B, d_C_2, M, K, N);
        h.parallel_for(sycl::range<2>(M, N), functor);
    }).wait();

    // Launch the kernel with a name
    q.submit([&](sycl::handler& h) {
        h.parallel_for<class MatrixCheck>(sycl::range<2>(M, N), [=](sycl::id<2> index) {
            int row = index[0];
            int col = index[1];
            for (int i = 0; i < K; ++i) {
                if (d_C_1[row * N + col] != d_C_2[row * N + col]) {
                    printf("Error at [%d, %d]: %f\n", row, col, d_C_1[row * N + col]);
                }
            }
        });
    }).wait();


    // Copy result back to host
    q.memcpy(h_C_1, d_C_1, sizeC);
    q.memcpy(h_C_2, d_C_2, sizeC).wait();

    std::cout << "Done! Element [0,0] is: " << h_C_1[0] << " (Expected: " << K * 2.0f << ")" << std::endl;
    std::cout << "Done! Element [0,0] is: " << h_C_2[0] << " (Expected: " << K * 2.0f << ")" << std::endl;

    // Cleanup
    free(d_A, q); // USM free requires the queue
    free(d_B, q);
    free(d_C_1, q);
    free(d_C_2, q);
    free(h_A);
    free(h_B);
    free(h_C_1);
    free(h_C_2);

    return 0;

}