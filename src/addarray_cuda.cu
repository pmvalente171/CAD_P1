#include <cadlabs.hpp>
#include <marrow/timer.hpp>

using namespace std;

__global__ void add(const int* a, const int* b, int* c, const unsigned size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
        c[index] = a[index] + b[index];
}


int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "usage " << argv[0] << " array_size \n";
       return 1;
    }

    const auto size = std::stoi( argv[1] );
    const auto size_in_bytes = size * sizeof(int);

    // allocate in the heap, because there may not be enough space in the stack
    const unique_ptr<vector<int>> a = make_unique<vector<int>>(size, 1);
    // vector<int> *a = new vector<int>(size, 1)
    const unique_ptr<vector<int>> b = make_unique<vector<int>>(size, 2);
    unique_ptr<vector<int>> c = make_unique<vector<int>>(size);

    marrow::timer<> t;
    t.start();
    // allocate in the GPU
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size_in_bytes);
    cudaMalloc((void **)&d_b, size_in_bytes);
    cudaMalloc((void **)&d_c, size_in_bytes);

    // Copy to the GPU
    cudaMemcpy(d_a, a->data(), size_in_bytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data(), size_in_bytes,  cudaMemcpyHostToDevice);

    // Perform the computation
    const auto nb = (size + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    add<<<nb, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, size);

    // Copy the results back to the host
    cudaMemcpy(c->data(), d_c, size_in_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    t.stop();

    t.output_stats(cout);
    cout << " milliseconds\n ";

    // check if the result is the one expected
    unique_ptr<vector<int>> expected = make_unique<vector<int>>(size, 3);
    expect_container_eq(*c, *expected);

    return 0;
}