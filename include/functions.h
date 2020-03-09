//
// Created by Hervé Paulino on 06/03/18.
//

#ifndef CADLABS_FUNCTIONS_H
#define CADLABS_FUNCTIONS_H

#include <vector>

namespace cad {

    template <typename T>
    __global__ void add(const T* a, const T* b, T* c, const unsigned size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < size)
            c[index] = a[index] + b[index];
    }

    /**
     * Pairwise addition of two containers
     * @tparam Stride Stride between accesses to the containers. Default is 1.
     * @tparam Container Type of the containers. Assumed to be the same for all three arguments
     * @param result The container to received the addition
     * @param a The first operand
     * @param b The second operand
     */
    template<typename Container>
    void array_add(Container& result, Container& a, Container& b) {

        static constexpr auto THREADS_PER_BLOCK = 512;
        using ValueType = typename Container::value_type;

        ValueType *d_a, *d_b, *d_result;
        const auto size_in_bytes = result.size() * sizeof(ValueType);

        cudaMalloc((void **)&d_a, size_in_bytes);
        cudaMalloc((void **)&d_b, size_in_bytes);
        cudaMalloc((void **)&d_result, size_in_bytes);

        // Copy to the GPU
        cudaMemcpy(d_a, a.data(), size_in_bytes,  cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), size_in_bytes,  cudaMemcpyHostToDevice);

        // Perform the computation
        const auto nb = (result.size() + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
        add<<<nb, THREADS_PER_BLOCK>>>(d_a, d_b, d_result, result.size());

        // Copy the results back to the host
        cudaMemcpy(result.data(), d_result, size_in_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_result);
    }



    /**
     * Compute de cumulative distribution function of the values of a given container for the received probability function.
     * @tparam Container Type of the container
     * @param c The container
     * @param prob_func The probalility function
     * @return
     */
    template <class Container>
    auto cdf(Container& c, std::function<float(typename Container::value_type)>&& prob_func) {

        std::vector<float> result (c.size());
        result.reserve(c.size());
        result[0] = prob_func(c[0]);
        for (std::size_t i = 0; i < c.size(); i++)
            result[i] = result[i - 1] + prob_func(c[i]);

        return result;
    }
}

#endif //CADLABS_FUNCTIONS_H
