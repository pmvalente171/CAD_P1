//
// Created by Hervé Paulino on 06/03/18.
//

#ifndef CADLABS_FUNCTIONS_H
#define CADLABS_FUNCTIONS_H

#include <vector>

namespace cad {

    /**
     * Pairwise addition of two containers
     * @tparam Stride Stride between accesses to the containers. Default is 1.
     * @tparam Container Type of the containers. Assumed to be the same for all three arguments
     * @param result The container to received the addition
     * @param a The first operand
     * @param b The second operand
     */
    template<std::size_t Stride = 1, class Container>
    void array_add(Container &result, Container &a, Container &b) {

        for (std::size_t i = 0; i < result.size(); i+=Stride)
            result[i] = a[i] + b[i];
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
