/*
 *
 *  Header with auxiliary functions for the test suites
 *
 */

#ifndef CADLABS_CAD_TEST
#define CADLABS_CAD_TEST

#include <gtest/gtest.h>

namespace cad {

    /**
     * Assert that all elements of a container equal a given value
     */
    template<std::size_t Stride, typename Container, typename T>
    inline void expect_container_value(Container &c, T value) {

        for (unsigned int i = 0; i < c.size(); i+= Stride)
            EXPECT_EQ(value, c[i]);

    }

    template<typename Container, typename T>
    inline void expect_container_value(Container* c, T value, std::size_t size) {

        for (unsigned int i = 0; i < size; i++)
            EXPECT_EQ(value, c[i]);

    }

    /**
     * Assert that the contents of two containers are the same
     */
    template<typename Container1, typename Container2>
    inline void expect_container_eq(Container1& a, Container2&& b) {

        EXPECT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); i++)
            EXPECT_EQ(a[i], b[i]);
    }

    /**
 * Assert that the contents of two containers are the same
 */
    template<typename Container1, typename Container2>
    inline void expect_container_float_eq(Container1& a, Container2&& b) {

        EXPECT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); i++)
            EXPECT_FLOAT_EQ(a[i], b[i]);
    }
}

#endif // CADLABS_CAD_TEST