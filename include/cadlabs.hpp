#ifndef CADBLABS_HPP
#define CADBLABS_HPP

#include <array>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iterator>

#include <gtest/gtest.h>

using namespace std;

constexpr auto THREADS_PER_BLOCK = 512;

#define cadLog(message) {  std::cout << "[INFO] " <<  message << std::endl; }


template <typename T, size_t Size>
ostream& operator<<(ostream& out, const array<T, Size>& a) {

    out << "[ ";
    copy(a.begin(),
         a.end(),
         ostream_iterator<T>(out, " "));
    out << "]";

    return out;
}

template <typename T>
ostream& operator<<(ostream& out, const vector<T>& a) {

    out << "[ ";
    copy(a.begin(),
         a.end(),
         ostream_iterator<T>(out, " "));
    out << "]";

    return out;

}


/**
 * Assert that the contents of two vectors are the same
 */
template<typename Container>
inline void expect_container_eq(Container &a, Container &b) {

    EXPECT_EQ(a.size(), b.size());
    auto aptr = a.data();
    auto bptr = b.data();
    for (std::size_t i = 0; i < a.size(); i++)
        EXPECT_EQ(aptr[i], bptr[i]);
}

#endif //CADBLABS_HPP
