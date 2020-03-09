/**
 * Tests for functions
 */

#include <memory>
#include <array>
#include <algorithm>

#include "../cad_test.h"

#include <functions.h>
#include <marrow/timer.hpp>

using namespace cad;



template <unsigned NRuns, std::size_t SIZE>
double array_add_test() {

    auto result = std::make_unique<std::array<int, SIZE>>();

    auto a = std::make_unique<std::array<int, SIZE>>();
    auto b = std::make_unique<std::array<int, SIZE>>();

    std::fill (a->begin(), a->end(), 1);
    std::fill (b->begin(), b->end(), 2);

    marrow::timer<> t;
    for (int i = 0 ; i < NRuns; i++) {
        t.start();
        array_add<>(*result, *a, *b);
        t.stop();
    }

    //    cad::expect_container_value(*result, 3);

    return t.average();
};


////////////// Tests for contiguous data

static constexpr unsigned NRuns = 10;

TEST(ArrayAdd, SIZE_1000000) {
    double elapsed  = array_add_test<NRuns, 1000000>();
    std::cout << "Elapsed time: " << elapsed << " milliseconds \n";
}

TEST(ArrayAdd, SIZE_10000000) {
    double elapsed  = array_add_test<NRuns, 10000000>();
    std::cout << "Elapsed time: " << elapsed << " milliseconds \n";
}

TEST(ArrayAdd, Size_100000000) {
    double elapsed  = array_add_test<NRuns, 100000000>();
    std::cout << "Elapsed time: " << elapsed << " milliseconds \n";
}

