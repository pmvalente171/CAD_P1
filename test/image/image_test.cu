/**
 * Tests for image
 */


#include "../cad_test.h"

#include <image.h>
#include <functions.h>
#include <marrow/timer.h>

using namespace cad;

static const std::string data_folder = "../../../data/";

TEST(Image, Example) {

    image input (data_folder + "input01.ppm");

    // To a integer representation
    auto img = input.to_integer();
    // Change to a grey scale
    auto greyImage = img.to_greyscale();
    // Compute the color histogram
    auto hist = greyImage.histogram<256>();
    // Compute the cumulative distribution function of the histogram
    auto cdf_hist = cdf(hist, [size = greyImage.size()](unsigned c) { return c / (float) size ;  });
    // Use the CDF to correct the image's color
    img.correct_color(cdf_hist);

    // Obtain the image as an array of floats
    auto output = img.to_float();

    // Check if the result is correct
    image expected (data_folder + "output01.ppm");
    expect_container_float_eq(output, expected);

}