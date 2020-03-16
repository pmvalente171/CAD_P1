//
// Created by Hervé Paulino on 06/03/18.
//

#ifndef CADLABS_IMAGE_H
#define CADLABS_IMAGE_H

#include <array>
#include <algorithm>
#include <type_traits>
#include <wb/wbImport.h>

namespace cad {

    /**
     * Image with the raw data stored as an array of T
     * @tparam T The type of the array storing the image's data
     */
    template<typename T = float>
    class image {
        

    private:

        /**
         * Width of the image
         */
         unsigned width;

        /**
         * Height of the image
         */
         unsigned height;
         
         /**
          * Number of channels
          */
         unsigned n_channels;
         
         /**
          * Size of the image's raw data array
          */
         unsigned data_size;

        /**
         * Pointer to data
         */
        std::unique_ptr<T[]> data;

        /**
         * Pointer to data on the GPU
         */
        T* gpu_data;

    public:
        /**
         * Construct an image object to store an image
         * @param width Width of the image
         * @param height Height of the image
         * @param n_channels Number of channels of the image. Default 1.
         */
        image(const unsigned width, const unsigned height, const unsigned n_channels  = 1) :
                width(width),
                height(height),
                n_channels (n_channels),
                data_size(width * height * n_channels),
                data(std::make_unique<T[]>(data_size)),
                gpu_data (nullptr){}

        /**
         * Construct image object of a given object file.
         * Only enabled for image<float> type.
         * @param filename The name of the file to load.
         */
        template <typename U = T, std::enable_if_t<std::is_same<U, float>::value>* = nullptr>
        image(const std::string& filename) {
            wbImage_t inputImage = wbImport(filename.c_str());

            if (inputImage == NULL)
                throw std::runtime_error("Image corrupted");

            width = wbImage_getWidth(inputImage);
            height = wbImage_getHeight(inputImage);
            n_channels = wbImage_getChannels(inputImage);
            data_size = width * height * n_channels;
            data = std::unique_ptr<T[]>(wbImage_getData(inputImage));
            gpu_data = nullptr;
        }

        /**
         * Move constructor
         */
        image(image<T>&& other) :
                width(other.width),
                height(other.height),
                data(std::move(other.data)),
                gpu_data(other.gpu_data) {

            other.gpu_data = nullptr;
        }

        /**
         * Destructor
         */
        ~image() {
            if (gpu_data)
                cudaFree(gpu_data);
        }

        /**
         * Size of the image in number of bytes
         * @return
         */
        auto size() const {
            return data_size;
        }

        /**
         * Obtain a image object with the raw data in N (unsigned chars)
         * Enabled only for image<float>
         * @return The new image
         */
        template <typename U = T, std::enable_if<std::is_same<U, float>::value>* = nullptr>
        image<unsigned char> to_integer() {

            image<unsigned char> ucharImage (width, height, n_channels);
            for (unsigned i = 0; i < data_size; i++)
                    ucharImage[i] = (unsigned char) (255 * data[i]);
            return ucharImage;
        }

        /**
         * Obtain a image object with the raw data in floating point (float)
         * Enabled only for image<unsigned char>
         * @return The new image
         */
        template <typename U = T, std::enable_if_t<std::is_same<U, unsigned char>::value>* = nullptr>
        image<float> to_float() {

            image<float> t_img (width, height, n_channels);
            for (unsigned i = 0; i < data_size; i++)
                t_img[i] = (float) (data[i]/255.0);
            return t_img;
        }

        /**
         * Obtain a new image that results from the conversion of the current to grey scale
         * @return The new image
         */
       auto to_greyscale() {

           image<T> grayImage(width, height);

            for (unsigned i = 0; i < width; i++) {
                auto index = i * height;
                for (unsigned j = 0; j < height; j++, index++) {
                    auto r = data[3 * index];
                    auto g = data[3 * index + 1];
                    auto b = data[3 * index + 2];
                    grayImage[index] = (T) (0.21 * r + 0.71 * g + 0.07 * b);
                }
            }

            return grayImage;
        }

        /**
         * Color histogram
         * @tparam Size The number of bins of the histogram
         * @return The histogram
         */
        template <unsigned Size>
        std::array<unsigned, Size> histogram() {
            std::array<unsigned, Size> hist { 0 };

            for (unsigned i = 0; i < data_size; i++)
                hist[data[i]]++;

            return hist;
        }

        /**
         * Correct the image's color given a cumulative distribution function of the colors
         * @tparam CDF The type of cumulative distribution function
         * @param cdf The cumulative distribution function
         */
        template <typename CDF>
        void correct_color(CDF& cdf) {

            auto end = cdf.size()-1;
            auto clamp = [end = end](unsigned char c) { return std::min(std::max(c, (unsigned char ) 0), (unsigned char ) end); };

            auto cdfmin = cdf[0];
            for (unsigned i = 0; i < data_size; i++)
                data[i] = clamp(end*(cdf[data[i]] - cdfmin)/(1 - cdfmin));
        }

        /**
         * Subscript operator
         * @param index
         * @return
         */
        T& operator[] (const int index) {
            return data[index];
        }

        /**
         * Subscript operator for const objects
         * @param index
         * @return
         */
        const T& operator[] (const int index) const {
            return data[index];
        }

        ////////////////// GPU support

        /**
         * Allocate space for the image's data in the GPU
         */
        void alloc_gpu() {
            /* if (!gpu_data)
                gpu_data = ... TODO
            */
        }

        /**
         * Copy the image's data to the GPU
         */
        void copy_to_gpu() {
            // TODO
        }

        /**
        * Copy the image's data from the GPU
        */
        void copy_from_gpu() {
            // TODO
        }

    };

}

#endif //CADLABS_IMAGE_H
