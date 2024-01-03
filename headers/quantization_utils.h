#pragma once

#include <algorithm> // For std::min and std::max
#include <cmath> // For std::round

namespace Quantization {

    // Quantizes a single floating-point number to int8_t
    inline int8_t quantize(float value, float scale, int zero_point) {
        // Ensure that the zero_point is adjusted to int8_t range
        zero_point = std::max(-128, std::min(127, zero_point));

        // Quantization process
        int32_t quantized = static_cast<int32_t>(std::round(value / scale) + zero_point);
        // Clamping the result to the valid range of int8_t
        quantized = std::max(-128, std::min(127, quantized));

        return static_cast<int8_t>(quantized);
    }

    // Dequantizes a single int8_t number to floating-point
    inline float dequantize(int8_t quantized_value, float scale, int zero_point) {
        return (quantized_value - zero_point) * scale;
    }

    // Example of a batch quantization function for an entire matrix
    template<typename MatrixType, typename QuantizedMatrixType>
    void quantize_batch(const MatrixType& matrix, QuantizedMatrixType& quantized_matrix, float scale, int zero_point) {
        for(int i = 0; i < matrix.rows(); ++i) {
            for(int j = 0; j < matrix.cols(); ++j) {
                quantized_matrix(i, j) = quantize(matrix(i, j), scale, zero_point);
            }
        }
    }

    // Example of a batch dequantization function for an entire matrix
    template<typename QuantizedMatrixType, typename MatrixType>
    void dequantize_batch(const QuantizedMatrixType& quantized_matrix, MatrixType& dequantized_matrix, float scale, int zero_point) {
        for(int i = 0; i < quantized_matrix.rows(); ++i) {
            for(int j = 0; j < quantized_matrix.cols(); ++j) {
                dequantized_matrix(i, j) = dequantize(quantized_matrix(i, j), scale, zero_point);
            }
        }
    }

    // ... Potentially other utility functions related to quantization ...

} // namespace Quantization

/*#pragma once

#include <algorithm> // For std::min and std::max
#include <cmath> // For std::round

namespace Quantization {

    // Quantizes a single floating-point number to int8_t
    inline int8_t quantize(float value, float scale, int zero_point) {
        int32_t quantized = static_cast<int32_t>(std::round(value / scale) + zero_point);
        quantized = std::max(-128, std::min(127, quantized));
        return static_cast<int8_t>(quantized);
    }

    // Dequantizes a single int8_t number to floating-point
    inline float dequantize(int8_t quantized_value, float scale, int zero_point) {
        return (quantized_value - zero_point) * scale;
    }

    // Example of a batch quantization function for an entire matrix
    template<typename T, typename U>
    void quantize_batch(const T& matrix, U& quantized_matrix, float scale, int zero_point) {
        for(int i = 0; i < matrix.rows(); ++i) {
            for(int j = 0; j < matrix.cols(); ++j) {
                quantized_matrix(i, j) = quantize(matrix(i, j), scale, zero_point);
            }
        }
    }

    // Example of a batch dequantization function for an entire matrix
    template<typename T, typename U>
    void dequantize_batch(const T& quantized_matrix, U& dequantized_matrix, float scale, int zero_point) {
        for(int i = 0; i < quantized_matrix.rows(); ++i) {
            for(int j = 0; j < quantized_matrix.cols(); ++j) {
                dequantized_matrix(i, j) = dequantize(quantized_matrix(i, j), scale, zero_point);
            }
        }
    }

    // ... Potentially other utility functions related to quantization ...

} // namespace Quantization
*/