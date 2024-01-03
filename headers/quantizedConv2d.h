#pragma once
#include "layer.h"
#include "convolutional_layer.h"
#include "quantization_utils.h"  
#include <vector>
#include <cstdint> // for int8_t

namespace simple_nn {
    template<typename T>
    class QuantizedConv2d : public Conv2d<T> {
        private:
        // Utility function for performing quantized convolution
        MatX<int32_t> quantizedConvolution(const MatX<int8_t>& input,
                                               const MatX<int8_t>& weights,
                                               const VecX<int32_t>& bias,
                                               float input_scale, int input_zero_point,
                                               float weight_scale, int weight_zero_point,
                                               float output_scale, int output_zero_point,
                                               int stride, int padding);
        
    public:
    
        MatX<int8_t> qkernel; // Quantized weights
        VecX<int32_t> qbias; // Quantized bias
        float in_scale; // Input scale
        int in_zero_point; // Input zero point
        float weight_scale; // Weight scale
        int weight_zero_point; // Weight zero point
        float out_scale; // Output scale
        int out_zero_point; // Output zero point

        // Constructor to initialize the quantized layer with quantization parameters
        QuantizedConv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, 
                        bool use_bias = true, string option = "kaiming_uniform",
                        float in_scale = 1.0, int in_zero_point = 0,
                        float weight_scale = 1.0, int weight_zero_point = 0,
                        float out_scale = 1.0, int out_zero_point = 0) :
            Conv2d<T>(in_channels, out_channels, kernel_size, stride, padding, use_bias, option),
            in_scale(in_scale), in_zero_point(in_zero_point), 
            weight_scale(weight_scale), weight_zero_point(weight_zero_point), 
            out_scale(out_scale), out_zero_point(out_zero_point) 
        {
            // Here you would initialize or load your quantized kernel (qkernel) and quantized bias (qbias)
            // This might involve loading from a file, or quantizing an existing set of weights and biases
            // For example:
            // qkernel = some_quantization_function(Conv2d<T>::kernel, weight_scale, weight_zero_point);
            // if (use_bias) qbias = some_quantization_function(Conv2d<T>::bias, weight_scale, weight_zero_point);
        }

        // Override the forward method to perform quantized convolution
        void forward(const MatX<T>& prev_out, bool is_training) override {
            // Step 1: Quantize the input
            MatX<int8_t> quantized_input(prev_out.rows(), prev_out.cols());
            for (int i = 0; i < prev_out.size(); i++) {
                quantized_input(i) = Quantization::quantize(prev_out(i), this->in_scale, this->in_zero_point);
            }

            // Step 2: Perform convolution with quantized inputs and weights
            // Here, implement your method for quantized convolution
            // This will depend on how your convolution is implemented and how you handle quantized values
                // Step 2: Perform convolution with quantized inputs and weights
            MatX<int32_t> quantized_output = quantizedConvolution(
                quantized_input,        // Quantized input
                this->qkernel,          // Quantized weights
                this->qbias,            // Quantized bias
                this->in_scale,         // Input scale
                this->in_zero_point,    // Input zero point
                this->weight_scale,     // Weight scale
                this->weight_zero_point,// Weight zero point
                this->out_scale,        // Output scale
                this->out_zero_point,   // Output zero point
                this->stride,           // Stride
                this->pad               // Padding
            );
            
            // Step 3: Dequantize the output
            MatX<T> dequantized_output(quantized_output.rows(), quantized_output.cols());
            for (int i = 0; i < quantized_output.size(); i++) {
                dequantized_output(i) = Quantization::dequantize(quantized_output(i), this->out_scale, this->out_zero_point);
            }

            // Store the result in this->output
            this->output = dequantized_output;

            // Add bias if used (after dequantizing the output)
            if (this->use_bias) {
                for (int i = 0; i < this->output.rows(); i++) {
                    this->output.row(i) += this->bias; // Make sure bias is in the correct format
                }
            }
        }

        // ... (Any additional methods required for your class) ...
    };
    template<typename T>
    MatX<int32_t> QuantizedConv2d<T>::quantizedConvolution(const MatX<int8_t>& input,
                                               const MatX<int8_t>& weights,
                                               const VecX<int32_t>& bias,
                                               float input_scale, int input_zero_point,
                                               float weight_scale, int weight_zero_point,
                                               float output_scale, int output_zero_point,
                                               int stride, int padding) {
    // Define the size of the output based on input, stride, padding, etc.
    // Here, just placeholders for the actual calculations
    int output_height = (this->input_height + 2 * this->padding_height - this->kernel_height) / this->stride_height + 1;
    int output_width = (this->input_width + 2 * this->padding_width - this->kernel_width) / this->stride_width + 1;

    MatX<int32_t> output(output_height, output_width); // Initialize output matrix

    // Iterate over the output elements
    for (int oh = 0; oh < output_height; oh++) {
        for (int ow = 0; ow < output_width; ow++) {
            int32_t sum = 0;
            // Perform the dot product for each output element
            for (int kh = 0; kh < weights.rows(); kh++) {
                for (int kw = 0; kw < weights.cols(); kw++) {
                    // Calculate input indices considering stride and padding
                    int ih = oh * this->stride - this->padding + kh;
                    int iw = ow * this->stride - this->padding + kw;
                    if (ih >= 0 && ih < input.rows() && iw >= 0 && iw < input.cols()) {
                        // Accumulate the product of the quantized input and weight
                        sum += (input(ih, iw) - this->input_zero_point) * (weights(kh, kw) - this->weight_zero_point);
                    }
                }
            }
            // Add quantized bias, scale, and shift the result back to the output range
            sum += bias(oh, ow); // Assuming bias is already appropriately quantized or scaled
            // Dequantize output to a floating-point number if necessary
            output(oh, ow) = (sum - output_zero_point) * output_scale;
        }
    }

    return output;
}
    
}
