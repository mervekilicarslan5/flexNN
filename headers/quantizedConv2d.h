#pragma once
#include "layer.h"
#include <vector>
//try int32 first
#include <cstdint> // for int8_t

namespace simple_nn {
    template<typename T>
    class QuantizedConv2d : public Conv2d<T> {
    private:
    
        MatX<int8_t> qkernel; // Quantized weights
        VecX<int32_t> qbias; // Quantized bias
        float in_scale; // Input scale
        int in_zero_point; // Input zero point
        float weight_scale; // Weight scale
        int weight_zero_point; // Weight zero point
        float out_scale; // Output scale
        int out_zero_point; // Output zero point

    public:
        // Constructor to initialize the quantized layer with quantization parameters
        QuantizedConv2d(const MatX<T>& prev_out, bool is_training, float in_scale, int in_zero_point, float weight_scale, int weight_zero_point, float out_scale, int out_zero_point);

        // Override the forward method to perform quantized convolution
        void forward(const MatX<T>& prev_out, bool is_training) override;
    };

    template<typename T>
    QuantizedConv2d<T>::QuantizedConv2d(const MatX<T>& prev_out, bool is_training, float in_scale, int in_zero_point, float weight_scale, int weight_zero_point, float out_scale, int out_zero_point)
        : Conv2d<T>(const MatX<T>& prev_out, bool is_training, in_scale(in_scale), in_zero_point(in_zero_point), weight_scale(weight_scale), weight_zero_point(weight_zero_point), out_scale(out_scale), out_zero_point(out_zero_point) {
    }




    template<typename T>
    void QuantizedConv2d<T>::forward(const MatX<T>& prev_out, bool is_training) {

        MatXi q_prev_out = ((prev_out.array() / in_scale).round().template cast<int>() + in_zero_point).matrix().template cast<int8_t>();

        // Initialize quantized output matrix. This will be in int32 due to accumulation over int8 products.
        MatXi q_output = MatXi::Zero(this->batch, this->ohw * this->oc);

        // Perform quantized convolution using qkernel, q_prev_out, and qbias
        for (int n = 0; n < this->batch; ++n) {
            for (int oc = 0; oc < this->oc; ++oc) {
                for (int oh = 0; oh < this->oh; ++oh) {
                    for (int ow = 0; ow < this->ow; ++ow) {
                        // Perform the convolution for one output element
                        int32_t sum = 0;
                        for (int ic = 0; ic < this->ic; ++ic) {
                            for (int kh = 0; kh < this->kh; ++kh) {
                                for (int kw = 0; kw < this->kw; ++kw) {
                                    // Calculate the input indices with stride and padding considered.
                                    int ih = oh * stride - pad + kh;
                                    int iw = ow * stride - pad + kw;
                                    if (ih >= 0 && ih < this->ih && iw >= 0 && iw < this->iw) {
                                        sum += q_prev_out(n, ic * this->ih * this->iw + ih * this->iw + iw) * qkernel(oc, ic * this->kh * this->kw + kh * this->kw + kw);
                                    }
                                }
                            }
                        }
                        if (use_bias) {
                            sum += qbias(oc);
                        }
                        // Store the sum into quantized output matrix
                        q_output(n, oc * this->ohw + oh * this->ow + ow) = sum;
                    }
                }
            }
        }
        this->output = ((q_output.cast<float>() - out_zero_point) * out_scale).matrix();

        // Now this->output contains the dequantized output which can be used by subsequent layers.
    }
}