#pragma once
#include "common.h"
#include "layer.h"
#include "quantization_utils.h" 

namespace simple_nn
{
    template<typename T>
    class QuantizedLinear : public Layer<T>
    {
    private:
        int batch;
        int in_feat;
        int out_feat;
        string option;
        MatX<int32_t> dW; // Gradient in higher precision
        VecX<int32_t> db; // Gradient in higher precision
        
        // Quantization parameters (assuming per-layer quantization for simplicity)
        float input_scale;
        int input_zero_point;
        float weight_scale;
        int weight_zero_point;
        float output_scale;
        int output_zero_point;

    public:
        MatX<int8_t> W; // Quantized weights
        VecX<int32_t> b; // Quantized biases
        
        QuantizedLinear(int in_features, int out_features, string option = "kaiming_uniform");
        void set_layer(const vector<int>& input_shape) override;
        void forward(const MatX<T>& prev_out, bool is_training) override;
        // Other functions...
    };

    template<typename T>
    QuantizedLinear<T>::QuantizedLinear(int in_features, int out_features, string option)
    : Layer<T>(LayerType::LINEAR), batch(0), in_feat(in_features), out_feat(out_features), option(option) {
        // Initialize quantization parameters, weights, and biases here
        // W and b should be filled with quantized values
    }

    template<typename T>
    void QuantizedLinear<T>::set_layer(const vector<int>& input_shape)
    {
        batch = input_shape[0];
        this->output.resize(batch, out_feat);
        this->delta.resize(batch, out_feat);

        // Allocate space for quantized weights and biases
        W.resize(out_feat, in_feat);
        dW.resize(out_feat, in_feat);
        b.resize(out_feat);
        db.resize(out_feat);

        // Initialize quantized weights and biases
        // Typically, this would involve loading pre-quantized values
        // or quantizing initial values here
    }

    template<typename T>
    void QuantizedLinear<T>::forward(const MatX<T>& prev_out, bool is_training)
    {
        // Quantize input if not already quantized
        MatX<int8_t> q_input = Quantization::quantize_batch(prev_out, input_scale, input_zero_point);

        // Perform matrix multiplication with quantized weights and input
        MatX<int32_t> raw_output = q_input * W.transpose(); // Result might be in int32 for accumulation
        
        // Add quantized bias and scale back the result
        for (int n = 0; n < batch; ++n) {
            for (int i = 0; i < out_feat; ++i) {
                int32_t temp = raw_output(n, i) + b[i]; // Adding bias
                // Scale and shift back using output scale and zero_point
                this->output(n, i) = Quantization::dequantize(temp, output_scale, output_zero_point);
            }
        }
    }

    // ... Implement other necessary member functions...

}
