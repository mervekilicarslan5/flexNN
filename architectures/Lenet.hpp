#pragma once
#include "../headers/simple_nn.h"
#include <iostream>

using namespace simple_nn;


template <typename T>
class LeNet : public SimpleNN<T>
{
public:
    // Add a constructor for LeNet that accepts quantization parameters
    LeNet(bool quantization = false,
        float conv0_scale = 1.0, int conv0_zero_point = 0,
        float conv3_scale = 1.0, int conv3_zero_point = 0,
        // Fully connected layer quantization parameters
        float fc0_scale = 1.0, int fc0_zero_point = 0,
        float fc2_scale = 1.0, int fc2_zero_point = 0,
        float fc4_scale = 1.0, int fc4_zero_point = 0,
        // Overall quantization parameters (if applicable)
        float quant_scale = 1.0, int quant_zero_point = 0)
    {
        if (quantization)
        {
            // Assuming your quantization layers are defined similarly to the provided snippets
            // Add the quantized layers here with the appropriate parameters
            this->add(new QuantizedConv2d<T>(/* parameters including quantization parameters */));
            this->add(new ReLU<T>());
            this->add(new AvgPool2d<T>(2, 2));

            this->add(new QuantizedConv2d<T>(/* parameters including quantization parameters */));
            this->add(new ReLU<T>());
            this->add(new AvgPool2d<T>(2, 2));

            // Flatten layer will be needed before adding linear layers
            this->add(new Flatten<T>());

            // Adding quantized linear (fully connected) layers
            this->add(new QuantizedLinear<T>(/* parameters including quantization parameters */));
            this->add(new ReLU<T>());
            this->add(new QuantizedLinear<T>(/* parameters including quantization parameters */));
            this->add(new ReLU<T>());
            this->add(new QuantizedLinear<T>(/* parameters including quantization parameters */));
        }
        else
        {
                // Add convolutional layers and pooling layers similar to the Python version
            this->add(new Conv2d<T>(1, 6, 5, 2)); // Conv2d with padding=2
            this->add(new ReLU<T>());
            this->add(new AvgPool2d<T>(2, 2));

            this->add(new Conv2d<T>(6, 16, 5, 0)); // Conv2d with no padding
            this->add(new ReLU<T>());
            this->add(new AvgPool2d<T>(2, 2));

            // Flatten layer will be needed before adding linear layers
            this->add(new Flatten<T>());

            // Adding linear (fully connected) layers
            this->add(new Linear<T>(400, 120)); // Adjust the input size according to your specific architecture
            this->add(new ReLU<T>());
            this->add(new Linear<T>(120, 84));
            this->add(new ReLU<T>());
            this->add(new Linear<T>(84, 10));

        }
    }
};


