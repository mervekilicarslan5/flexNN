#pragma once
#include "../headers/simple_nn.h"
using namespace simple_nn;

template <typename T>
void make_alexnet(SimpleNN<T> &net)
{
    // AlexNet feature layers
    net.add(new Conv2d<T>(3, 64, 11, 4, 2));  // (in_channels, out_channels, kernel_size, stride, padding)
    net.add(new ReLU<T>());
    net.add(new AvgPool2d<T>(2, 2));          // (kernel_size, stride)
    net.add(new Conv2d<T>(64, 192, 5, 1, 2));
    net.add(new ReLU<T>());
    net.add(new AvgPool2d<T>(2, 2));
    net.add(new Conv2d<T>(192, 384, 3, 1, 1));
    net.add(new ReLU<T>());
    net.add(new Conv2d<T>(384, 256, 3, 1, 1));
    net.add(new ReLU<T>());
    net.add(new Conv2d<T>(256, 256, 3, 1, 1));
    net.add(new ReLU<T>());
    net.add(new AvgPool2d<T>(2, 2));

    // AlexNet uses an Adaptive Average Pooling layer, ensure to match the output size (6x6).
    net.add(new AdaptiveAvgPool2d<T>(6, 6));

    // AlexNet classifier layers
    net.add(new Flatten<T>());               // Flatten the output of the last layer before passing it to the classifier
    net.add(new Linear<T>(9216, 4096));      // (in_features, out_features)
    net.add(new ReLU<T>());
    //net.add(new Dropout<T>(0.5));            // Dropout with a probability of 0.5
    net.add(new Linear<T>(4096, 4096));
    net.add(new ReLU<T>());
    //net.add(new Dropout<T>(0.5));
    net.add(new Linear<T>(4096, 10));        // Assuming 10 output classes; adjust as necessary
}
