#pragma once
#include "../headers/simple_nn.h"
#include <iostream>

using namespace simple_nn;


template <typename T>
class QuantLeNet : public SimpleNN<T>
{
public:
    QuantLeNet(int num_classes = 10, bool quantized = true)
    {
        this->add(new Conv2d<T>(1,6,5,1,2, quantized));
        this->add(new ReLU<T>());
        this->add(new AvgPool2d<T>(2,2));
        this->add(new Conv2d<T>(6,16,5,1,0, quantized));
        this->add(new ReLU<T>());
        this->add(new AvgPool2d<T>(2,2));
        this->add(new Flatten<T>());
        this->add(new Linear<T>(400,120, quantized));
        this->add(new ReLU<T>());
        this->add(new Linear<T>(120,84, quantized));
        this->add(new ReLU<T>());
        this->add(new Linear<T>(84,num_classes, quantized));
    }
};



