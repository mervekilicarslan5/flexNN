#pragma once
#include "../headers/simple_nn.h"

using namespace simple_nn;
/* LeNet( */
/*   (features): Sequential( */
/*     (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1)) */
/*     (1): ReLU() */
/*     (2): AvgPool2d(kernel_size=2, stride=2, padding=0) */
/*     (3): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1)) */
/*     (4): ReLU() */
/*     (5): AvgPool2d(kernel_size=2, stride=2, padding=0) */
/*   ) */
/*   (classifier): Sequential( */
/*     (0): Flatten(start_dim=1, end_dim=-1) */
/*     (1): Linear(in_features=800, out_features=500, bias=True) */
/*     (2): ReLU() */
/*     (3): Linear(in_features=500, out_features=10, bias=True) */
/*   ) */
/* ) */

template <typename T>
void make_lenet(SimpleNN<T> &net, int num_classes = 10)
{
    net.add(new Conv2d<T>(1,20,5,1,0));
    net.add(new ReLU<T>());
    net.add(new AvgPool2d<T>(2,2));
    net.add(new Conv2d<T>(20,50,5,1,0));
    net.add(new ReLU<T>());
    net.add(new AvgPool2d<T>(2,2));
    net.add(new Flatten<T>());
    net.add(new Linear<T>(800,500));
    net.add(new ReLU<T>());
    net.add(new Linear<T>(500,num_classes));
}




