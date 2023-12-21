#pragma once
#include "../headers/simple_nn.h"
/*
 VGG(
    (features): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True)
      (4): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace=True)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace=True)
      (9): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace=True)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace=True)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace=True)
      (16): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (18): ReLU(inplace=True)
      (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace=True)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace=True)
      (23): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (25): ReLU(inplace=True)
      (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (27): ReLU(inplace=True)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace=True)
      (30): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
if classes == 10:
    (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
    (classifier): Sequential(
      (0): Linear(in_features=25088, out_features=4096, bias=True)
      (1): ReLU(inplace=True)
      (2): Dropout(p=0.5, inplace=False)
      (3): Linear(in_features=4096, out_features=4096, bias=True)
      (4): ReLU(inplace=True)
      (5): Dropout(p=0.5, inplace=False)
      (6): Linear(in_features=4096, out_features=1000, bias=True)
    )
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=512, out_features=256, bias=True)
    (2): ReLU()
    (3): Linear(in_features=256, out_features=256, bias=True)
    (4): ReLU()
    (5): Linear(in_features=256, out_features=10, bias=True)
  )
if classes == 100:

)
*/

// Conv2d(int in_channels, int out_channels, int kernel_size, int padding, string option);
// Linear(int in_features, int out_features, string option);
// MaxPool2d(int kernel_size, int stride);
// AvgPool2d(int kernel_size, int stride);

using namespace simple_nn;

template <typename T>
void make_vgg16(SimpleNN<T> &net, int num_classes = 10)
{
    net.add(new Conv2d<T>(3,64,3,1,1));
    net.add(new ReLU<T>());
    net.add(new Conv2d<T>(64,64,3,1,1));
    net.add(new ReLU<T>());
    net.add(new AvgPool2d<T>(2,2));
    net.add(new Conv2d<T>(64,128,3,1,1));
    net.add(new ReLU<T>());
    net.add(new Conv2d<T>(128,128,3,1,1));
    net.add(new ReLU<T>());
    net.add(new AvgPool2d<T>(2,2));
    net.add(new Conv2d<T>(128,256,3,1,1));
    net.add(new ReLU<T>());
    net.add(new Conv2d<T>(256,256,3,1,1));
    net.add(new ReLU<T>());
    net.add(new Conv2d<T>(256,256,3,1,1));
    net.add(new ReLU<T>());
    net.add(new AvgPool2d<T>(2,2));
    net.add(new Conv2d<T>(256,512,3,1,1));
    net.add(new ReLU<T>());
    net.add(new Conv2d<T>(512,512,3,1,1));
    net.add(new ReLU<T>());
    net.add(new Conv2d<T>(512,512,3,1,1));
    net.add(new ReLU<T>());
    net.add(new AvgPool2d<T>(2,2));
    net.add(new Conv2d<T>(512,512,3,1,1));
    net.add(new ReLU<T>());
    net.add(new Conv2d<T>(512,512,3,1,1));
    net.add(new ReLU<T>());
    net.add(new Conv2d<T>(512,512,3,1,1));
    net.add(new ReLU<T>());
    net.add(new AvgPool2d<T>(2,2));
    net.add(new AdaptiveAvgPool2d<T>(7,7));
    
    net.add(new Flatten<T>());
    net.add(new Linear<T>(25088,4096));
    net.add(new ReLU<T>());
    net.add(new Linear<T>(4096,4096));
    net.add(new ReLU<T>());
    net.add(new Linear<T>(4096,1000));

    if(num_classes == 10)
    {
        /* net.add(new Flatten<T>()); */
        net.add(new Linear<T>(512,256));
        net.add(new ReLU<T>());
        net.add(new Linear<T>(256,256));
        net.add(new ReLU<T>());
        net.add(new Linear<T>(256,10));
    }
    else if(num_classes == 200)
    {
        net.add(new AvgPool2d<T>(2,2));
        net.add(new Flatten<T>());
        net.add(new Linear<T>(512,512));
        net.add(new ReLU<T>());
        net.add(new Linear<T>(512,200));
    }
    else if(num_classes == 1000)
    {
        net.add(new AvgPool2d<T>(2,2));
        /* net.add(new Flatten<T>()); */
        net.add(new Linear<T>(4608,4096));
        net.add(new Linear<T>(4096,4096));
        net.add(new Linear<T>(4096,1000));
    }
    else
    {
        std::cout << "Error: num_classes must be 10, 200, or 1000" << std::endl;
        exit(1);
    }

}

        

