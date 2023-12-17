#pragma once
#include "../headers/simple_nn.h"

// Conv2d(int in_channels, int out_channels, int kernel_size, int padding, string option);
// Linear(int in_features, int out_features, string option);
// MaxPool2d(int kernel_size, int stride);
// AvgPool2d(int kernel_size, int stride);

using namespace simple_nn;
/*
 class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
*/
template <typename T>
void add_block(SimpleNN<T> &net, int in_channels, int intermediate_channels, int identity_downsample, int stride, string option) {
    int expansion = 4;
    net.push_back(new Conv2d<T>(in_channels, intermediate_channels, 1,1, 0, option)); //conv1
    net.push_back(new BatchNorm2d<T>); //bn1
    net.push_back(new ReLU<T>);
    net.push_back(new Conv2d<T>(intermediate_channels, intermediate_channels, 3, stride, 1, option)); //conv2
    net.push_back(new BatchNorm2d<T>); //bn2
    net.push_back(new ReLU<T>);
    net.push_back(new Conv2d<T>(intermediate_channels, intermediate_channels * expansion, 1, 1, 0, option)); //conv3
    net.push_back(new BatchNorm2d<T>); //bn3
    if (identity_downsample == 1) {
        net.push_back(new Conv2d<T>(in_channels, intermediate_channels * 4, stride, 1, option)); //needs to be applied to identity
        net.push_back(new BatchNorm2d<T>); //needs to be applied to identity
    }
   //x+=identity 
    net.push_back(new ReLU<T>);
}

/*
     def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
*/

template <typename T>
void make_layer(SimpleNN<T> &net, int num_residual_blocks, int intermediate_channels, int stride, string option) {
    int identity_downsample = 0;
    if (stride != 1 || net.in_channels != intermediate_channels * 4) {
        identity_downsample = 1;
    }
    add_block(net, net.in_channels, intermediate_channels, identity_downsample, stride, option);
    net.in_channels = intermediate_channels * 4;
    for (int i = 0; i < num_residual_blocks - 1; i++) {
        add_block(net, net.in_channels, intermediate_channels, 0, 1, option);
    }
}

/*
 class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
*/

template <typename T>
void make_resnet(SimpleNN<T> &net, int image_channels, int num_classes, string option) {
    net.in_channels = 64;
    net.push_back(new Conv2d<T>(image_channels, 64, 7, 3, 2, option));
    net.push_back(new BatchNorm2d<T>);
    net.push_back(new ReLU<T>);
    net.push_back(new MaxPool2d<T>(3, 2));
    make_layer(net, 3, 64, 1, option);
    make_layer(net, 4, 128, 2, option);
    make_layer(net, 6, 256, 2, option);
    make_layer(net, 3, 512, 2, option);
    net.push_back(new AvgPool2d<T>(1, 1)); //https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work
                                           // adjust stride and kernel size
    net.push_back(new Flatten<T>);
    net.push_back(new Linear<T>(512 * 4, num_classes, option));
}

template <typename T>
class ResNet
{

ResNet<T>(int image_channels, int num_classes, string option) {
}

}


