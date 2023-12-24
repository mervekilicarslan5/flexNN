#pragma once
#include "../headers/simple_nn.h"
#include <queue>

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



/* class ResNet(nn.Module): */
/*     def __init__(self, block, layers, image_channels, num_classes): */
/*         super(ResNet, self).__init__() */
/*         self.in_channels = 64 */
/*         self.conv1 = nn.Conv2d( */
/*             image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False */
/*         ) */
/*         self.bn1 = nn.BatchNorm2d(64) */
/*         self.relu = nn.ReLU() */
/*         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) */

/*         # Essentially the entire ResNet architecture are in these 4 lines below */
/*         self.layer1 = self._make_layer( */
/*             block, layers[0], intermediate_channels=64, stride=1 */
/*         ) */
/*         self.layer2 = self._make_layer( */
/*             block, layers[1], intermediate_channels=128, stride=2 */
/*         ) */
/*         self.layer3 = self._make_layer( */
/*             block, layers[2], intermediate_channels=256, stride=2 */
/*         ) */
/*         self.layer4 = self._make_layer( */
/*             block, layers[3], intermediate_channels=512, stride=2 */
/*         ) */

/*         self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) */
/*         self.fc = nn.Linear(512 * 4, num_classes) */

/*     def forward(self, x): */
/*         x = self.conv1(x) */
/*         x = self.bn1(x) */
/*         x = self.relu(x) */
/*         x = self.maxpool(x) */
/*         x = self.layer1(x) */
/*         x = self.layer2(x) */
/*         x = self.layer3(x) */
/*         x = self.layer4(x) */

/*         x = self.avgpool(x) */
/*         x = x.reshape(x.shape[0], -1) */
/*         x = self.fc(x) */

/*         return x */

/* def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride): */
/*         identity_downsample = None */
/*         layers = [] */

/*         # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes */
/*         # we need to adapt the Identity (skip connection) so it will be able to be added */
/*         # to the layer that's ahead */
/*         if stride != 1 or self.in_channels != intermediate_channels * 4: */
/*             identity_downsample = nn.Sequential( */
/*                 nn.Conv2d( */
/*                     self.in_channels, */
/*                     intermediate_channels * 4, */
/*                     kernel_size=1, */
/*                     stride=stride, */
/*                     bias=False, */
/*                 ), */
/*                 nn.BatchNorm2d(intermediate_channels * 4), */
/*             ) */

/*         layers.append( */
/*             block(self.in_channels, intermediate_channels, identity_downsample, stride) */
/*         ) */

/*         # The expansion size is always 4 for ResNet 50,101,152 */
/*         self.in_channels = intermediate_channels * 4 */

/*         # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer, */
/*         # then finally back to 256. Hence no identity downsample is needed, since stride = 1, */
/*         # and also same amount of channels. */
/*         for i in range(num_residual_blocks - 1): */
/*             layers.append(block(self.in_channels, intermediate_channels)) */

/*         return nn.Sequential(*layers) */
template <typename T>
class ResNet : public SimpleNN<T>
{
private:
    int in_channels;
    vector<int> identity_layers;
    vector<string> identity_layers_type;
public:
    ResNet(int residual_blocks[4], int image_channels, int num_classes, string option) {
        in_channels = 64;
        this->add(new Conv2d<T>(image_channels, 64, 7, 2, 3, false, option));
        this->add(new BatchNorm2d<T>());
        this->add( new ReLU<T>());
        this->add( new MaxPool2d<T>(3, 2, 1));
        /* this->add(new AvgPool2d<T>(3, 2)); */
        /* this->add( new AvgPool2d<T>(3, 2)); */
        this->make_layer( residual_blocks[0], 64, 1, option);
        this->make_layer( residual_blocks[1], 128, 2, option);
        this->make_layer( residual_blocks[2], 256, 2, option);
        this->make_layer( residual_blocks[3], 512, 2, option);
        /* this->add( new AvgPool2d<T>(1, 1)); */
        this->add( new AdaptiveAvgPool2d<T>(1, 1));
        this->add( new Flatten<T>());
        this->add(new Linear<T>(512 * 4, num_classes, option));
    }

    void add_identity_layer(string type) {
        this->identity_layers.push_back(this->net.size());
        this->identity_layers_type.push_back(type);
    }
 
    void add_block(int in_channels, int intermediate_channels, bool identity_downsample, int stride, string option) {
        const int expansion = 4;
        this->add_identity_layer("Identity_Store");
        this->add( new Conv2d<T>(in_channels, intermediate_channels, 1, 1, 0, false, option));
        this->add( new BatchNorm2d<T>());
        this->add( new ReLU<T>());
        this->add( new Conv2d<T>(intermediate_channels, intermediate_channels, 3, stride, 1, false, option));
        this->add( new BatchNorm2d<T>());
        this->add( new ReLU<T>());
        this->add( new Conv2d<T>(intermediate_channels, intermediate_channels * expansion, 1, 1, 0, false, option));
        this->add( new BatchNorm2d<T>());
        if (identity_downsample)
        {
            this->add_identity_layer("Identity_OP_Start");
            this->add(new Conv2d<T>(in_channels, intermediate_channels * 4, 1, stride, 0, false, option));
            this->add(new BatchNorm2d<T>());
            this->add_identity_layer("Identity_OP_Finish");
        }
        this->add_identity_layer("Identity_ADD");
        this->add( new ReLU<T>());
    }
    

    void make_layer(int num_residual_blocks, int intermediate_channels, int stride, string option) {
        bool identity_downsample = false;
        /* vector<SimpleNN<T>> layers; */
        if (stride != 1 || in_channels != intermediate_channels * 4) {
            identity_downsample = true;
        }
        this->add_block(in_channels, intermediate_channels, identity_downsample, stride, option);
        in_channels = intermediate_channels * 4;
        for (int i = 0; i < num_residual_blocks - 1; i++) {
            this->add_block(in_channels, intermediate_channels, false, 1, option);
        }
    }

	void forward(const MatX<T>& X, bool is_training) override {
        MatX<T> identity = X;
        MatX<T> out = X;
        MatX<T> temp = X;
        int i = 0;
		for (int l = 0; l < this->net.size(); l++) {
            if(this->identity_layers.size() != 0 && i < this->identity_layers.size()) {
                while(this->identity_layers[i] == l)  { 
                        if(this->identity_layers_type[i] == "Identity_Store") {
                            identity = out; //store identity of current layer

                        }
                        else if(this->identity_layers_type[i] == "Identity_OP_Start") {
                            //network starts operating on identity, storing last output
                            temp = out; 
                            out = identity;
                        }
                        else if(this->identity_layers_type[i] == "Identity_OP_Finish") {
                            //network finished processing identity, loading back last output
                            identity = out;
                            out = temp;
                        }
                        else if(this->identity_layers_type[i] == "Identity_ADD") {
                            out += identity;
                        }
                    i++;
                    if(i >= this->identity_layers.size()) {
                        break;
                    }
                    }

            }
                this->net[l]->forward(out, is_training);
                out = this->net[l]->output;
            }
        }

	void compile(vector<int> input_shape, Optimizer* optim=nullptr, Loss<T>* loss=nullptr) override
	{
		// set optimizer & loss
		this->optim = optim;
		this->loss = loss;

		// set first & last layer
		this->net.front()->is_first = true;
		this->net.back()->is_last = true;
    
        vector<int> identity = input_shape;
       vector<int> out = input_shape;
        vector<int> temp = input_shape; 


		// set network
        int i = 0;

		for (int l = 0; l < this->net.size(); l++) {
            if(this->identity_layers.size() != 0 && i < this->identity_layers.size()) {
                while(this->identity_layers[i] == l)  { 
                        if(this->identity_layers_type[i] == "Identity_Store") {
                            identity = out; //store identity of current layer
                            /* std::cout << "Identity_Store" << std::endl; */
                        }
                        else if(this->identity_layers_type[i] == "Identity_OP_Start") {
                            //network starts operating on identity, storing last output
                            temp = out; 
                            out = identity;
                            /* std::cout << "Identity_OP_Start" << std::endl; */
                        }
                        else if(this->identity_layers_type[i] == "Identity_OP_Finish") {
                            //network finished processing identity, loading back last output
                            identity = out;
                            out = temp;
                            /* std::cout << "Identity_OP_Finish" << std::endl; */
                        }
                    i++;
                    if(i >= this->identity_layers.size()) {
                        break;
                    }
                    }

            }
            /* if (toString(this->net[l]->type) == "LINEAR" || toString(this->net[l]->type) == "BATCHNORM2D" || toString(this->net[l]->type) == "CONV2D") */
            /* { */
            /* std::cout << "Layer: " << l << ", Layer Type: " << toString(this->net[l]->type) << std::endl; */
            /* } */
            this->net[l]->set_layer(out);
            out = this->net[l]->output_shape();
		
        }

		// set Loss layer
		if (loss != nullptr) {
			loss->set_layer(this->net.back()->output_shape());
	}
	}

        
        



};
    

template <typename T>
ResNet<T> ResNet18(int num_classes, string option, int image_channels = 3) {
    int residual_blocks[4] = {2, 2, 2, 2};
    return ResNet<T>(residual_blocks, image_channels, num_classes, option);
}

template <typename T>
ResNet<T> ResNet50(int num_classes, string option, int image_channels = 3) {
    int residual_blocks[4] = {3, 4, 6, 3};
    return ResNet<T>(residual_blocks, image_channels, num_classes, option);
}

template <typename T>
ResNet<T> ResNet101(int num_classes, string option, int image_channels = 3) {
    int residual_blocks[4] = {3, 4, 23, 3};
    return ResNet<T>(residual_blocks, image_channels, num_classes, option);
}

template <typename T>
ResNet<T> ResNet152(int num_classes, string option, int image_channels = 3) {
    int residual_blocks[4] = {3, 8, 36, 3};
    return ResNet<T>(residual_blocks, image_channels, num_classes, option);
}


/*
class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1,isCulled=False,isTopThinned=False,isBottomThinned=False):
		super(BasicBlock, self).__init__()

		self.isCulled = isCulled
		self.isTopThinned = isTopThinned
		self.isBottomThinned = isBottomThinned

		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		if (not self.isCulled) and (not self.isTopThinned):
			self.relu1 = nn.ReLU()

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		if (not self.isCulled) and (not self.isBottomThinned):
			self.relu2 = nn.ReLU()

		self.shortcut = nn.Sequential()
			

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		if (not self.isCulled) and (not self.isTopThinned):
			out = self.relu1(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out += self.shortcut(x)
		if (not self.isCulled) and (not self.isBottomThinned):
			out = self.relu2(out)
		return out



class RESNET(nn.Module):
	def __init__(self, block, num_blocks, num_classes, alpha = 1.0, rho = 1.0, isCulled=[False,False,False,False], isThinned=[False,False]):
		super(ResNet, self).__init__()

		self.alpha = alpha
		self.rho = rho
		self.in_planes = int(64*alpha)
		
		self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,stride=int(1//rho), padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(self.in_planes)

		self.layer1 = self._make_layer(block, int(64*alpha),  num_blocks[0], isCulled[0],isThinned,stride=1)
		self.layer2 = self._make_layer(block, int(128*alpha), num_blocks[1], isCulled[1],isThinned,stride=2)
		self.layer3 = self._make_layer(block, int(256*alpha), num_blocks[2], isCulled[2],isThinned,stride=2)
		self.layer4 = self._make_layer(block, int(512*alpha), num_blocks[3], isCulled[3],isThinned,stride=2)
		
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(int(512*alpha)*block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, culled_stages_status,thinning_layer,stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride,culled_stages_status,thinning_layer[0],thinning_layer[1]))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		#out = F.relu(self.bn1(self.conv1(x))) 
		out = self.bn1(self.conv1(x))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.avgpool(out)
		out = torch.flatten(out, 1)
		out = self.fc(out)
		return out

## CIFAR-100 models ###
def DRD_C100_230K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,False])

def DRD_C100_115K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_C100_57K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_C100_49K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 1.0, isCulled=[True,False,False,True], isThinned=[False,True])

def DRD_C100_29K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_C100_14K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_C100_12K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 0.5, isCulled=[True,False,False,True], isThinned=[False,True])

def DRD_C100_7K(num_classes):
	return ResNet(BasicBlock, [2,1,1,1], num_classes, alpha = 0.5, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])


## TinyImageNet models ##

def DRD_TINY_918K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,False])

def DRD_TINY_459K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_TINY_393K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,True,False,False], isThinned=[False,False])

def DRD_TINY_229K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_TINY_197K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,True,False,False], isThinned=[False,True])

def DRD_TINY_115K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_TINY_98K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 1.0, isCulled=[True,True,False,False], isThinned=[False,True])

def DRD_TINY_57K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_TINY_49K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 0.5, isCulled=[True,True,False,False], isThinned=[False,True])
*/

    template <typename T>
class RESNET : public SimpleNN<T>
{
    private:
        int in_planes;
        int alpha;
        int rho;
        vector<int> identity_layers;
        vector<string> identity_layers_type;
    public:
        RESNET(int num_blocks[4], int num_classes, int alpha = 1.0, int rho = 1.0, string option = "default") {
            this->alpha = alpha;
            this->rho = rho;
            in_planes = int(64*alpha);
            this->add(new Conv2d<T>(3, in_planes, 3, int(1/rho), 1, false, option));
            this->add(new BatchNorm2d<T>());
            this->add(new ReLU<T>());
            this->make_layer(num_blocks[0], int(64*alpha), false, false, 1, option);
            this->make_layer(num_blocks[1], int(128*alpha), true, false, 2, option);
            this->make_layer(num_blocks[2], int(256*alpha), true, false, 2, option);
            this->make_layer(num_blocks[3], int(512*alpha), true, false, 2, option);
            this->add(new AdaptiveAvgPool2d<T>(1, 1));
            this->add(new Flatten<T>());
            this->add(new Linear<T>(int(512*alpha)*4, num_classes, option));
        }

        void add_identity_layer(string type) {
            this->identity_layers.push_back(this->net.size());
            this->identity_layers_type.push_back(type);
        }

        void add_block(int in_planes, int planes, string option, bool isCulled = false, bool isTopThinned = false, bool isBottomThinned = false, int stride = 1) {
            const int expansion = 1;
            this->add_identity_layer("Identity_Store");
            this->add(new Conv2d<T>(in_planes, planes, 3, stride, 1, false, option));
            this->add(new BatchNorm2d<T>());
            if (!isCulled && !isTopThinned) {
                this->add(new ReLU<T>());
            }
            this->add(new Conv2d<T>(planes, planes, 3, 1, 1, false, option));
            this->add(new BatchNorm2d<T>());
            if (stride != 1 || in_planes != planes * expansion) {
                this->add_identity_layer("Identity_OP_Start");
                this->add(new Conv2d<T>(in_planes, planes * expansion, 1, stride, 0, false, option));
                this->add(new BatchNorm2d<T>());
                this->add_identity_layer("Identity_OP_Finish");
            }
            this->add_identity_layer("Identity_ADD");
            if (!isCulled && !isBottomThinned) {
                this->add(new ReLU<T>());
            }
        }

	def __init__(self, in_planes, planes, stride=1,isCulled=False,isTopThinned=False,isBottomThinned=False):
	
        def _make_layer(self, block, planes, num_blocks, culled_stages_status,thinning_layer,stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride,culled_stages_status,thinning_layer[0],thinning_layer[1]))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

        void make_layer(int num_blocks, int planes, bool isCulled, bool thinning_layer[2], int stride, string option) {
            vector<int> strides;
            strides.push_back(stride);
            for (int i = 0; i < num_blocks - 1; i++) {
                strides.push_back(stride);
            }
            for (int stride : strides) {
                this->add_block(in_planes, planes, option, isCulled, thinning_layer[0], thinning_layer[1], stride);
                in_planes = planes * 1;
            }
        }

        void forward(const MatX<T>& X, bool is_training) override {
            MatX<T> identity = X;
            MatX<T> out = X;
            MatX<T> temp = X;
            int i = 0;
            for (int l = 0; l < this->net.size(); l++) {
                if(this->identity_layers.size() != 0 && i < this->identity_layers.size()) {
                    while(this->identity_layers[i] == l)  { 
                        if(this->identity_layers_type[i] == "Identity_Store") {
                            identity = out; //store identity of current layer

                        }
                        else if(this->identity_layers_type[i] == "Identity_OP_Start") {
                            //network starts operating on identity, storing last output
                            temp = out; 
                            out = identity;
                        }
                        else if(this->identity_layers_type[i] == "Identity_OP_Finish") {
                            //network finished processing identity, loading back last output
                            identity = out;
                            out = temp;
                        }
                        else if(this->identity_layers_type[i] == "Identity_ADD") {
                            out += identity;
                        }
                        i++;
                        if(i >= this->identity_layers.size()) {
                            break;
                        }
                    }

                }
                this->net[l]->forward(out, is_training);
                out = this->net[l]->output;
            }
        }

        void compile(vector<int> input_shape, Optimizer* optim=nullptr, Loss<T>* loss=nullptr) override
        {
            // set optimizer & loss
            this->optim = optim;
            this->loss = loss;

            // set first & last layer
            this->net.front()->is_first = true;
            this->net.back()->is_last = true;

            vector<int> identity = input_shape;
            vector<int> out = input_shape;
            vector<int> temp = input_shape;


            // set network
            int i = 0;




