#pragma once
#include "../headers/simple_nn.h"
#include <queue>

using namespace simple_nn;

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
class BasicBlock : public SimpleNN<T>
{
    private:
        int in_planes;
        int planes;
        int stride;
        bool isCulled;
        bool isTopThinned;
        bool isBottomThinned;
        int expansion = 1;
    public:
        Conv2d<T>* conv1;
        BatchNorm2d<T>* bn1;
        ReLU<T>* relu1;
        Conv2d<T>* conv2;
        BatchNorm2d<T>* bn2;
        ReLU<T>* relu2;
        Conv2d<T>* shortcut_conv;
        BatchNorm2d<T>* shortcut_bn;
        MatX<T> out;
        BasicBlock(int in_planes, int planes, int stride, bool isCulled, bool isTopThinned, bool isBottomThinned, string option) {
            this->in_planes = in_planes;
            this->planes = planes;
            this->stride = stride;
            this->isCulled = isCulled;
            this->isTopThinned = isTopThinned;
            this->isBottomThinned = isBottomThinned;
            this->shortcut_conv = nullptr;
            this->shortcut_bn = nullptr;
            this->relu1 = nullptr;
            this->relu2 = nullptr;
            
            this->conv1 = new Conv2d<T>(in_planes, planes, 3, stride, 1, false, option);
            this->bn1 = new BatchNorm2d<T>();

            if (!isCulled && !isTopThinned) {
                this->relu1 = new ReLU<T>();
            }
            this->conv2 = new Conv2d<T>(planes, planes, 3, 1, 1, false, option);
            this->bn2 = new BatchNorm2d<T>();
            if (stride != 1 || in_planes != planes * expansion) {
                this->shortcut_conv = new Conv2d<T>(in_planes, planes * expansion, 1, stride, 0, false, option);
                this->shortcut_bn = new BatchNorm2d<T>();
            }
            if (!isCulled && !isBottomThinned) {
                this->relu2 = new ReLU<T>();
            }

        }
    
	void forward(const MatX<T>& X, bool is_training) override 
        {
            out = X;
            this->conv1->forward(X, is_training);
            out = this->conv1->output;
            this->bn1->forward(out, is_training);
            out = this->bn1->output;
            if (!isCulled && !isTopThinned) {
                this->relu1->forward(out, is_training);
                out = this->relu1->output;
            }
            this->conv2->forward(out, is_training);
            out = this->conv2->output;
            this->bn2->forward(out, is_training);
            out = this->bn2->output;
            MatX<T> shortcut = X;
            if (stride != 1 || in_planes != planes * expansion) {
                this->shortcut_conv->forward(X, is_training);
                shortcut = this->shortcut_conv->output;
                this->shortcut_bn->forward(shortcut, is_training);
                shortcut = this->shortcut_bn->output;
            }
            out += shortcut;
        }

        vector<int> output_shape()
        {
            return this->bn2->output_shape();
        }

        void compile(vector<int> input_shape, Optimizer* optim=nullptr, Loss<T>* loss=nullptr) override
        {
            // set optimizer & loss
            this->optim = optim;
            this->loss = loss;

            // set first & last layer
            /* this->net.front()->is_first = true; */
            /* this->net.back()->is_last = true; */
    
            // set network
            this->conv1->set_layer(input_shape);
            this->bn1->set_layer(this->conv1->output_shape());
            if (!isCulled && !isTopThinned) {
                this->relu1->set_layer(this->bn1->output_shape());
                this->conv2->set_layer(this->relu1->output_shape());
            }
            else {
                this->conv2->set_layer(this->bn1->output_shape());
            }
            this->bn2->set_layer(this->conv2->output_shape());
            if (stride != 1 || in_planes != planes * expansion) {
                this->shortcut_conv->set_layer(input_shape);
                this->shortcut_bn->set_layer(this->shortcut_conv->output_shape());
            }
            if (!isCulled && !isBottomThinned) {
                this->relu2->set_layer(this->bn2->output_shape());
            }




        }
};

template <typename T>
class RESNET : public SimpleNN<T>
{
    private:
        int num_classes;
        int alpha;
        int rho;
        int in_planes;
        Conv2d<T>* conv1;
        BatchNorm2d<T>* bn1;
        vector<BasicBlock<T>*> layer1;
        vector<BasicBlock<T>*> layer2;
        vector<BasicBlock<T>*> layer3;
        vector<BasicBlock<T>*> layer4;
        AdaptiveAvgPool2d<T>* avgpool;
        Flatten<T>* flatten;
        Linear<T>* fc;
    public:
        RESNET(int num_blocks[4], bool isCulled[4], bool isThinned[2], int num_classes, float alpha = 1.0, float rho = 1.0, string option = "kaiming_uniform") {
            const int block_expansion = 1;
            this->alpha = alpha;
            this->rho = rho;
            this->in_planes = int(64*alpha);
            this->conv1 = new Conv2d<T>(3, this->in_planes, 3, int(1/rho), 1, false, option);
            this->bn1 = new BatchNorm2d<T>();
            this->make_layer(layer1,num_blocks[0], int(64*alpha), isCulled[0], isThinned[0], isThinned[1], 1, option);
            this->make_layer(layer2,num_blocks[1], int(128*alpha), isCulled[1], isThinned[0], isThinned[1], 2, option);
            this->make_layer(layer3,num_blocks[2], int(256*alpha), isCulled[2], isThinned[0], isThinned[1], 2, option);
            this->make_layer(layer4, num_blocks[3], int(512*alpha), isCulled[3], isThinned[0], isThinned[1], 2, option);
            this->avgpool = new AdaptiveAvgPool2d<T>(1, 1);
            this->flatten = new Flatten<T>();
            this->fc = new Linear<T>(int(512*alpha)*block_expansion, num_classes, option);
        }

         void make_layer(vector<BasicBlock<T>*>& layer, int num_blocks, int planes, bool isCulled, bool isTopThinned, bool isBottomThinned, int stride, string option) {
            const int block_expansion = 1;
            vector<int> strides;
            strides.push_back(stride);
            for (int i = 0; i < num_blocks - 1; i++) {
                strides.push_back(1);
            }
            for (int stride : strides) {
                layer.push_back(new BasicBlock<T>(this->in_planes, planes, stride, isCulled, isTopThinned, isBottomThinned, option));
                this->in_planes = planes * block_expansion;
            }
        }

		void load(string save_dir, string fname) override
        {
            this->net.push_back(this->conv1);
            this->net.push_back(this->bn1);
            //vector containing all 4 layers
            vector<vector<BasicBlock<T>*>> layers = {this->layer1, this->layer2, this->layer3, this->layer4};
            for (int i = 0; i < layers.size(); i++) {
                for (int j = 0; j < layers[i].size(); j++) {
                    this->net.push_back(layers[i][j]->conv1);
                    this->net.push_back(layers[i][j]->bn1);
                    if (layers[i][j]->relu1 != nullptr) {
                        this->net.push_back(layers[i][j]->relu1);
                    }
                    this->net.push_back(layers[i][j]->conv2);
                    this->net.push_back(layers[i][j]->bn2);
                    if (layers[i][j]->relu2 != nullptr) {
                        this->net.push_back(layers[i][j]->relu2);
                    }
                    if (layers[i][j]->shortcut_conv != nullptr) {
                        this->net.push_back(layers[i][j]->shortcut_conv);
                    }
                    if (layers[i][j]->shortcut_bn != nullptr) {
                        this->net.push_back(layers[i][j]->shortcut_bn);
                    }
                }
            }
            this->net.push_back(this->avgpool);
            this->net.push_back(this->flatten);
            this->net.push_back(this->fc);
            SimpleNN<T>::load(save_dir, fname);
        }

         void forward(const MatX<T>& X, bool is_training) override {
            MatX<T> out = X;
            this->conv1->forward(X, is_training);
            out = this->conv1->output;
            this->bn1->forward(out, is_training);
            out = this->bn1->output;
            //create vector containing all 4 layers
            vector<vector<BasicBlock<T>*>> layers = {this->layer1, this->layer2, this->layer3, this->layer4};
            for (int i = 0; i < layers.size(); i++) {
                for (int j = 0; j < layers[i].size(); j++) {
                    layers[i][j]->forward(out, is_training);
                    out = layers[i][j]->out;
                }
            }
            this->avgpool->forward(out, is_training);
            out = this->avgpool->output;
            this->flatten->forward(out, is_training);
            out = this->flatten->output;
            this->fc->forward(out, is_training);
            out = this->fc->output;
        }

         void compile(vector<int> input_shape, Optimizer* optim=nullptr, Loss<T>* loss=nullptr) override
         {
            // set optimizer & loss
            this->optim = optim;
            this->loss = loss;

            // set first & last layer
            /* this->net.front()->is_first = true; */
            /* this->net.back()->is_last = true; */
            this->conv1->is_first = true;
            this->fc->is_last = true;
    
            // set network
            this->conv1->set_layer(input_shape);
            this->bn1->set_layer(this->conv1->output_shape());
            auto last_output_shape = this->bn1->output_shape();
            //create vector containing all 4 layers
            vector<vector<BasicBlock<T>*>> layers = {this->layer1, this->layer2, this->layer3, this->layer4};
            for (int i = 0; i < layers.size(); i++) {
                for (int j = 0; j < layers[i].size(); j++) {
                    layers[i][j]->compile(last_output_shape, optim, loss);
                    last_output_shape = layers[i][j]->output_shape();
                }
            }
            this->avgpool->set_layer(last_output_shape);

            this->flatten->set_layer(this->avgpool->output_shape());
            this->fc->set_layer(this->flatten->output_shape());

            // set Loss layer
            if (loss != nullptr) {
                loss->set_layer(this->fc->output_shape());
            }

        }
};

        



    template <typename T>
class ReducedNet : public SimpleNN<T>
{
    private:
        int in_planes;
        int alpha;
        int rho;
        vector<int> identity_layers;
        vector<string> identity_layers_type;
    public:
        ReducedNet(int num_blocks[4], bool isCulled[4], bool isThinned[2], int num_classes, float alpha = 1.0, float rho = 1.0, string option = "kaiming_uniform") {
            const int block_expansion = 1;
            this->alpha = alpha;
            this->rho = rho;
            this->in_planes = int(64*alpha);
            this->add(new Conv2d<T>(3, this->in_planes, 3, int(1/rho), 1, false, option));
            this->add(new BatchNorm2d<T>());
            this->make_layer(num_blocks[0], int(64*alpha), isCulled[0], isThinned[0], isThinned[1], 1, option);
            this->make_layer(num_blocks[1], int(128*alpha), isCulled[1], isThinned[0], isThinned[1], 2, option);
            this->make_layer(num_blocks[2], int(256*alpha), isCulled[2], isThinned[0], isThinned[1], 2, option);
            this->make_layer(num_blocks[3], int(512*alpha), isCulled[3], isThinned[0], isThinned[1], 2, option);
            this->add(new AdaptiveAvgPool2d<T>(1, 1));
            this->add(new Flatten<T>());
            this->add(new Linear<T>(int(512*alpha)*block_expansion, num_classes, option));
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

        /* void add_block(int in_planes, int planes, string option, bool isCulled = false, bool isTopThinned = false, bool isBottomThinned = false, int stride = 1) { */
        /*     const int expansion = 1; */
        /*     this->add_identity_layer("Identity_Store"); */
        /*     this->add(new Conv2d<T>(in_planes, planes, 3, stride, 1, false, option)); */
        /*     this->add(new BatchNorm2d<T>()); */
        /*     if (!isCulled && !isTopThinned) { */
        /*         this->add(new ReLU<T>()); */
        /*     } */
        /*     this->add(new Conv2d<T>(planes, planes, 3, 1, 1, false, option)); */
        /*     this->add(new BatchNorm2d<T>()); */
        /*     if (stride != 1 || in_planes != planes * expansion) { */
        /*         this->add_identity_layer("Identity_OP_Start"); */
        /*         this->add(new Conv2d<T>(in_planes, planes * expansion, 1, stride, 0, false, option)); */
        /*         this->add(new BatchNorm2d<T>()); */
        /*         this->add_identity_layer("Identity_OP_Finish"); */
        /*     } */
        /*     this->add_identity_layer("Identity_ADD"); */
        /*     if (!isCulled && !isBottomThinned) { */
        /*         this->add(new ReLU<T>()); */
        /*     } */
        /* } */

        void make_layer(int num_blocks, int planes, bool isCulled, bool isTopThinned, bool isBottomThinned, int stride, string option) {
            const int block_expansion = 1;
            vector<int> strides;
            strides.push_back(stride);
            for (int i = 0; i < num_blocks - 1; i++) {
                strides.push_back(1);
            }
            for (int stride : strides) {
                this->add_block(this->in_planes, planes, option, isCulled, isTopThinned, isBottomThinned, stride);
                this->in_planes = planes * block_expansion;
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
RESNET<T> DRD_C100_230K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,false};
    return RESNET<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 1.0);
}

template <typename T>
RESNET<T> DRD_C100_115K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,true};
    return  RESNET<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 1.0);
}

template <typename T>
RESNET<T> DRD_C100_57K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,true};
    return  RESNET<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 1.0);
}

template <typename T>
RESNET<T> DRD_C100_49K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,true};
    bool isThinned[2] = {false,true};
    return  RESNET<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 1.0);
}

template <typename T>
RESNET<T> DRD_C100_29K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,true};
    return  RESNET<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 0.5);
}

template <typename T>
RESNET<T> DRD_C100_14K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,true};
    return  RESNET<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 0.5);
}

template <typename T>
RESNET<T> DRD_C100_12K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,true};
    bool isThinned[2] = {false,true};
    return  RESNET<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 0.5);
}

template <typename T>
RESNET<T> DRD_C100_7K(int num_classes) {
    int num_blocks[4] = {2,1,1,1};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,true};
    return  RESNET<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 0.5);
}

template <typename T>
ReducedNet<T> DRD_TINY_918K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 1.0);
}

template <typename T>
ReducedNet<T> DRD_TINY_459K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 1.0);
}

template <typename T>
ReducedNet<T> DRD_TINY_393K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,true,false,false};
    bool isThinned[2] = {false,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 1.0);
}

template <typename T>
ReducedNet<T> DRD_TINY_229K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 1.0);
}

template <typename T>
ReducedNet<T> DRD_TINY_197K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,true,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 1.0);
}

template <typename T>
ReducedNet<T> DRD_TINY_115K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 0.5);
}

template <typename T>
ReducedNet<T> DRD_TINY_98K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,true,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 1.0);
}

template <typename T>
ReducedNet<T> DRD_TINY_57K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 0.5);
}

template <typename T>
ReducedNet<T> DRD_TINY_49K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,true,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 0.5);
}



