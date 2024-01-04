#include "headers/file_manage.h"
#include "headers/simple_nn.h"
#include "headers/config.h"
#include <cstdint>
#include <sys/types.h>
#include "architectures/CNNs.hpp"
#include "architectures/ResNetT.hpp"
#include "architectures/DeepReduce.hpp"
#include "architectures/QuantLeNet.hpp"
using namespace std;
using namespace simple_nn;
using namespace Eigen;



template<typename T>
void load_model(const Config& cfg, SimpleNN<T>& model);

int main(int argc, char** argv)
{
    using DATATYPE = int32_t;
    using FLOATTYPE = float;
    using UINTTYPE = int32_t;
    using INTTYPE = int32_t;
    using SHARETYPE = Wrapper<FLOATTYPE, INTTYPE, UINTTYPE, FRACTIONAL_VALUE, DATATYPE>;

	Config cfg;
	cfg.parse(argc, argv);
	cfg.print_config();

	int n_train = 60000, n_test = 100, ch = 1, h = 28, w = 28, num_classes = 10;


	DataLoader<SHARETYPE> train_loader, test_loader;
    

    //TODO: load your quantized dataset here (int32_t)
    /* auto test_X = read_custom_images<UINTTYPE>("./dataset/cifar10-test-images.bin", n_test, ch, h, w); */
    /* auto test_Y = read_custom_labels("./dataset/cifar10-test-labels.bin", n_test); */
    auto test_X = read_dummy_images<UINTTYPE>(n_test, ch, h, w);
    auto test_Y = read_dummy_labels(n_test);
    
    MatX<SHARETYPE> test_XX = test_X.unaryExpr([](float val) { 
    return SHARETYPE(val);
    });
	test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test);

	cout << "Dataset loaded." << endl;

    /* auto model = VGG<SHARETYPE>(num_classes); */
    auto model = QuantLeNet<SHARETYPE>(num_classes, true);


    model.compile({ cfg.batch, ch, h, w });
    cout << "Model construction completed." << endl;
    std::cout << "Loading Model Parameters..." << std::endl;
    //TODO: load your quantized model weights and biases here (int32_t)
    model.load(cfg.save_dir, cfg.pretrained); //load regular Parameters 
    //TODO: load your quantized zero point and scale here (float)
    model.load_quant(cfg.save_dir, cfg.pretrained_quant); // load quantized Parameters
    /* model.load(cfg.save_dir, "dummy"); */
    model.evaluate(test_loader);
    /* } */

	return 0;
}



template<typename T>
void load_model(const Config& cfg, SimpleNN<T>& model)
{
	if (cfg.model == "lenet5") {
		for (int i = 0; i < 6; i++) {
			if (i < 2) {
				if (i == 0) {
					model.add(new Conv2d<T>(1, 6, 5,1, 2, true, cfg.init));
				}
				else {
					model.add(new Conv2d<T>(6, 16, 5,1, 0, true, cfg.init));
				}
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm2d<T>);
				}
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
				if (cfg.pool == "max") {
					model.add(new MaxPool2d<T>(2, 2));
				}
				else {
					model.add(new AvgPool2d<T>(2, 2));
				}
			}
			else if (i == 2) {
				model.add(new Flatten<T>);
			}
			else if (i < 5) {
				if (i == 3) {
					model.add(new Linear<T>(400, 120, cfg.init));
				}
				else {
					model.add(new Linear<T>(120, 84, cfg.init));
				}
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm1d<T>);
				}
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
			}
			else {
				model.add(new Linear<T>(84, 10, cfg.init));
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm1d<T>);
				}
				if (cfg.loss == "cross_entropy") {
					model.add(new Softmax<T>);
				}
				/* else { */
				/* 	model.add(new Sigmoid<T>); */
				/* } */
			}
		}
	}
	else {
		for (int i = 0; i < 3; i++) {
			if (i < 2) {
				if (i == 0) {
					model.add(new Linear<T>(784, 500, cfg.init));
				}
				else {
					model.add(new Linear<T>(500, 150, cfg.init));
				}
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm1d<T>);
				}
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
			}
			else {
				model.add(new Linear<T>(150, 10, cfg.init));
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm1d<T>);
				}
				if (cfg.loss == "cross_entropy") {
					model.add(new Softmax<T>);
				}
				/* else { */
				/* 	model.add(new Sigmoid<T>); */
				/* } */
			}
		}
	}
}

/* int main_old(int argc, char** argv) */
/* { */
/*     using DATATYPE = float; */
/*     using FLOATTYPE = float; */
/*     using UINTTYPE = float; */
/*     using SHARETYPE = Share<DATATYPE>; */
/* 	Config cfg; */
/* 	cfg.parse(argc, argv); */
/* 	cfg.print_config(); */

/* 	int n_train = 60000, n_test = 10000, ch = 1, h = 28, w = 28; */

/* 	MatX<float> train_X, test_X; */

/* 	VecXi train_Y, test_Y; */

/* 	DataLoader<SHARETYPE> train_loader, test_loader; */

/* 	if (cfg.mode == "train") { */
/* 		train_X = read_mnist(cfg.data_dir, "train-images.idx3-ubyte", n_train); */
/* 		train_Y = read_mnist_label(cfg.data_dir, "train-labels.idx1-ubyte", n_train); */
/*         MatX<SHARETYPE> train_XX = train_X.unaryExpr([](float val) { */ 
/*     return SHARETYPE( FloatFixedConverter<FLOATTYPE, UINTTYPE, ANOTHER_FRACTIONAL_VALUE>::fixedToFloat(val)); */ 
/* }); */

/* 		train_loader.load(train_XX, train_Y, cfg.batch, ch, h, w, cfg.shuffle_train); */
/* 	} */

/* 	test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test); */
/* 	test_Y = read_mnist_label(cfg.data_dir, "t10k-labels.idx1-ubyte", n_test); */
/*     MatX<SHARETYPE> test_XX = test_X.unaryExpr([](float val) { */ 
/*     return SHARETYPE( FloatFixedConverter<FLOATTYPE, UINTTYPE, ANOTHER_FRACTIONAL_VALUE>::fixedToFloat(val)); */
/*     }); */
/* 	test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test); */

/* 	cout << "Dataset loaded." << endl; */

/* 	SimpleNN<SHARETYPE> model; */
/* 	load_model(cfg, model); */

/* 	cout << "Model construction completed." << endl; */

/* 	if (cfg.mode == "train") { */
/* 		if (cfg.loss == "cross_entropy") { */
/* 			model.compile({ cfg.batch, ch, h, w }, new SGD(cfg.lr, cfg.decay), new CrossEntropyLoss<SHARETYPE>); */
/* 		} */
/* 		else { */
/* 			model.compile({ cfg.batch, ch, h, w }, new SGD(cfg.lr, cfg.decay), new MSELoss<SHARETYPE>); */
/* 		} */
/* 		model.fit(train_loader, cfg.epoch, test_loader); */
/* 		model.save("./model_zoo", cfg.model + ".pth"); */
/* 	} */
/* 	else { */
/* 		model.compile({ cfg.batch, ch, h, w }); */
/* 		model.load(cfg.save_dir, cfg.pretrained); */
/* 		model.evaluate(test_loader); */
/* 	} */

/* 	return 0; */
/* } */

