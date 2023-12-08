#include "headers/simple_nn.h"
#include "headers/config.h"
using namespace std;
using namespace simple_nn;
using namespace Eigen;

template<typename T>
void load_model(const Config& cfg, SimpleNN<T>& model);

int main(int argc, char** argv)
{
    using DATATYPE = float;
    using FLOATTYPE = float;
    using UINTTYPE = float;
    using SHARETYPE = Wrapper<DATATYPE>;
	Config cfg;
	cfg.parse(argc, argv);
	cfg.print_config();

	int n_train = 50000, n_test = 10000, ch = 3, h = 32, w = 32;

	MatX<float> train_X, test_X;

	VecXi train_Y, test_Y;

	DataLoader<SHARETYPE> train_loader, test_loader;

    auto [train_images, train_labels] = read_cifar10_file(cifar10_train_file, n_train);
    auto [test_X, test_Y] = read_cifar10_file(cifar10_test_file, n_test);
	test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test);
	test_Y = read_mnist_label(cfg.data_dir, "t10k-labels.idx1-ubyte", n_test);
    MatX<SHARETYPE> test_XX = test_X.unaryExpr([](float val) { 
    return SHARETYPE(val);
    });
	test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test);

	cout << "Dataset loaded." << endl;

    SimpleNN<SHARETYPE> model;
	load_model(cfg, model);

	cout << "Model construction completed." << endl;


    model.compile({ cfg.batch, ch, h, w });
    model.load(cfg.save_dir, cfg.pretrained);
    model.evaluate(test_loader);

	return 0;
}



template<typename T>
void load_model(const Config& cfg, SimpleNN<T>& model)
{
	if (cfg.model == "lenet5") {
		for (int i = 0; i < 6; i++) {
			if (i < 2) {
				if (i == 0) {
					model.add(new Conv2d<T>(1, 6, 5, 2, cfg.init));
				}
				else {
					model.add(new Conv2d<T>(6, 16, 5, 0, cfg.init));
				}
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm2d<T>); */
				/* } */
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
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm1d<T>); */
				/* } */
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
			}
			else {
				model.add(new Linear<T>(84, 10, cfg.init));
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm1d<T>); */
				/* } */
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
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm1d<T>); */
				/* } */
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
			}
			else {
				model.add(new Linear<T>(150, 10, cfg.init));
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm1d<T>); */
				/* } */
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
