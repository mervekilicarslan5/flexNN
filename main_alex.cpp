#include "headers/simple_nn.h"
#include "headers/config.h"
#include <cstdint>
#include <sys/types.h>
#include "architectures/Alexnet.hpp"
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
    using INTTYPE = float;
    using SHARETYPE = Wrapper<FLOATTYPE, INTTYPE, UINTTYPE, FRACTIONAL_VALUE, DATATYPE>;

	Config cfg;
	cfg.parse(argc, argv);
	cfg.print_config();

	int n_train = 60000, n_test = 512, ch = 3, h = 32, w = 32;

	DataLoader<SHARETYPE> train_loader, test_loader;
    int num_images = 100; // Number of images in your CIFAR-10 file

    MatX<float> test_X = read_cifar10_images("./dataset_cifar", "t10k-images.idx3-ubyte", n_test);
	VecXi test_Y = read_cifar10_labels("./dataset_cifar", "t10k-labels.idx1-ubyte", n_test);
    MatX<SHARETYPE> test_XX = test_X.unaryExpr([](float val) { 
    return SHARETYPE(val);
    });
	test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test);

	cout << "Dataset loaded." << endl;
    SimpleNN<SHARETYPE> model;
    make_alexnet(model);

	cout << "Model construction completed." << endl;
    model.compile({ cfg.batch, ch, h, w });
    std::cout << "Loading Model Parameters..." << std::endl;
    model.load(cfg.save_dir, cfg.pretrained);
    model.evaluate_quant(test_loader);

	return 0;
}

