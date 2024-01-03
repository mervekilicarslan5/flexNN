#include "headers/simple_nn.h"
#include "headers/config.h"
#include "architectures/Lenet.hpp" // Your quantized LeNet header
#include <cstdint>
#include <sys/types.h>
using namespace std;
using namespace simple_nn;
using namespace Eigen;

template<typename T>
void load_model(const Config& cfg, SimpleNN<T>& model);

int main(int argc, char** argv)
{
    using DATATYPE = float;
    using FLOATTYPE = float;
    using UINTTYPE = unsigned int;
    using INTTYPE = int;
    using SHARETYPE = Wrapper<FLOATTYPE, INTTYPE, UINTTYPE, FRACTIONAL_VALUE, DATATYPE>;


	int n_train = 60000, n_test = 10000, ch = 1, h = 28, w = 28;
    DataLoader<SHARETYPE> train_loader, test_loader;
    int num_images = 512; 

    Config cfg;
    cfg.parse(argc, argv);
    cfg.print_config();

    MatX<float> test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test);
	VecXi test_Y = read_mnist_label(cfg.data_dir, "t10k-labels.idx1-ubyte", n_test);
    MatX<SHARETYPE> test_XX = test_X.unaryExpr([](float val) { 
    return SHARETYPE(val);
    });
	test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test);

    SimpleNN<SHARETYPE> model; // Declare your model

    // Parameters for each layer in LeNet (example values, replace with actual quantization parameters)
    float conv0_scale = 0.08070193976163864, conv0_zero_point = 123;
    float conv3_scale = 0.17637449502944946, conv3_zero_point = 117;
    float fc0_scale = 0.15647564828395844, fc0_zero_point = 107;
    float fc2_scale = 0.08906485885381699, fc2_zero_point = 112;
    float fc4_scale = 0.14233525097370148, fc4_zero_point = 115;
    float quant_scale = 0.0078, quant_zero_point = 128; // Overall quantization parameters

    // Initialize the quantized LeNet model with the specified parameters
    model = LeNet<SHARETYPE>(
        conv0_scale, conv0_zero_point,
        conv3_scale, conv3_zero_point,
        fc0_scale, fc0_zero_point,
        fc2_scale, fc2_zero_point,
        fc4_scale, fc4_zero_point,
        quant_scale, quant_zero_point
    );

    cout << "Quantized LeNet model initialized." << endl;

    model.compile({ cfg.batch, ch, h, w });
    std::cout << "Loading Model Parameters..." << std::endl;
    model.load(cfg.save_dir, cfg.pretrained);
    model.evaluate_quant(test_loader);
    return 0;
}
