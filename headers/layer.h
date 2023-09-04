#pragma once
#include "common.h"

namespace simple_nn
{
	enum class LayerType
	{
		LINEAR,
		CONV2D,
		MAXPOOL2D,
		AVGPOOL2D,
		ACTIVATION,
		BATCHNORM1D,
		BATCHNORM2D,
		FLATTEN
	};

    template<typename T>
	class Layer
	{
	public:
		LayerType type;
		bool is_first;
		bool is_last;
		MatX<T> output;
		MatX<T> delta;
	public:
		Layer(LayerType type) : type(type), is_first(false), is_last(false) {}
		virtual void set_layer(const vector<int>& input_shape) = 0;
		virtual void forward(const MatX<T>& prev_out, bool is_training = true) = 0;
		virtual void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) = 0;
		virtual void update_weight(float lr, float decay) { return; }
		virtual void zero_grad() { return; }
		virtual vector<int> output_shape() = 0;
	};
}
