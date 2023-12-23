#pragma once
#include "layer.h"

namespace simple_nn
{
    template<typename T>
	class Identity : public Layer<T>
	{
	private:
	public:
		Identity();
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

    template<typename T>
	Identity<T>::Identity() : Layer<T>(LayerType::FLATTEN) {}

    template<typename T>
	void Identity<T>::set_layer(const vector<int>& input_shape)
	{
	}

    template<typename T>
	void Identity<T>::forward(const MatX<T>& prev_out, bool is_training)
	{
	}

    template<typename T>
	void Identity<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
	}

    template<typename T>
	void Identity<T>::zero_grad() { this->delta.setZero(); }

    template<typename T>
	vector<int> Identity<T>::output_shape() { return { batch, channels, height, width }; }
}
