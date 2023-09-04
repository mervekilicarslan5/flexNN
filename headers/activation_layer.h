#pragma once
#include "layer.h"

namespace simple_nn
{
    template<typename T>
	class Activation : public Layer<T>
	{
	protected:
		int batch;
		int channels;
		int height;
		int width;
		int out_block_size;
	public:
		Activation() : Layer<T>(LayerType::ACTIVATION), batch(0), channels(0), height(0), width(0), out_block_size(0) {}

		void set_layer(const vector<int>& input_shape) override
		{
			if (input_shape.size() == 4) {
				batch = input_shape[0];
				channels = input_shape[1];
				height = input_shape[2];
				width = input_shape[3];
				out_block_size = batch * channels * height * width;

				this->output.resize(batch * channels, height * width);
				this->delta.resize(batch * channels, height * width);
			}
			else {
				batch = input_shape[0];
				height = input_shape[1];
				out_block_size = batch * height;

				this->output.resize(batch, height);
				this->delta.resize(batch, height);
			}
		}

		void forward(const MatX<T>& prev_out, bool is_training) override { return; }

		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override { return; }

		void zero_grad() override { this->delta.setZero(); }

		vector<int> output_shape() override
		{
			if (channels == 0) return { batch, height };
			else return { batch, channels, height, width };
		}
	};

    template<typename T>
	class Tanh : public Activation<T>
	{
	public:
		Tanh() : Activation<T>() {}

		void forward(const MatX<T>& prev_out, bool is_training) override
		{
			assert(!this->is_last && "Tanh::forward(const vector<float>, bool): Hidden layer activation.");
			std::transform(
				prev_out.data(), 
				prev_out.data() + this->out_block_size, 
				this->output.data(),
				[](const float& e) { return 2 / (1 + std::exp(-2.f * e)) - 1; }
			);
		}

		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override
		{
			std::transform(
				prev_out.data(), 
				prev_out.data() + this->out_block_size, 
				this->delta.data(), 
				prev_delta.data(),
				[](const float& e1, const float& e2) {
					float tanh = 2 / (1 + std::exp(-2.f * e1)) - 1;
					return e2 * (1 - tanh * tanh);
				}
			);
		}
	};

    template<typename T>
	class Sigmoid : public Activation<T>
	{
	public:
		Sigmoid() : Activation<T>() {}

		void forward(const MatX<T>& prev_out, bool is_training) override
		{
			assert(this->is_last && "Sigmoid::forward(const vector<float>, bool): Output layer activation.");
			std::transform(
				prev_out.data(), 
				prev_out.data() + this->out_block_size, 
				this->output.data(),
				[](const float& e) { return 1 / (1 + std::exp(-e)); }
			);
		}

		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override
		{
			std::transform(
				prev_out.data(), 
				prev_out.data() + this->out_block_size, 
				this->delta.data(), 
				prev_delta.data(),
				[](const float& e1, const float& e2) {
					float sigmoid = 1 / (1 + std::exp(-e1));
					return e2 * (1 - sigmoid * sigmoid);
				}
			);
		}
	};

	float sum_exp(const float* p, int size, float max)
	{
		float out = std::accumulate(p, p + size, 0.f,
			[&](const float& sum, const float& elem) { return sum + std::exp(elem - max); });
		return out;
	}

    template<typename T>
	class Softmax : public Activation<T>
	{
	public:
		Softmax() : Activation<T>() {}

		void set_layer(const vector<int>& input_shape) override
		{
			assert(input_shape.size() == 2 && "Softmax::set_layer(const vector<int>&): Does not support 2d activation.");
			assert(this->is_last && "Softmax::set_layer(const vector<int>&): Does not support hidden layer activation.");
			this->batch = input_shape[0];
			this->height = input_shape[1];
			this->out_block_size = this->batch * this->height;
			this->output.resize(this->batch, this->height);
			this->delta.resize(this->batch, this->height);
		}

		void forward(const MatX<T>& prev_out, bool is_training) override
		{
			this->output.setZero();
			for (int n = 0; n < this->batch; n++) {
				int offset = this->height * n;
				const float* begin = prev_out.data() + offset;
				float max = *std::max_element(begin, begin + this->height);
				float sum = sum_exp(begin, this->height, max);
				std::transform(begin, begin + this->height, this->output.data() + offset,
					[&](const float& e) {
						return std::exp(e - max) / sum;
					});
			}
		}

		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override
		{
			std::copy(this->delta.data(), this->delta.data() + this->out_block_size, prev_delta.data());
		}
	};

    template<typename T>
	class ReLU : public Activation<T>
	{
	public:
		ReLU() : Activation<T>() {}

		void forward(const MatX<T>& prev_out, bool is_training) override
		{
			this->output.setZero();
			std::transform(
				prev_out.data(), 
				prev_out.data() + this->out_block_size, 
				this->output.data(),
				[](const float& e) { return std::max(0.f, e); }
			);
		}

		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override
		{
			std::transform(
				prev_out.data(), 
				prev_out.data() + this->out_block_size, 
				this->delta.data(), 
				prev_delta.data(),
				[](const float& e1, const float& e2) {
					return (e1 <= 0) ? 0 : e2;
				}
			);
		}
	};
}
