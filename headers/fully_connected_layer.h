#pragma once
#include "common.h"
#include "layer.h"

namespace simple_nn
{
    template<typename T>
	class Linear : public Layer<T>
	{
	private:
		int batch;
		int in_feat;
		int out_feat;
		string option;
		MatX<T> dW;
		RowVecX<T> db;
	public:
		MatX<T> W;
		RowVecX<T> b;
        bool quantize;
        float scale;
        T zero_point;
		Linear(int in_features, int out_features, bool quantize = false, string option = "kaiming_uniform");
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		void update_weight(float lr, float decay) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

    template<typename T>
	Linear<T>::Linear(int in_features, int out_features, bool quantize, string option) :
		Layer<T>(LayerType::LINEAR),
		batch(0),
		in_feat(in_features),
		out_feat(out_features),
        quantize(quantize),
		option(option) {}

    template<typename T>
	void Linear<T>::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];

		this->output.resize(batch, out_feat);
		this->delta.resize(batch, out_feat);
		W.resize(out_feat, in_feat);
		dW.resize(out_feat, in_feat);
		b.resize(out_feat);
		db.resize(out_feat);

		init_weight(W, in_feat, out_feat, option);
		b.setZero();
	}


    template<typename T>
	void Linear<T>::forward(const MatX<T>& prev_out, bool is_training)
	{
		for (int n = 0; n < batch; n++) {
            auto lW = W.data();
            auto lprev_out = prev_out.data();
            auto loutput = this->output.data();
            auto w_rows = W.rows();
            auto w_cols = W.cols();
            auto prev_out_cols = prev_out.cols();
            auto output_cols = this->output.cols();
            for(int i = 0; i < w_rows; ++i) {
                T sum = T(0);
                for(int j = 0; j < w_cols; ++j) {
                    sum += (lW[i * w_cols + j] * lprev_out[n * prev_out_cols + j]);  // Use custom * and + operators
                }
                sum.mask_and_send_dot(); // send immediately to utilize network better
                loutput[n * output_cols + i] = sum;
            }
        }

            T::communicate();
            /* for (int i = 0; i < this->output.size(); i++) */ 
            /*     this->output(i).mask_and_send_dot(); */
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }

		    for (int n = 0; n < batch; n++) 
			    this->output.row(n).noalias() += b;
        
        if (this->quantize) {
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i) = this->output(i) * scale + zero_point;
            }
        }
		
	}

    template<typename T>
	void Linear<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
		// dW = delta(Vector) * prev_out(RowVector)
		// db = delta
		for (int n = 0; n < batch; n++) {
			dW.noalias() += this->delta.row(n).transpose() * prev_out.row(n);
			db.noalias() += this->delta.row(n);
		}

		// prev_delta = W.T * delta(Vector)
		if (!this->is_first) {
			for (int n = 0; n < batch; n++) {
				prev_delta.row(n).noalias() = W.transpose() * this->delta.row(n).transpose();
			}
		}
	}

    template<typename T>
	void Linear<T>::update_weight(float lr, float decay)
	{
		float t1 = (1 - (2 * lr * decay) / batch);
		float t2 = lr / batch;

		if (t1 != 1) {
			W *= t1;
			b *= t1;
		}

		W -= t2 * dW;
		b -= t2 * db;
	}

    template<typename T>
	void Linear<T>::zero_grad()
	{
		this->delta.setZero();
		dW.setZero();
		db.setZero();
	}

    template<typename T>
	vector<int> Linear<T>::output_shape() { return { batch, out_feat }; }
}
