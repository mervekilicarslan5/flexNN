#pragma once
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
		Linear(int in_features, int out_features, string option);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		void update_weight(float lr, float decay) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

    template<typename T>
	Linear<T>::Linear(int in_features, int out_features, string option) :
		Layer<T>(LayerType::LINEAR),
		batch(0),
		in_feat(in_features),
		out_feat(out_features),
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
        using ART = Wrapper<float,int64_t,uint64_t,ANOTHER_FRACTIONAL_VALUE,uint64_t>;
        MatX<ART> tmp_prev_out = prev_out.unaryExpr([](T x) { return ART(x.reveal()); });
        MatX<ART> tmp_W = W.unaryExpr([](T x) { return ART(x.reveal()); });
        RowVecX<ART> tmp_b = b.unaryExpr([](T x) { return ART(x.reveal()); });
        MatX<ART> tmp_output = this->output.unaryExpr([](T x) { return ART(x.reveal()); });
        MatX<T> tmp_output2 = this->output.unaryExpr([](T x) { return T(x.reveal()); });
            /* for (int i = 0; i < tmp_W.size(); i++) { */
        /* auto x = ART(5.3); */
        /* auto y = ART(2.1); */
        /* std::cout << "x_s1 : " << x.get_s1() << std::endl; */
        /* std::cout << "y_s1 : " << y.get_s1() << std::endl; */
        /* std::cout << "xy_s1 : " << (x*y).get_s1() << std::endl; */
        /* std::cout << "xy : " << (x*y).reveal() << std::endl; */
        /* std::cout << "x : " << x.reveal() << std::endl; */
        /* std::cout << "y : " << y.reveal() << std::endl; */
        /* auto c = x*y; */
        /* c.mask_and_send_dot(); */
        /* std::cout << "xy trunc: : " << c.reveal() << std::endl; */
                /* std::cout << "W before mult : " << tmp_W(i).reveal() << " " << W(i).reveal() << std::endl; */
            /* } */
            /* for (int i = 0; i < tmp_prev_out.size(); i++) { */
                /* std::cout << "pre_out before mult : " << tmp_prev_out(i).reveal() << " " << prev_out(i).reveal() << std::endl; */
            /* } */


		for (int n = 0; n < batch; n++) {
			/* this->output.row(n).noalias() = W * prev_out.row(n).transpose(); */
			/* this->output.row(n).noalias() = W * prev_out.row(n).transpose(); */
			tmp_output.row(n).noalias() = tmp_W * tmp_prev_out.row(n).transpose();
            /* tmp_output2.row(n).noalias() = W * prev_out.row(n).transpose(); */
        }
            for (int i = 0; i < tmp_output.size(); i++) {
                /* this->output(i).mask_and_send_dot(); */
                /* std::cout << "before trunc : " << tmp_output(i).reveal() << " " << tmp_output2(i).reveal() << std::endl; */
                tmp_output(i).mask_and_send_dot();
                /* tmp_output2(i).mask_and_send_dot(); */
                /* tmp_output(i).mask_and_send_dot(); */
                /* std::cout << "after trunc : " << tmp_output(i).reveal() << " " << tmp_output2(i).reveal() << std::endl; */
            }

		for (int n = 0; n < batch; n++) {
			/* this->output.row(n).noalias() += b; */
			tmp_output.row(n).noalias() += tmp_b;
			/* tmp_output2.row(n).noalias() += b; */
            // loop over all elements of output
		}
            /* for (int i = 0; i < tmp_output.size(); i++) { */
            /*     std::cout << "after add : " << tmp_output(i).reveal() << " " << tmp_output2(i).reveal() << std::endl; */
            /* } */
        this->output = tmp_output.unaryExpr([](ART x) { return T(x.reveal()); });
        /* tmp_output2 = tmp_output.unaryExpr([](ART x) { return T(x.reveal()); }); */
        /* this->output = tmp_output2.unaryExpr([](T x) { return T(x.reveal()); }); */
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
