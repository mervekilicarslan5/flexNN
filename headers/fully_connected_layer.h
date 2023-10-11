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

    void test() {
          int batch = 2;

    MatXf tmp_W(3, 2);
    MatXf tmp_prev_out(2, 2);
    MatXf tmp_output(batch, 3);

    tmp_W << 2, 3,
             4, 5,
             6, 7;

    tmp_prev_out << 9, 10,
                    11, 12;

    MatX<FCART> tmp_prev_out2 = tmp_prev_out.unaryExpr([](float x) { return FCART(x); });
    MatX<FCART> tmp_W2 = tmp_W.unaryExpr([](float x) { return FCART(x); });
    MatX<FCART> tmp_output2 = tmp_output.unaryExpr([](float x) { return FCART(x); });

    /* for (int n = 0; n < batch; n++) { */
    /* // For each row of tmp_W2 */
    /* for (int i = 0; i < tmp_W2.rows(); i++) { */
    /*     FCART sum = FCART(0); */
    /*     // Compute the dot product of the i-th row of tmp_W2 and the n-th row of tmp_prev_out2 */
    /*     for (int j = 0; j < tmp_W2.cols(); j++) { */
    /*         sum += tmp_W2(i, j) * tmp_prev_out2(n, j); */
    /*     } */
    /*     tmp_output2(n,i) = sum;  // Note: we're assigning to the row */
    /* } */
/* } */
    PRINT = true;
    for (int n = 0; n < batch; n++) {
        tmp_output2.row(n).noalias() = tmp_W2 * tmp_prev_out2.row(n).transpose();
        if (n > 0) {
        PRINT = false;
        }
    }
            /* for (int i = 0; i < tmp_output.rows(); ++i) { */
            /*     for (int j = 0; j < tmp_output.cols(); ++j) { */
            /*         tmp_output2(i, j).mask_and_send_dot(); */
            /*     } */
            /* } */

    MatXf tmp_Wr = tmp_W2.unaryExpr([](FCART x) { return x.reveal(); });
    MatXf tmp_prev_outr = tmp_prev_out2.unaryExpr([](FCART x) { return x.reveal(); });
    MatXf tmp_outputr = tmp_output2.unaryExpr([](FCART x) { return x.reveal(); });

    std::cout << "tmp_W : " << tmp_Wr << std::endl;
    std::cout << "tmp_prev_out : " << tmp_prev_outr << std::endl;
    std::cout << "tmp_output : " << tmp_outputr << std::endl;
    auto result = FCART(1) * FCART(1) + FCART(2) * FCART(0.5);
    std::cout << "result_check : " << result.reveal() << std::endl;

}

    template<typename T>
	void Linear<T>::forward(const MatX<T>& prev_out, bool is_training)
	{
        /* test(); */
        MatX<FCART> tmp_prev_out = prev_out.unaryExpr([](T x) { return FCART(x.reveal()); });
        MatX<FCART> tmp_W = W.unaryExpr([](T x) { return FCART(x.reveal()); });
        RowVecX<FCART> tmp_b = b.unaryExpr([](T x) { return FCART(x.reveal()); });
        MatX<FCART> tmp_output = this->output.unaryExpr([](T x) { return FCART(x.reveal()); });
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
			/* tmp_output.row(n).noalias() = tmp_W * tmp_prev_out.row(n).transpose(); */
            for(int i = 0; i < tmp_W.rows(); ++i) {
            FCART sum = FCART(0);
            for(int j = 0; j < tmp_W.cols(); ++j) {
                sum += (tmp_W(i, j) * tmp_prev_out(n, j));  // Use custom * and + operators
            }
            tmp_output(n, i) = sum;
        }

            // loop over all elements in tmp_output
            /* tmp_output2.row(n).noalias() = W * prev_out.row(n).transpose(); */
        }
            for (int i = 0; i < tmp_output.rows(); ++i) {
                for (int j = 0; j < tmp_output.cols(); ++j) {
                    tmp_output(i, j).mask_and_send_dot();
                }
            }
            /* for (int i = 0; i < tmp_output.size(); i++) { */
                /* this->output(i).mask_and_send_dot(); */
                /* std::cout << "before trunc : " << tmp_output(i).reveal() << " " << tmp_output2(i).reveal() << std::endl; */
                /* tmp_output(i).mask_and_send_dot(); */
                /* tmp_output2(i).mask_and_send_dot(); */
                /* tmp_output(i).mask_and_send_dot(); */
                /* std::cout << "after trunc : " << tmp_output(i).reveal() << " " << tmp_output2(i).reveal() << std::endl; */
            /* } */

		for (int n = 0; n < batch; n++) {
			/* this->output.row(n).noalias() += b; */
			tmp_output.row(n).noalias() += tmp_b;
			/* tmp_output2.row(n).noalias() += b; */
            // loop over all elements of output
		}
            /* for (int i = 0; i < tmp_output.size(); i++) { */
            /*     std::cout << "after add : " << tmp_output(i).reveal() << " " << tmp_output2(i).reveal() << std::endl; */
            /* } */
        this->output = tmp_output.unaryExpr([](FCART x) { return T(x.reveal()); });
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
