#pragma once
#include "layer.h"

namespace simple_nn
{
    template<typename T>
	class Conv2d : public Layer<T>
	{
	private:
		int batch;
		int ic;
		int oc;
		int ih;
		int iw;
		int ihw;
		int oh;
		int ow;
		int ohw;
		int kh;
		int kw;
		int pad;
		string option;
		MatX<T> dkernel;
		VecX<T> dbias;
		MatX<T> im_col;
	public:
		MatX<T> kernel;
		VecX<T> bias;
		Conv2d(int in_channels, int out_channels, int kernel_size, int padding,
			string option);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		void update_weight(float lr, float decay) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

    template<typename T>
	Conv2d<T>::Conv2d(
		int in_channels,
		int out_channels,
		int kernel_size,
		int padding,
		string option
	) :
		Layer<T>(LayerType::CONV2D),
		batch(0),
		ic(in_channels),
		oc(out_channels),
		ih(0),
		iw(0),
		ihw(0),
		oh(0),
		ow(0),
		ohw(0),
		kh(kernel_size),
		kw(kernel_size),
		pad(padding),
		option(option) {}

    template<typename T>
	void Conv2d<T>::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];
		ic = input_shape[1];
		ih = input_shape[2];
		iw = input_shape[3];
		ihw = ih * iw;
		oh = calc_outsize(ih, kh, 1, pad);
		ow = calc_outsize(iw, kw, 1, pad);
		ohw = oh * ow;

		this->output.resize(batch * oc, ohw);
		this->delta.resize(batch * oc, ohw);
		kernel.resize(oc, ic * kh * kw);
		dkernel.resize(oc, ic * kh * kw);
		bias.resize(oc);
		dbias.resize(oc);
		im_col.resize(ic * kh * kw, ohw);

		int fan_in = kh * kw * ic;
		int fan_out = kh * kw * oc;
		init_weight(kernel, fan_in, fan_out, option);
		bias.setZero();
	}

    template<typename T>
	void Conv2d<T>::forward(const MatX<T>& prev_out, bool is_training)
	{
        using ART = Wrapper<float,int64_t,uint64_t,ANOTHER_FRACTIONAL_VALUE,uint64_t>;
        /* using ART = Wrapper<float,float,float,ANOTHER_FRACTIONAL_VALUE,float>; */
        MatX<ART> tmp_output = this->output.unaryExpr([](T x) { return ART(x.reveal()); });
        MatX<ART> tmp_prev_out = prev_out.unaryExpr([](T x) { return ART(x.reveal()); });
        MatX<ART> tmp_kernel = kernel.unaryExpr([](T x) { return ART(x.reveal()); });
        VecX<ART> tmp_bias = bias.unaryExpr([](T x) { return ART(x.reveal()); });
        MatX<ART> tmp_im_col = im_col.unaryExpr([](T x) { return ART(x.reveal()); });
		for (int n = 0; n < batch; n++) {
			/* const T* im = prev_out.data() + (ic * ihw) * n; */
			/* im2col(im, ic, ih, iw, kh, 1, pad, im_col.data()); */
			/* this->output.block(oc * n, 0, oc, ohw).noalias() = kernel * im_col; */
			const ART* im = tmp_prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, tmp_im_col.data());
			tmp_output.block(oc * n, 0, oc, ohw).noalias() = tmp_kernel * tmp_im_col;
            

            /* this->output.block(oc * n, 0, oc, ohw).unaryExpr([](T &val) { val.mask_and_send_dot(); return val; }); */
            //loop over output block
            /* for (int i = 0; i < this->output.block(oc * n, 0, oc, ohw).rows(); i++) { */
            /*     for (int j = 0; j < this->output.block(oc * n, 0, oc, ohw).cols(); j++) { */
            /*         this->output.block(oc * n, 0, oc, ohw)(i, j).mask_and_send_dot(); */
            /*     } */
            /* } */
        }
            for (int i = 0; i < this->output.size(); i++) {
                /* this->output(i).mask_and_send_dot(); */
                tmp_output(i).mask_and_send_dot();
            }
		for (int n = 0; n < batch; n++) {
			/* this->output.block(oc * n, 0, oc, ohw).colwise() += bias; */
			tmp_output.block(oc * n, 0, oc, ohw).colwise() += tmp_bias;
		}
        this->output = tmp_output.unaryExpr([](ART x) { return T(x.reveal()); });
	}

    template<typename T>
	void Conv2d<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
			dkernel += this->delta.block(oc * n, 0, oc, ohw) * im_col.transpose();
			dbias += this->delta.block(oc * n, 0, oc, ohw).rowwise().sum();
		}

		if (!this->is_first) {
			for (int n = 0; n < batch; n++) {
				T* begin = prev_delta.data() + ic * ihw * n;
				im_col = kernel.transpose() * this->delta.block(oc * n, 0, oc, ohw);
				col2im(im_col.data(), ic, ih, iw, kh, 1, pad, begin);
			}
		}
	}

    template<typename T>
	void Conv2d<T>::update_weight(float lr, float decay)
	{
		float t1 = (1 - (2 * lr * decay) / batch);
		float t2 = lr / batch;

		if (t1 != 1) {
			kernel *= t1;
			bias *= t1;
		}

		kernel -= t2 * dkernel;
		bias -= t2 * dbias;
	}

    template<typename T>
	void Conv2d<T>::zero_grad()
	{
		this->delta.setZero();
		dkernel.setZero();
		dbias.setZero();
	}

    template<typename T>
	vector<int> Conv2d<T>::output_shape() { return { batch, oc, oh, ow }; }
}
