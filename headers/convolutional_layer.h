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
        int stride; // add stride
	    bool use_bias;	
        string option;
		MatX<T> dkernel;
		VecX<T> dbias;
		MatX<T> im_col;
	public:
		MatX<T> kernel;
		VecX<T> bias;
        bool quantize; 
        float scale;
        T zero_point;
		Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool quantize = false, bool use_bias = true,
			string option = "kaiming_uniform");
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
        int stride, // add stride
		int padding,
        bool quantize,
        bool use_bias,
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
        stride(stride), // add stride
		pad(padding),
        use_bias(use_bias),
        quantize(quantize),
		option(option) {}

    template<typename T>
	void Conv2d<T>::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];
		ic = input_shape[1];
		ih = input_shape[2];
		iw = input_shape[3];
		ihw = ih * iw;
		oh = calc_outsize(ih, kh, stride, pad);
		ow = calc_outsize(iw, kw, stride, pad);
		ohw = oh * ow;

		this->output.resize(batch * oc, ohw);
		this->delta.resize(batch * oc, ohw);
		kernel.resize(oc, ic * kh * kw);
		dkernel.resize(oc, ic * kh * kw);
        if (use_bias) {
            bias.resize(oc);
            dbias.resize(oc);
        }
        else {
            bias.resize(0);
            dbias.resize(0);
        }
		im_col.resize(ic * kh * kw, ohw);

		int fan_in = kh * kw * ic;
		int fan_out = kh * kw * oc;
		init_weight(kernel, fan_in, fan_out, option);
		bias.setZero();
	}

    template<typename T>
	void Conv2d<T>::forward(const MatX<T>& prev_out, bool is_training)
	{

		
        for (int i = 0; i < this->output.size(); i++) {
            this->output(i) = T(0,0);
			
        }
		for (int n = 0; n < batch; n++) {
            /* std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now(); */
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
		

    auto A = kernel.data();
    MatX<T> BM = im_col.transpose();
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
    const int TILE_SIZE = 64;


	/*int columns_to_print = 5;

	std::cout << "im2col transformed data (first " << columns_to_print << " columns):" << std::endl;
	for (int row = 0; row < im_col.rows(); ++row) {  // Iterate over all rows
		for (int col = 0; col < columns_to_print; ++col) {  // Print the first few columns
			std::cout << std::setw(10) << im_col(row, col).reveal() << " ";  // Adjust setw as needed for your data
		}
		std::cout << std::endl;  // New line after each row
	}
	std::cout << std::endl;  // Extra new line for better separation

	*/

  for (int i = 0; i < m; i += TILE_SIZE) {
      /* _mm_prefetch(A + i * f, _MM_HINT_T0); */
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            /* _mm_prefetch(B + j * f, _MM_HINT_T0); */
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    for (int jj = j; jj < j_max; ++jj) {
                    auto temp = T(0,0);
                        for (int kk = k; kk < k_max; ++kk) {
                            /* _mm_prefetch(C + ii * p + jj, _MM_HINT_T0); */
                       temp += A[ii*f+kk].prepare_dot(B[jj*f + kk]);
                        }
                        C[ii*p + jj] += temp;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
    }

            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
       /* std::cout << "before of conv 10 values of output: ";
		for (int i = 0; i < 10 && i < this->output.size(); ++i) {

			std::cout << this->output(i).reveal()<< " ";

		}*/
		if (use_bias)
            for (int n = 0; n < batch; n++) 
			    this->output.block(oc * n, 0, oc, ohw).colwise() += bias;

        if (this->quantize){
			for (int i = 0; i < this->output.size(); i++) {
				//this->output(i) = this->output(i).quantize(scale, zero_point);
				this->output(i) = this->output(i) * scale + zero_point;
				
			}
		}
	/*	std::cout << "end of conv 10 values of output: ";
		for (int i = 0; i < 10 && i < this->output.size(); ++i) {

			std::cout << this->output(i).reveal()<< " ";

		}*/
         
                

	}

    template<typename T>
	void Conv2d<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
			dkernel += this->delta.block(oc * n, 0, oc, ohw) * im_col.transpose();
            if (use_bias)
                dbias += this->delta.block(oc * n, 0, oc, ohw).rowwise().sum();
		}

		if (!this->is_first) {
			for (int n = 0; n < batch; n++) {
				T* begin = prev_delta.data() + ic * ihw * n;
				im_col = kernel.transpose() * this->delta.block(oc * n, 0, oc, ohw);
				col2im(im_col.data(), ic, ih, iw, kh, stride, pad, begin);
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
