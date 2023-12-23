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
		Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool use_bias = true,
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
            /* std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now(); */
            /* std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1); */
            /* std::cout << "im2col time: " << time_span.count() << " seconds." << std::endl; */
			/* tmp_output.block(oc * n, 0, oc, ohw).noalias() = tmp_kernel * tmp_im_col; */
            /* t1 = std::chrono::high_resolution_clock::now(); */
            /* for(int i = 0; i < oc; ++i) { */
        /* for(int k = 0; k < kernel.cols(); ++k) { */
            /* T temp = kernel(i, k); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*         this->output(oc * n + i, j) += temp * im_col(k, j);  // Use custom * and + operators */
            /*         } */
            /*     } */
            /* } */
              // For each chunk of columns
    /* auto A = im_col.data(); */
    /* auto B = kernel.data(); */
    /* auto C = this->output.data(); */
    /* auto A = kernel; */
    /* auto B = im_col; */
    /* auto C = this->output; */
    /* const int Mtile = 16; // tile size in M dimension */
    /* const int Ntile = 16; // tile size in N dimension */
    /* const int M = oc; */
    /* const int N = ohw; */ 
    /* const int K = kernel.cols(); */ 

    /* for (int m = 0; m < M; m += Mtile)                // iterate over M dimension */
    /* for (int n = 0; n < N; n += Ntile)            // iterate over N dimension */
    /*     for (int k = 0; k < K; ++k) */
    /*         for (int i = 0; i < Mtile; ++i)       // compute one tile */ 
    /*             for (int j = 0; j < Ntile; ++j) { */
    /*                 int row = m + i; */
    /*                 int col = n + j; */
    /*                 C(row,col) += A(row,k) * B(k,col); */
    /*                 /1* C[row][col] += A[row][k] * B[k][col]; *1/ */
    /*             } */
  /* for (std::size_t col_chunk = 0; col_chunk < ohw; col_chunk += 16) */
  /*   // For each row in that chunk of columns... */
  /*   for (std::size_t row = 0; row < oc; row++) */
  /*     // For each block of elements in this row of this column chunk */
  /*     // Solve for 16 elements at a time */
  /*     for (std::size_t tile = 0; tile < kernel.cols(); tile += 16) */
  /*       // For each row in the tile */
  /*       for (std::size_t tile_row = 0; tile_row < 16; tile_row++) */
  /*         // Solve for each element in this tile row */
  /*         for (std::size_t idx = 0; idx < 16; idx++) */
  /*           /1* C[row * N + col_chunk + idx] += *1/ */
  /*           /1*     A[row * N + tile + tile_row] * *1/ */
  /*           /1*     B[tile * N + tile_row * N + col_chunk + idx]; *1/ */
  /*           C[row * ohw + col_chunk + idx] += */
  /*               A[row * ohw + tile + tile_row] * */
  /*               B[tile * ohw + tile_row * ohw + col_chunk + idx]; */
			/* this->output.block(oc * n, 0, oc, ohw).noalias() = kernel * im_col; */
    /* auto A = kernel; */
    /* auto B = im_col; */
    /* auto C = this->output; */
    /* const int Mtile = std::min(oc, 16); // tile size in M dimension */
    /* const int Ntile = std::min(ohw, 16); // tile size in N dimension */

    /* const int TILE_SIZE = 64; */
    /* const int M = oc; */
    /* const int N = ohw; */ 
    /* const int K = kernel.cols(); */ 
    /* auto A = kernel.data(); */
    /* auto B = im_col.data(); */
    /* auto C = this->output.data() + (oc * ohw) * n; */
    /* /1* std::cout << "N: " << N << " M: " << M << " K: " << K << " Mtile: " << Mtile << " Ntile: " << Ntile << std::endl; *1/ */

    /* for (int m = 0; m < M; m += TILE_SIZE)                // iterate over M dimension */
    /* { */
    /*     int m_max = std::min(m + TILE_SIZE, M); */
    /* for (int q = 0; q < N; q += TILE_SIZE)            // iterate over N dimension */
    /*                                                 { */
    /*     int q_max = std::min(q + TILE_SIZE, N); */
    /*     for (int k = 0; k < K; ++k){ */
    /*         for (int i = 0; i < m_max; ++i) {      // compute one tile */ 
    /*                 int row = m + i; */
    /*             for (int j = 0; j < q_max; ++j) { */
    /*                 int col = q + j; */
    /*                 /1* C(n*oc + row,col) += A(row,k) * B(k,col); *1/ */
    /*                 /1* this->output(n*oc + row,col += kernel(row,k) * im_col(k,col)); *1/ */
    /*                C[row * N + col] += A[row * K + k] * B[k * N + col]; */
    /*                 /1* C[row][col] += A[row][k] * B[k][col]; *1/ */
    /*             } */
    /*           } */
            
    /*         } */
    /*         for (int i = 0; i < m_max; ++i) {      // compute one tile */ 
    /*                 int row = m + i; */
    /*             for (int j = 0; j < q_max; ++j) { */
    /*                 int col = q + j; */
    /*                 /1* C(n*oc + row,col) += A(row,k) * B(k,col); *1/ */
    /*                 /1* this->output(n*oc + row,col += kernel(row,k) * im_col(k,col)); *1/ */
    /*     C[row * N + col].mask_and_send_dot(); */
    /*                 /1* C[row][col] += A[row][k] * B[k][col]; *1/ */
    /*             } */
    /*           } */
        
    /* } */
    /* } */
// TILING ---
            /* auto A = kernel.data(); */
    /* auto B = im_col.data(); */
    /* auto C = this->output.data() + (oc * ohw) * n; */
    /* const int TILE_SIZE = 64; */
    /* const int m = oc; */
    /* const int f = kernel.cols(); */
    /* const int p = ohw; */
  /* for (int i = 0; i < m; i += TILE_SIZE) { */
        /* int i_max = std::min(i + TILE_SIZE, m); */
        /* for (int j = 0; j < p; j += TILE_SIZE) { */
            /* int j_max = std::min(j + TILE_SIZE, p); */
            /* for (int k = 0; k < f; k += TILE_SIZE) { */
            /*     int k_max = std::min(k + TILE_SIZE, f); */
            /*             for (int kk = k; kk < k_max; ++kk) { */
            /*         const int row2 = kk*p; */
            /*     for (int ii = i; ii < i_max; ++ii) { */
            /*         const int row = ii*p; */
            /*         /1* const int row2 = ii*f+kk; *1/ */
            /*         auto temp = A[ii*f+kk]; */
            /*         for (int jj = j; jj < j_max; ++jj) { */
            /*            C[row + jj] += temp * B[row2 + jj]; */ 
            /*             } */
            /*         } */
            /*     } */
            /* } */
            /* for (int ii = i; ii < i_max; ++ii) { */
            /*     const int row = ii*p; */
            /*     for (int jj = j; jj < j_max; ++jj) { */
            /*         C[row + jj].mask_and_send_dot(); */
            /*     } */
            /* } */
        /* } */
    /* } */
            auto A = kernel.data();
    /* auto B = im_col.transpose().data(); */
    MatX<T> BM = im_col.transpose();
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
    const int TILE_SIZE = 64;


  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                    for (int jj = j; jj < j_max; ++jj) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                        T temp_sum = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                       temp_sum += A[ii*f+kk] * B[jj*f + kk]; 
                        }
                        C[ii*p + jj] += temp_sum;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                for (int jj = j; jj < j_max; ++jj) {
                    C[ii*p + jj].mask_and_send_dot();
                }
            }
        }
    }




        

            /* for(int k = 0; k < kernel.cols(); ++k) { */
        /*     for(int i = 0; i < oc; ++i) { */
        /*         for(int j = 0; j < ohw; ++j) { */
        /*             this->output(oc *n + i, j) += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        /*             } */
        /*         } */
        /* } */

        /* t2 = std::chrono::high_resolution_clock::now(); */
        /* time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1); */
        /* std::cout << "matmul time: " << time_span.count() << " seconds." << std::endl; */
            

        }
        /* std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now(); */
            /* for (int i = 0; i < this->output.size(); i++) { */
            /*     this->output(i).mask_and_send_dot(); */
            /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
            /* std::cout << "Output size: " << this->output.size() << std::endl; */
		if (use_bias)
            for (int n = 0; n < batch; n++) 
			    this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
		
        /* auto t2 = std::chrono::high_resolution_clock::now(); */
        /* auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1); */
        /* std::cout << "mask and bias time: " << time_span.count() << " seconds." << std::endl; */
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
