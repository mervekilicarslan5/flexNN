#pragma once
#include "fully_connected_layer.h"
#include "convolutional_layer.h"
#include "max_pooling_layer.h"
#include "average_pooling_layer.h"
#include "adaptive_average_pooling_layer.h"
#include "activation_layer.h"
#include "batch_normalization_1d_layer.h"
#include "batch_normalization_2d_layer.h"
#include "flatten_layer.h"
#include "loss_layer.h"
#include "optimizers.h"
#include "data_loader.h"
#include "file_manage.h"

namespace simple_nn
{
    template<typename T>
	class SimpleNN
	{
	public:
		vector<Layer<T>*> net;
		Optimizer* optim;
		Loss<T>* loss;
	public:
		void add(Layer<T>* layer);
		virtual void compile(vector<int> input_shape, Optimizer* optim=nullptr, Loss<T>* loss=nullptr);
		void fit(const DataLoader<T>& train_loader, int epochs, const DataLoader<T>& valid_loader);
		void save(string save_dir, string fname);
		virtual void load(string save_dir, string fname);
        virtual void load_quant(string save_dir, string fname);
		void evaluate(const DataLoader<T>& data_loader);
        MatX<T> forward_return(const MatX<T>& X, bool is_training);
	private:
		virtual void forward(const MatX<T>& X, bool is_training);
		void classify(const MatX<T>& output, VecXi& classified);
		/* void error_criterion(const VecXi& classified, const VecXi& labels, T& error_acc); */
		void error_criterion(const VecXi& classified, const VecXi& labels, float& error_acc);
		void loss_criterion(const MatX<T>& output, const VecXi& labels, T& loss_acc);
		void zero_grad();
		void backward(const MatX<T>& X);
		void update_weight();
		int count_params();
        int count_quant_params();
        template<typename S>
		void write_or_read_params(S& fs, string mode);
        template<typename S>
        void write_or_read_quant_params(S& fs, string mode);
	};

    template<typename T>
	void SimpleNN<T>::add(Layer<T>* layer) { net.push_back(layer); }

    template<typename T>
	void SimpleNN<T>::compile(vector<int> input_shape, Optimizer* optim, Loss<T>* loss)
	{
		// set optimizer & loss
		this->optim = optim;
		this->loss = loss;

		// set first & last layer
		net.front()->is_first = true;
		net.back()->is_last = true;

		// set network
		for (int l = 0; l < net.size(); l++) {
			if (l == 0) net[l]->set_layer(input_shape);
			else net[l]->set_layer(net[l - 1]->output_shape());
		}

		// set Loss layer
		if (loss != nullptr) {
			loss->set_layer(net.back()->output_shape());
		}
	}

    template<typename T>
	void SimpleNN<T>::fit(const DataLoader<T>& train_loader, int epochs, const DataLoader<T>& valid_loader)
	{
		if (optim == nullptr || loss == nullptr) {
			cout << "The model must be compiled before fitting the data." << endl;
			exit(1);
		}

		int batch = train_loader.input_shape()[0];
		int n_batch = train_loader.size();

		MatX<T> X;
		VecXi Y;
		VecXi classified(batch);

		for (int e = 0; e < epochs; e++) {
			T loss(0);
			/* T error(0); */
            float error(0);

			system_clock::time_point start = system_clock::now();
			for (int n = 0; n < n_batch; n++) {
				X = train_loader.get_x(n);
				Y = train_loader.get_y(n);

				forward(X, true);
				classify(net.back()->output, classified);
				error_criterion(classified, Y, error);

				zero_grad();
				loss_criterion(net.back()->output, Y, loss);
				backward(X);
				update_weight();

				cout << "[Epoch:" << setw(3) << e + 1 << "/" << epochs << ", ";
				cout << "Batch: " << setw(4) << n + 1 << "/" << n_batch << "]";

				if (n + 1 < n_batch) {
					cout << "\r";
				}
			}
			system_clock::time_point end = system_clock::now();
			duration<float> sec = end - start;

			T loss_valid(0); 
			/* T error_valid(0); */
			float error_valid(0);

			int n_batch_valid = valid_loader.size();
			if (n_batch_valid != 0) {
				for (int n = 0; n < n_batch_valid; n++) {
					X = valid_loader.get_x(n);
					Y = valid_loader.get_y(n);

					forward(X, false);
					classify(net.back()->output, classified);
					error_criterion(classified, Y, error_valid);
					loss_criterion(net.back()->output, Y, loss_valid);
				}
			}

			cout << fixed << setprecision(2);
			cout << " - t: " << sec.count() << 's';
			cout << " - loss: " << (loss).reveal();
			/* cout << " - error: " << (error / n_batch).reveal() * 100 << "%"; */
			cout << " - error: " << (error / n_batch) * 100 << "%";
			if (n_batch_valid != 0) {
				cout << " - loss(valid): " << loss_valid.reveal() / n_batch_valid;
				/* cout << " - error(valid): " << (error_valid / n_batch_valid).reveal() * 100 << "%"; */
				cout << " - error(valid): " << (error_valid / n_batch_valid) * 100 << "%";
			}
			cout << endl;
		}
	}
		
    template<typename T>
	MatX<T> SimpleNN<T>::forward_return(const MatX<T>& X, bool is_training)
	{
        MatX<T> x = X;
		for (int l = 0; l < net.size(); l++) 
            x = net[l]->forward_return(x, is_training);
        return x;
    }


    template<typename T>
	void SimpleNN<T>::forward(const MatX<T>& X, bool is_training)
	{
		for (int l = 0; l < net.size(); l++) {
            /* std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now(); */
			if (l == 0) net[l]->forward(X, is_training);
			else net[l]->forward(net[l - 1]->output, is_training);
            
            

            /* std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now(); */
            /* std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); */
            /* std::cout << "Layer " << toString(net[l]->type) << " took " << time_span.count() << " seconds.\n"; */
		}
	}

    template<typename T>
	void SimpleNN<T>::classify(const MatX<T>& output, VecXi& classified)
	{
		// assume that the last layer is linear, not 2d.
		assert(output.rows() == classified.size());

        //loop over all elements in output and save them in float Matrix
        MatXf output_float(output.rows(), output.cols());
        for (int i = 0; i < output.rows(); i++) {
            for (int j = 0; j < output.cols(); j++) {
                output_float(i,j) = output(i,j).reveal_float();
            }
        }
		
        for (int i = 0; i < classified.size(); i++) {
			/* output.row(i).maxCoeff(&classified[i]); */
			output_float.row(i).maxCoeff(&classified[i]);
		}
    }

    template<typename T>
	void SimpleNN<T>::error_criterion(const VecXi& classified, const VecXi& labels, float& error_acc)
	/* void SimpleNN<T>::error_criterion(const VecXi& classified, const VecXi& labels, T& error_acc) */
	{
		int batch = (int)classified.size();

		/* T error(0); */
        int error(0);
		for (int i = 0; i < batch; i++) {
            std::cout << "classified[i] = " << classified[i] << " labels[i] = " << labels[i] << "\n";
			if (classified[i] != labels[i]) 
            {
                error+=1;
                /* error+=T(1); */
                /* std::cout << "classified[i] = " << classified[i] << " labels[i] = " << labels[i] << "\n"; */
		    }
        }
		/* error_acc += error / batch; */
        error_acc += error;
	}

    template<typename T>
	void SimpleNN<T>::loss_criterion(const MatX<T>& output, const VecXi& labels, T& loss_acc)
	{
		loss_acc += loss->calc_loss(output, labels, net.back()->delta);
	}

    template<typename T>
	void SimpleNN<T>::zero_grad()
	{
		for (const auto& l : net) l->zero_grad();
	}

    template<typename T>
	void SimpleNN<T>::backward(const MatX<T>& X)
	{
		for (int l = (int)net.size() - 1; l >= 0; l--) {
			if (l == 0) {
				MatX<T> empty;
				net[l]->backward(X, empty);
			}
			else {
				net[l]->backward(net[l - 1]->output, net[l - 1]->delta);
			}
		}
	}

    template<typename T>
	void SimpleNN<T>::update_weight()
	{
		float lr = optim->lr();
		float decay = optim->decay();
		for (const auto& l : net) {
			l->update_weight(lr, decay);
		}
	}

    template<typename T>
	void SimpleNN<T>::save(string save_dir, string fname)
	{
		string path = save_dir + "/" + fname;
		fstream fout(path, ios::out | ios::binary);

		int total_params = count_params();
		fout.write((char*)&total_params, sizeof(int));

		write_or_read_params(fout, "write");
		cout << "Model parameters are saved in " << path << endl;

		fout.close();

		return;
	}
    
    template<typename T>
	void SimpleNN<T>::load_quant(string save_dir, string fname)
	{
        if("dummy" == fname)
        {
            //create fake filestream with total_params*4 bytes
            int k = count_quant_params();
              std::stringstream ss;
    // Generate k random floating point numbers and write them to the stringstream
    for (int i = 0; i < k; ++i) {
        /* ss << static_cast<float>(rand()) / RAND_MAX << " "; */
        ss << 1.0f << " ";
    }

    // Create a file stream and open it in read/write mode
    ss.seekg(0, std::ios::beg); 
    write_or_read_quant_params(ss, "read");
	cout << "Pretrained dummy quant weights are loaded." << endl;
    return;

        }
		string path = save_dir + "/" + fname;
		fstream fin(path, ios::in | ios::binary);

		if (!fin) {
			cout << path << " does not exist." << endl;
			exit(1);
		}

		int total_params;
		fin.read((char*)&total_params, sizeof(int32_t));

		if (total_params != count_quant_params()) {
			fin.close();
			exit(1);
		}

		write_or_read_quant_params(fin, "read");
		fin.close();

		cout << "Pretrained quant weights are loaded." << endl;
        

		return;
	}

    template<typename T>
	void SimpleNN<T>::load(string save_dir, string fname)
	{
        if("dummy" == fname)
        {
            //create fake filestream with total_params*4 bytes
            int k = count_params();
              std::stringstream ss;
    // Generate k random floating point numbers and write them to the stringstream
    for (int i = 0; i < k; ++i) {
        /* ss << static_cast<float>(rand()) / RAND_MAX << " "; */
        ss << 1.0f << " ";
    }

    // Create a file stream and open it in read/write mode
    ss.seekg(0, std::ios::beg); 
    write_or_read_params(ss, "read");
	cout << "Pretrained dummy weights are loaded." << endl;
    return;

        }
		string path = save_dir + "/" + fname;
		fstream fin(path, ios::in | ios::binary);

		if (!fin) {
			cout << path << " does not exist." << endl;
			exit(1);
		}

		int total_params;
		fin.read((char*)&total_params, sizeof(int32_t));

		if (total_params != count_params()) {
			/* cout << "The number of parameters does not match." << "\n"; */
            /* cout << "Loaded Parameters = " << total_params << " Compiled Parameters = " << count_params() << "\n"; */
			fin.close();
                        /* cout << " " << total_params << " " << count_params() << "\n"; */

			exit(1);
		}

		write_or_read_params(fin, "read");
		fin.close();

		cout << "Pretrained weights are loaded." << endl;
        

		return;
	}
    template<typename T>
	int SimpleNN<T>::count_quant_params()
	{
        int total_params = 0;
		for (const Layer<T>* l : net) {
			if (l->type == LayerType::LINEAR) {
				const Linear<T>* lc = dynamic_cast<const Linear<T>*>(l);
                total_params += 2;
			}
			else if (l->type == LayerType::CONV2D) {
				const Conv2d<T>* lc = dynamic_cast<const Conv2d<T>*>(l);
                total_params += 2;
			}

    }
    return total_params;
    }

    template<typename T>
	int SimpleNN<T>::count_params()
	{
		int total_params = 0;
        int counter = 0;
		for (const Layer<T>* l : net) {
            /* std::cout << "\n"; */
            std::cout << "Layer: " << counter << ", " << "Layer Type: " << toString(l->type);
            counter++;
			if (l->type == LayerType::LINEAR) {
				const Linear<T>* lc = dynamic_cast<const Linear<T>*>(l);
				total_params += (int)lc->W.size();
				total_params += (int)lc->b.size();
                total_params+= 1; // zero point
                std::cout << ": " << (int)lc->W.size() << " " << (int)lc->b.size() << "\n";
			}
			else if (l->type == LayerType::CONV2D) {
				const Conv2d<T>* lc = dynamic_cast<const Conv2d<T>*>(l);
				total_params += (int)lc->kernel.size();
				total_params += (int)lc->bias.size();
                total_params+= 1; // zero point
                std::cout << " : " << (int)lc->kernel.size() << " " << (int)lc->bias.size() << "\n";
			}
			else if (l->type == LayerType::BATCHNORM1D) {
				const BatchNorm1d<T>* lc = dynamic_cast<const BatchNorm1d<T>*>(l);
				total_params += (int)lc->move_mu.size();
				total_params += (int)lc->move_var.size();
				total_params += (int)lc->gamma.size();
				total_params += (int)lc->beta.size();
                std::cout << " : " << (int)lc->move_mu.size() << " " << (int)lc->move_var.size() << " " << (int)lc->gamma.size() << " " << (int)lc->beta.size() << "\n";
			}
			else if (l->type == LayerType::BATCHNORM2D) {
				const BatchNorm2d<T>* lc = dynamic_cast<const BatchNorm2d<T>*>(l);
				total_params += (int)lc->move_mu.size();
				total_params += (int)lc->move_var.size();
				total_params += (int)lc->gamma.size();
				total_params += (int)lc->beta.size();
                std::cout << " : " << (int)lc->move_mu.size() << " " << (int)lc->move_var.size() << " " << (int)lc->gamma.size() << " " << (int)lc->beta.size() << "\n";
			}
			else {
                std::cout << "\n";
				continue;
			}
		}
		return total_params;
	}

    template<typename T>
template<typename S>
void SimpleNN<T>::write_or_read_quant_params(S& fs, string mode)
{
    for (Layer<T>* l : net) {
        if (l->type == LayerType::LINEAR) {
            Linear<T>* lc = dynamic_cast<Linear<T>*>(l);
                if (lc->quantize) 
                {
                    float scale;
                    /* float zero_point; */
                    fs.read((char*) &scale, sizeof(float));
                    /* fs.read((char*) &zero_point, sizeof(float)); */
                    lc->scale = scale;
                    /* lc->zero_point = zero_point; */
                }
                }
        else if (l->type == LayerType::CONV2D) {
            Conv2d<T>* lc = dynamic_cast<Conv2d<T>*>(l);
                if (lc->quantize) 
                {
                    float scale;
                    /* float zero_point; */
                    fs.read((char*) &scale, sizeof(float));
                    /* fs.read((char*) &zero_point, sizeof(float)); */
                    lc->scale = scale;
                    /* lc->zero_point = zero_point; */
                }
            }
        

}
}


template<typename T>
template<typename S>
void SimpleNN<T>::write_or_read_params(S& fs, string mode)
{
    for (Layer<T>* l : net) {
        vector<float> tempMatrix1, tempMatrix2, tempMatrix3, tempMatrix4; // Temporary vectors for parameter storage

        if (l->type == LayerType::LINEAR) {
            Linear<T>* lc = dynamic_cast<Linear<T>*>(l);
            int s1 = lc->W.rows() * lc->W.cols();
            int s2 = lc->b.size();
            tempMatrix1.resize(s1);
            tempMatrix2.resize(s2);
    
            if (mode == "write") {
                for (int i = 0; i < s1; i++) 
                {
                    tempMatrix1[i] = lc->W(i / lc->W.cols(), i % lc->W.cols()).reveal();
                }
                for (int i = 0; i < s2; i++)
                {
                    tempMatrix2[i] = lc->b[i].reveal();
                }
                fs.write((char*)tempMatrix1.data(), sizeof(T) * s1);
                fs.write((char*)tempMatrix2.data(), sizeof(T) * s2);

                /* fs.write((char*) &(lc->zero_point.reveal()), sizeof(T)); */
            }
            else {
                fs.read((char*)tempMatrix1.data(), sizeof(T) * s1);
                fs.read((char*)tempMatrix2.data(), sizeof(T) * s2);
                int32_t zero_point;
                fs.read((char*) &zero_point, sizeof(int32_t));
                for (int i = 0; i < s1; i++) 
                {
                    lc->W(i / lc->W.cols(), i % lc->W.cols()) = T(tempMatrix1[i]);
                }
                for (int i = 0; i < s2; i++)
                {
                    lc->b[i] = T(tempMatrix2[i]);
                }
                lc->zero_point = T(zero_point);
            }
        }
        else if (l->type == LayerType::CONV2D) {
            Conv2d<T>* lc = dynamic_cast<Conv2d<T>*>(l);
            int s1 = lc->kernel.rows() * lc->kernel.cols();
            int s2 = lc->bias.size();
            tempMatrix1.resize(s1);
            tempMatrix2.resize(s2);

            if (mode == "write") {
                for (int i = 0; i < s1; i++) 
                {
                    tempMatrix1[i] = lc->kernel(i / lc->kernel.cols(), i % lc->kernel.cols()).reveal();
                }
                for (int i = 0; i < s2; i++) 
                {
                    tempMatrix2[i] = lc->bias[i].reveal();
                }
                fs.write((char*)tempMatrix1.data(), sizeof(T) * s1);
                fs.write((char*)tempMatrix2.data(), sizeof(T) * s2);
                /* fs.write((char*) &(lc->zero_point.reveal()), sizeof(T)); */
            }
            else {
                fs.read((char*)tempMatrix1.data(), sizeof(T) * s1);
                fs.read((char*)tempMatrix2.data(), sizeof(T) * s2);
                uint32_t zero_point;
                fs.read((char*) &zero_point, sizeof(uint32_t));
                for (int i = 0; i < s1; i++)
                {
                    lc->kernel(i / lc->kernel.cols(), i % lc->kernel.cols()) = T(tempMatrix1[i]);
                } 
                for (int i = 0; i < s2; i++)
                {
                    lc->bias[i] = T(tempMatrix2[i]);
                }
                lc->zero_point = T(zero_point);
            }
        }
        else if (l->type == LayerType::BATCHNORM1D) {
            BatchNorm1d<T>* lc = dynamic_cast<BatchNorm1d<T>*>(l);
            int s1 = (int)lc->move_mu.size();
            int s2 = (int)lc->move_var.size();
            int s3 = (int)lc->gamma.size();
            int s4 = (int)lc->beta.size();
            tempMatrix1.resize(s1);
            tempMatrix2.resize(s2);
            tempMatrix3.resize(s3);
            tempMatrix4.resize(s4);
            if (mode == "write") {
                for (int i = 0; i < s1; i++) 
                {
                    tempMatrix1[i] = lc->move_mu[i].reveal();
                }
                for (int i = 0; i < s2; i++) 
                {
                    tempMatrix2[i] = lc->move_var[i].reveal();
                }
                for (int i = 0; i < s3; i++) 
                {
                    tempMatrix3[i] = lc->gamma[i].reveal();
                }
                for (int i = 0; i < s4; i++) 
                {
                    tempMatrix4[i] = lc->beta[i].reveal();
                }
                fs.write((char*)tempMatrix1.data(), sizeof(T) * s1);
                fs.write((char*)tempMatrix2.data(), sizeof(T) * s2);
                fs.write((char*)tempMatrix3.data(), sizeof(T) * s3);
                fs.write((char*)tempMatrix4.data(), sizeof(T) * s4);
            }
            else {
                fs.read((char*)tempMatrix1.data(), sizeof(T) * s1);
                fs.read((char*)tempMatrix2.data(), sizeof(T) * s2);
                fs.read((char*)tempMatrix3.data(), sizeof(T) * s3);
                fs.read((char*)tempMatrix4.data(), sizeof(T) * s4);
                for (int i = 0; i < s1; i++)
                {
                    lc->move_mu[i] = T(tempMatrix1[i]);
                } 
                for (int i = 0; i < s2; i++)
                {
                    float var = 1 / std::sqrt(tempMatrix2[i] + 0.00001f);
                    lc->move_var[i] = T(var);
                }
                for (int i = 0; i < s3; i++)
                {
                    lc->gamma[i] = T(tempMatrix3[i]);
                }
                for (int i = 0; i < s4; i++)
                {
                    lc->beta[i] = T(tempMatrix4[i]);
                }
            }
        }
        else if (l->type == LayerType::BATCHNORM2D) {
            BatchNorm2d<T>* lc = dynamic_cast<BatchNorm2d<T>*>(l);
            int s1 = (int)lc->move_mu.size();
            int s2 = (int)lc->move_var.size();
            int s3 = (int)lc->gamma.size();
            int s4 = (int)lc->beta.size();
            tempMatrix1.resize(s1);
            tempMatrix2.resize(s2);
            tempMatrix3.resize(s3);
            tempMatrix4.resize(s4);
            if (mode == "write") {
                for (int i = 0; i < s1; i++) 
                {
                    tempMatrix1[i] = lc->move_mu[i].reveal();
                }
                for (int i = 0; i < s2; i++) 
                {
                    tempMatrix2[i] = lc->move_var[i].reveal();
                }
                for (int i = 0; i < s3; i++) 
                {
                    tempMatrix3[i] = lc->gamma[i].reveal();
                }
                for (int i = 0; i < s4; i++) 
                {
                    tempMatrix4[i] = lc->beta[i].reveal();
                }
                fs.write((char*)tempMatrix1.data(), sizeof(T) * s1);
                fs.write((char*)tempMatrix2.data(), sizeof(T) * s2);
                fs.write((char*)tempMatrix3.data(), sizeof(T) * s3);
                fs.write((char*)tempMatrix4.data(), sizeof(T) * s4);
            }
            else {
                fs.read((char*)tempMatrix1.data(), sizeof(T) * s1);
                fs.read((char*)tempMatrix2.data(), sizeof(T) * s2);
                fs.read((char*)tempMatrix3.data(), sizeof(T) * s3);
                fs.read((char*)tempMatrix4.data(), sizeof(T) * s4);
                for (int i = 0; i < s1; i++)
                {
                    lc->move_mu[i] = T(tempMatrix1[i]);
                } 
                for (int i = 0; i < s2; i++)
                {
                    float var = 1 / std::sqrt(tempMatrix2[i] + 0.00001f);
                    lc->move_var[i] = T(var);
                }
                for (int i = 0; i < s3; i++)
                {
                    lc->gamma[i] = T(tempMatrix3[i]);
                }
                for (int i = 0; i < s4; i++)
                {
                    lc->beta[i] = T(tempMatrix4[i]);
                }
            }
        }

    }
}



    template<typename T>
	void SimpleNN<T>::evaluate(const DataLoader<T>& data_loader)
	{
		int batch = data_loader.input_shape()[0];
		int n_batch = data_loader.size();
		/* T error_acc(0); */
        float error_acc(0);

		MatX<T> X;
		VecXi Y;
		VecXi classified(batch);

		system_clock::time_point start = system_clock::now();
		for (int n = 0; n < n_batch; n++) {
			MatX<T> X = data_loader.get_x(n);
			VecXi Y = data_loader.get_y(n);

			forward(X, false);
			classify(net.back()->output, classified);
			error_criterion(classified, Y, error_acc);
			
            std::cout << "[Batch: " << setw(3) << n + 1 << "/" << n_batch << "]";
			if (n + 1 < n_batch) {
                std::cout << "\r" << std::flush; 
			}
		}
		system_clock::time_point end = system_clock::now();
		duration<float> sec = end - start;

        /* cout << error_acc << ", " << batch << ", " << n_batch << "\n"; */
		cout << fixed << setprecision(2);
		cout << " - t: " << sec.count() << "s";
		cout << " - error(" << batch * n_batch << " images): ";
		/* cout << error_acc.reveal() / (batch * n_batch) * 100 << "%" << endl; */
		cout << error_acc / (batch * n_batch) * 100 << "%" << endl;
	}
}
