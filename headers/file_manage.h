#pragma once
#include "common.h"

namespace simple_nn
{
	int ReverseInt(int i)
	{
		unsigned char ch1, ch2, ch3, ch4;
		ch1 = i & 255;
		ch2 = (i >> 8) & 255;
		ch3 = (i >> 16) & 255;
		ch4 = (i >> 24) & 255;
		return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
	}

	MatXf read_mnist(string data_dir, string fname, int n_imgs, bool train = true)
	{
		MatXf img;

		string path = data_dir + "/" + fname;
		ifstream fin(path, ios::binary);
		if (fin.is_open()) {
			int magic_number = 0;
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;

			fin.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);
			fin.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);
			fin.read((char*)&n_rows, sizeof(n_rows));
			n_rows = ReverseInt(n_rows);
			fin.read((char*)&n_cols, sizeof(n_cols));
			n_cols = ReverseInt(n_cols);

			img.resize(n_imgs, n_rows * n_cols);

			float m = 0.1306604762738431f;
			float s = 0.3081078038564622f;

			if (!train) {
				m = 0.13251460584233699f;
				s = 0.3104802479305348f;
			}

			for (int n = 0; n < n_imgs; n++) {
				for (int i = 0; i < n_rows; i++) {
					for (int j = 0; j < n_cols; j++) {
						unsigned char temp = 0;
						fin.read((char*)&temp, sizeof(temp));
						img(n, j + n_cols * i) = (temp / 255.f - m) / s;
					}
				}
			}
		}
		else {
			cout << "The file(" << path << ") does not exist." << endl;
			exit(1);
		}

		return img;
	}

	VecXi read_mnist_label(string data_dir, string fname, int n_imgs)
	{
		VecXi label(n_imgs);

		string path = data_dir + "/" + fname;
		ifstream fin(path);
		if (fin.is_open()) {
			for (int i = 0; i < n_imgs + 8; ++i) {
				unsigned char temp = 0;
				fin.read((char*)&temp, sizeof(temp));
				if (i > 7) {
					label[i - 8] = (int)temp;
				}
			}
		}
		else {
			cout << "The file(" << path << ") does not exist." << endl;
			exit(1);
		}

		return label;
	}
    // CIFAR-10 image dimensions and channels
    const int CIFAR10_IMG_HEIGHT = 32;
    const int CIFAR10_IMG_WIDTH = 32;
    const int CIFAR10_IMG_CHANNELS = 3; // RGB channels

    // Reads a single CIFAR-10 file and returns the images and labels
    std::pair<MatXf, VecXi> read_cifar10_file(const std::string& filename, int n_imgs) {
        std::ifstream file(filename, std::ios::binary);
        MatXf images(n_imgs, CIFAR10_IMG_HEIGHT * CIFAR10_IMG_WIDTH * CIFAR10_IMG_CHANNELS);
        VecXi labels(n_imgs);

        if (file.is_open()) {
            for (int i = 0; i < n_imgs; ++i) {
                unsigned char label;
                file.read((char*)&label, 1);
                labels[i] = label;

                for (int c = 0; c < CIFAR10_IMG_CHANNELS; ++c) {
                    for (int h = 0; h < CIFAR10_IMG_HEIGHT; ++h) {
                        for (int w = 0; w < CIFAR10_IMG_WIDTH; ++w) {
                            unsigned char pixel = 0;
                            file.read((char*)&pixel, sizeof(pixel));
                            int index = c * CIFAR10_IMG_HEIGHT * CIFAR10_IMG_WIDTH + h * CIFAR10_IMG_WIDTH + w;
                            images(i, index) = pixel / 255.0f;
                        }
                    }
                }
            }
        } else {
            std::cerr << "Failed to open file: " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        return std::make_pair(images, labels);
    }
}
