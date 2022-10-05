## CSTP
Pytorch implementation for "Cascaded Compressed Sensing Networks".

## Dependencies
NVIDIA GPU + CUDA (may CuDNN) and corresponding PyTorch framework (version 1.8.0)
Python 3.7

## Datasets
We use CIFAR10 and MNIST dataset in our experiments. For CIFAR10 dataset, it has been given as a .npz type file [Here](./data/). For MNIST dataset, you can download from
[Here](http://yann.lecun.com/exdb/mnist/).
## Training
To train CSTP on CIFAR10 dataset with Default Parameters, you can just run [train.py](./train.py). The Network size and other Hyperparameter can be modified [Here](./train.py#L128).


The training codes, the training data, and some pre-trained models are given. Please feel free to contact me if you have any questions. My email is wzlu@sdu.edu.cn
