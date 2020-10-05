# Nebula: Lightweight Neural Network Benchmarks
Developed by Bogil Kim, Sungjae Lee, Chanho Park, Hyeonjin Kim, and William J. Song\
Intelligent Computing Systems Lab, Yonsei University\
Current release: v1.2 (Aug., 2020)

## Introduction
The evolution of computing systems and explosive data production propel the advance of machine learning. As neural networks become increasingly important applications, developing neural network benchmarks has emerged as an imminent engineering challenge to tackle. Recent neural networks tend to form deeper networks to enhance accuracy and applicability, but such approaches impose great challenges on the modeling, simulation, and analysis of computing systems since they require unbearably long execution time to process a large amount of operations and sizable data. Neural networks are mostly comprised of matrix and vector calculations that repeat numerous times on multi-dimensional data across channels, layers, batches, etc. This observation becomes a motive to develop a lightweight neural network benchmark suite named Nebula.

The Nebula suite is built on a C++ framework and currently consists of seven representative neural networks including, ResNet, VGG, AlexNet, MLP, DBN, LSTM, and RNN. We plan to add more neural network models to the pool in future releases, including MobileNet, YOLO, FCN, and GAN. Inspired by popular benchmark suites such as PARSEC and SPLASH-3 that provide users with options to choose a different input size per benchmark, Nebula offers multiple size options from large to small datasets for various types of neural networks. The large ones represent full-fledged neural networks that implement complete structures of the neural networks and execute on massive datasets, and the medium and small benchmarks are downsized representation of full-fledged networks. The benchmarks are implemented by formulating variable-sized datasets and compacting neural networks to support the datasets of different sizes.

Lightweight benchmarks aim at modeling the proxy behaviors of full-fledged neural networks to alleviate the challenges imposed by hefty neural network workloads. Nebula benchmarks as “proxy apps” intend to reduce the computational costs of full-fledged networks but still capture end-to-end neural network behaviors. We hope the multi-size options of Nebula benchmarks broaden the usability and affordability of them in diverse experiment environments from real hardware to microarchitecture simulations in which users can selectively use appropriate size of benchmarks.


## Prerequisites
Nebula uses g++ and nvcc to compile C++ codes to execute on CPUs and NVIDIA GPUs. It has dependency on OpenCV and OpenBLAS libraries to accelerate neural network algorithms. The Nebula benchmarks have been validated in 18.04 (Bionic Beaver) with g++-5.4, nvcc-9.0, OpenCV-3.2, and OpenBLAS-0.2 or any later versions of them (as of July, 2020).

    * g++-5.4 or later
    * nvcc-9.0 or later
    * OpenCV-3.2 or later
    * OpenBLAS-0.2 or later: optional for CPU acceleration
    * cuBLAS (of package nvidia-384 or later) and cuDNN (version 7.0.4 to 7.6.5): optional for GPU acceleration

In Ubuntu 18.04, use the following command to install the required libraries except for the NVIDIA driver package.

    $ sudo apt-get install build-essential g++ nvcc libopenblas-dev libopencv-dev

To install an NVIDIA driver package for cuBLAS and cuDNN libraries, refer to the following link: https://developer.nvidia.com/cuda-toolkit-archive. For example, installing an nvidia-384 package can be done by executing the following run command with sudo privilege.

    $ sudo ./cuda_9.0.176_384.81_linux-run

To install cuDNN libraries, refer to the following link:
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html. For example, installing cudnn-v7.6.5 can be done by executing the following commands with sudo privilege.

    $ tar xf cudnn-10.2-linux-x64-v7.6.5.32.tgz
    $ sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
    $ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    $ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

## Download
The latest release of Nebula benchmark suite is v1.2 (as of Aug., 2020). To obatin a copy of Nebula v1.1, use the following git command in a terminal.

    $ git clone --branch v1.2 https://github.com/yonsei-icsl/nebula

Or, if you wish to use the latest development version, simply clone the git respository as is.

    $ git clone https://github.com/yonsei-icsl/nebula

## Build
Nebula provides a script file named nebula.sh to facilitate the build and run processes of benchmark suite. To build the entire models of Nebula suite, execute the script file as follows in the main directory of Nebula.

    $ cd nebula/
    $ ./nebula.sh build all

Alternatively, you may specify a benchmark of a particular size to build it by typing a command in the following format.

    $ ./nebula.sh build <network> <size>

For instance, a small benchmark of AlexNet can be built as follows. Possible options for the <network> and <size> fields are listed after the example.

    $ ./nebula.sh build alexnet small

Nebula v1.0 includes seven different types of neural networks, and each network has three different size options, i) large (L), medium (M), and small (S). The large benchmark of a given network type represents the full-fledged neural network, and the medium and small benchmarks are down-sized representations. Small benchmarks on average about 10-15x faster to run than full-fldged counterparts, while exhibiting similar hardware performance and characteristics. Medium benchmarks in general reduce the runtime by 3-5x with more similar emulation of full-fledged networks. The benchmarks have been rigorously validated across a variety of platforms including CPUs, GPUs, FPGAs, and NPUs. The following lists possible <network> and <size> options to put in the script run command shown above.

    Build command: ./nebula.sh build <network> <size>

    <network> options: alexnet, dbn, lstm, mlp, resnet, rnn, vgg
    <size> options: large, medium, small


## Configs, Dataset, and Pre-trained Weights
Executing a Nebula benchmark requires three inputs, i) network configs, ii) input dataset, and optionally iii) pre-trained weights. The network configs specify detailed structure of the neural network, and the config files should be found in the nebula/benchmarks/ directory after cloning from the github repository. Each sub-directory inside the nebula/benchmarks/ is named after a network type and size such as alexnet_small for small-sized benchmark of AlexNet.

A dataset is a group of input data consumed by the neural network. Nebula uses ImageNet for convolutional networks (i.e., AlexNet, ResNet, VGG), NIST for fully-connected networks (i.e., DBN, MLP), and PTB for recurrent networks (i.e., LSTM, RNN). These well-known datasets have been reformulated to fit for variable-sized Nebula benchmarks. Due to a limited space in the github repository to accommodate sizable datasets, Nebula maintains them in a remote Google Drive. To obtain a copy of a particular dataset, execute a script named dataset.sh in the following format.

    $ ./dataset.sh <dataset> <size>

For instance, a small-sized ImageNet can be obtained using the script as follows. Executing the script creates a directory named nebula/dataset/ (if it does not exist), and it places downloaded dataset files in the directory. Possible options for the <dataset> and <size> fields are listed after the example.

    $ ./dataset.sh imagenet small

The following shows possible <dataset> and <size> options to put in the script run command shown above.

    Dataset command: ./dataset.sh <dataset> <size>

    <dataset> options: imagenet, nist, ptb
    <size> options: large, medium, small

Nebula provides a set of pre-trained weights for user convenience. The weights can be used for inference of neural network benchmarks or optionally as initial states of training. Similar to the process of obtaining a dataset, weights can be downloaded using the weight.sh script file as follows.

    $ ./weight.sh <network> <size>

The following lists possible <network> and <size> options to put in the script run command shown above.

    Weight command: weight.sh <network> <size>

    <network> options: alexnet, dbn, lstm, mlp, resnet, rnn, vgg
    <size> options: large, medium, small


## Run
After a Nebula benchmark is built and dataset and weight files are downloaded, the benchmark becomes ready to execute either for inference or training. The nebula.sh script facilitates the execution of Nebula benchmark. A run command to execute the benchmark follows the format shown below. The <run type> field in the command specifies an execution type, i.e., test or train. The <benchmark> field indicates a target benchmark to run, such as vgg small for small-sized vgg.

    $ ./nebula.sh <run type> <benchmark> <small>

For example, the following command runs the inference of small-sized benchmark of VGG.

    $ ./nebula.sh test vgg small

Similarly, training a medium-sized ResNet can be done using the following command.

    $ ./nebula.sh train resnet medium


## Reference and Contact
To reference our work, please use our TC paper.

    @inproceedings{kim_modsim2020,
      author    = {B. Kim and S. Lee and C. Park and H. Kim and W. Song},
      title     = {{The Nebula Benchmark Suite: Implications of Lightweight Neural Networks}},
      booktitle = {IEEE Transactions on Computers},
      month     = {Oct.},
      year      = {2020},
    }

For troubleshooting, bug reports, or any questions regarding the use of Nebula benchmark suite, please contact Bogil Kim via email: bogilkim {\at} yonsei {\dot} ac {\dot} kr. Or, visit our lab webpage: https://icsl.yonsei.ac.kr
