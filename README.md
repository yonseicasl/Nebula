# Nebula


## Introduction
The evolution of computing systems and explosive data production propel the advance of machine learning. As neural networks become increasingly important applications, developing neural network benchmarks has emerged as an imminent engineering challenge to tackle. Recent neural networks tend to form deeper networks to enhance accuracy and applicability, but such approaches impose great challenges on the modeling, simulation, and analysis of computing systems since they require unbearably long execution time to process a large amount of operations and sizable data. Neural networks are primarily composed of matrix and vector calculations that repeat numerous times on multi-dimensional data across channels, layers, batches, etc. This observation becomes a motive to develop lightweight neural network benchmark suite named Nebula.

The Nebula suite built on a C++ framework currently consists of seven representative neural networks including, ResNet, VGG, AlexNet, MLP, DBN, LSTM, and RNN. More models will be added to the pool in future releases, including MobileNet, YOLO, FCN, and GAN. Inspired by popular benchmark suites such as PARSEC and SPLASH-3 that allow users to choose a different input size per benchmark, Nebula offers multiple size options from large to small datasets for various types of neural networks. The large ones represent full-fledged neural networks that implement complete structures of neural networks and execute on massive datadasts, and the medium and small benchmarks are downsized representation of full-fledged networks. The benchmarks are implemented by formulating variable-sized datasets and compacting neural networks to support the datasets of different sizes.

Lightweight benchmarks aim at modeling the proxy behaviors of full-fledged neural networks to alleviate the modeling and simulation challenges imposed by hefty neural network workloads. Nebula benchmarks as “proxy apps” intend to reduce the computational costs of full-fledged networks but still capture end-to-end neural network behaviors. We hope the multi-size options of Nebula benchmarks broaden the usability and affordability in diverse experiment environments from real hardware to microarchitecture simulations in which users can selectively use appropriate size of benchmarks.


## Prerequisites
Nebula uses g++ and nvcc to compile C++ codes to execute on CPUs and NVIDIA GPUs, and it has dependency on OpenCV and OpenBLAS libraries to accelerate neural network algorithms. Nebula benchmarks have been validated in 18.04 (Bionic Beaver) with any later versions of g++-5.4, nvcc-9.0, OpenCV-3.2, and OpenBLAS-0.2 (as of July 2020).

    * g++-5.4 or later
    * nvcc-9.0 or later
    * OpenCV-3.2 or later
    * OpenBLAS-0.2 or later: optional for CPU acceleration
    * cuBLAS and cuDNN (of package nvidia-384 or later): optional for GPU acceleration

For instance in Ubuntu 18.04, use the following command to install the required libraries except for the NVIDIA driver package.

    $ sudo apt-get install build-essential g++ nvcc libopenblas-dev libopencv-dev

To install an NVIDIA driver packages for cuBLAS and cuDNN libraries, refer to the following link: https://developer.nvidia.com/cuda-toolkit-archive. For example, installing an nvidia-384 package can be done by typing the following run command with sudo privilege.

    $ sudo ./cuda_9.0.176_384.81_linux-run


## Download
The latest release of Nebula benchmark suite is v1.0 (as of July, 2020). To obatin a copy of Nebula v1.0, use the following git command in a terminal.

    $ git clone --branch v1.0 https://github.com/yonsei-icsl/nebula

Or, if you wish to use the latest development version, simply clone the git respository as follows.

    $ git clone https://github.com/yonsei-icsl/nebula


## Build
Nebula provides a script file named nebula.sh to facilitate the build and run processes of benchmark suite. To build the entire models of Nebula suite, execute the script file as follows in the main Nebula directory.

    $ cd nebula/
    $ ./nebula.sh build all

Alternatively, you may specify a benchmark of a particular size to build it by typing a command in the following format.

    $ ./nebula.sh build <benchmark> <size>

For instance, a small benchmark of AlexNet can be built as follows. Possible options for the <benchmark> field is listed below the example.

    $ ./nebula.sh build alexnet small


  

If you want to build a specific Nebula benchmark, using following commands.
The benchmarks supported at Nebula v1.0 is listed in following table.
We use Small size of VGG as an exemplary network.

    $ ./nebula.sh build lib
    $ ./nebula.sh build <benchmarks>
    (example) $ ./nebula.sh build vgg_S

Network type | Size  | command | Network type | Size  | command
---          | ---   | ---     | ---          | ---   | ---
AlexNet      | Large <br> Medium <br> Small | alexnet_L <br> alexnet_M <br> alexnet_S | VGG          | Large <br> Medium <br> Small | vgg_L <br> vgg_M <br> vgg_S
ResNet       | Large <br> Medium <br> Small | resnet_L <br> resnet_M <br> resnet_S | MLP          | Large <br> Medium <br> Small | mlp_L <br> mlp_M <br> mlp_S
DBN          | Large <br> Medium <br> Small | dbn_L <br> dbn_M <br> dbn_S | RNN          | Large <br> Medium <br> Small | rnn_L <br> rnn_M <br> rnn_S
LSTM         | Large <br> Medium <br> Small | lstm_L <br> lstm_M <br> lstm_S

# Running Nebula benchmarks
Running Nebula requires the neural network with network configuration file (i.e., <em>network.cfg</em>), dataset. Optionally, you can obtain weight file.

## Preparing dataset
You can download the dataset using the shell script <em>get_data.sh</em>. Nebula includes ImageNet for convolutional networks, NIST for fully connected networks, and PTB for recurrent networks. And each dataset supports variable-sized options from large to small. Following instructs the command to download the small size of ImageNet.

    $ ./get_data.sh
    Which dataset? [ImageNet[I] / NIST[N] / MNIST[M]] I
    Which size? [Large[L] / Medium[M] / Small[S]] S
    Get data ID of ImageNet_S

    ...

    Saving to: ImageNet_S.tar
    ImageNet_S.tar      [           <=>         ] 2.82G  5.15MB/s

After typing the command, data/ folder is created in the nebula directory. Now you should list file to the benchmark directory using following commands.

    $ cd nebula/
    $ cp data/ImageNet_S/.lst benchmarks/vgg_S

<!--
## Preparing Weight (Optional)
You can download weight file using the shell script <em>get_weight.sh</em>. The weight file is downloaded at each benchmark directory with the name <em>input.wgh</em>. Following shows the commands to get weight and example. Convolutional and fully connected networks' weight files are available now. The weight files of recurrent networks will be updated soon.

    $ ./get_weight.sh <benchmarks>
    (example) $ ./get_weight.sh vgg_S
-->

## Training
With input files (i.e., config file and dataset), you can run variable-sized neural networks. After training the networks, an instantaneous loss, and a cumulative loss, and runtime are printed out at the end of every iteration. And after training, <em>output.wgh</em> file is created which is a trained weight file. Following describes the command for training Nebula benchmarks.

    $ cd nebula/
    $ ./nebula.sh train <benchmarks>
    (example) $ ./nebula.sh train vgg_S


## Inference
With trained weight file, you can validate the efficacy of neural networks. At the end of inference, instantaneous accuracy, cumulative accuracy, and execution time are printed out.

    $ cd nebula/
    $ ./nebula.sh test <benchmarks>
    (example) $ ./nebula.sh test vgg_S

# Contact
In case you notice a bug or you have a question regarding the use of Nebula benchmarks, please feel free to contact me via email, bogilkim@yonsei.ac.kr.
