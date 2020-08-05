# Nebula
Nebula is a lightweight benchmark suite for neural networks. Recent neural networks become increasingly deeper and larger. This trend puts great challenges on the modeling and analysis of computer systems in that it takes longer execution time to process a large number of operations and sizable data. Nebula tackles this challenge by taking the opposite direction based on an observation that the computations of neural networks are mainly comprised of matrix and vector operations. Nebula benchmarks drastically reduce the number of operations for an expectation that the lightened networks will still have similar architectural behaviors as the full-fledged neural networks, and thus it is not always necessary to run the complete networks with sizable data if the purpose is to characterize micro-architectural behaviors in computer systems. Results based on hardware measurements prove that Nebula benchmarks indeed meet the similarity and affordability goal.

# Requirements
Nebula requires g++ compiler to build on CPU, nvcc on GPU, and OpenCV for image processing.
It utilizes several acceleration libraries encompassing OpenBLAS for CPU and CUDA libraries to speed up the computations of neural networks, but the implementations are not limited to external libraries.

    * g++
    * nvcc
    * OpenCV
    * OpenBLAS for CPU (optional)
    * cuBLAS for GPU (optional)
    * cuDNN for GPU (optional)

## For Ubuntu (>=18.04)
    $ sudo apt-get install build-essential
    $ sudo apt-get install g++ g++-multilib
    $ sudo apt-get install libopenblas-dev
    $ sudo apt-get install libopencv-dev

# Download and Install Nebula
The latest version of Nebula is v1.0 (as of July, 2020). It has been validated in Ubuntu 16.04 and 18.04. To obtain a Nebula v1.0, use the following git command in terminal.
Then, enter the nebula directory and build the Nebula benchmarks using ./nebula.sh command.

    $ git clone --branch v1.0 https://github.com/yonsei-icsl/nebula
    $ cd nebula/
    $ ./nebula.sh build all

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

## Preparing Weight (Optional)
You can download weight file using the shell script <em>get_weight.sh</em>. The weight file is downloaded at each benchmark directory with the name <em>input.wgh</em>. Following shows the commands to get weight and example. Convolutional and fully connected networks' weight files are available now. The weight files of recurrent networks will be updated soon.

    $ ./get_weight.sh <benchmarks>
    (example) $ ./get_weight.sh vgg_S

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
