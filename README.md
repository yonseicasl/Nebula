# Nebula
Nebula is a lightweight benchmark suite for neural networks. Recent neural networks become increasingly deeper and larger. This trend puts great challenges on the modeling and analysis of computer systems in that it takes longer execution time to process a large number of operations and sizable data. Nebula tackles this challenge by taking the opposite direction based on an observation that the computations of neural networks are mainly comprised of matrix and vector operations. Nebula benchmarks drastically reduce the number of operations for an expectation that the lightened networks will still have similar architectural behaviors as the full-fledged neural networks, and thus it is not always necessary to run the complete networks with sizable data if the purpose is to characterize micro-architectural behaviors in computer systems. Results based on hardware measurements prove that Nebula benchmarks indeed meet the similarity and affordability goal.

# Pre-requisite
Nebula requires g++ compiler for CPU, nvcc for GPU, and OpenCV for image processing.
it supported libraries encompass OpenBLAS for CPU, and CUDA libraries encompassing cuBLAS and cuDNN, but the implementations are not limited to those external libraries.

    * g++
    * nvcc
    * OpenCV
    * OpenBLAS for CPU (optional)
    * cuBLAS for GPU (optional)
    * cuDNN for GPU (optional)

# Install Nebula
The latest version of Nebula is v1.0 (as of Jan., 2020). It has been validated in Ubuntu 16.04 and 18.04. To obtain a Nebula v1.0, use the following command in a terminal.
    #git clone https://github.com/yonsei-icsl/nebula.git

# Build Nebula
    cd <working_dir>/nebula
    ./nebula.sh build [target]

# Preparing the Nebula dataset.
	cd <working_dir>/nebula
	./get_data.sh
    cd <working_dir>/nebula/data/<database>
    cp *.lst <working_dir>/nebula/benchmark/[network]

# Training
    cd <working_dir>/nebula
    ./nebula.sh train [network] -load_weight(optional)
    or
    cd benchmarks/[network] && ./[network] train network.cfg data.cfg input.wgh(optional) output.wgh(optional)

# Inference
    cd <working_dir>/nebula
    ./nebula.sh test [network] -load_weight(required)
    or
    cd benchmarks/[network] && ./[network] test network.cfg data.cfg input.wgh
