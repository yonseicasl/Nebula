# Nebula
Nebula is a neural network framework.

# Pre-requisite
    * g++
    * nvcc
    * OpenCV
    * OpenBLAS for CPU (optional)
    * cuBLAS for GPU (optional)
    * cuDNN for GPU (optional)

# Install Nebula
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
