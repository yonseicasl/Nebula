############################################
# Developed by                             #
# Intelligent Computing Systems Lab (ICSL) #
# Yonsei University, Seoul, Korea          #
############################################

#!/bin/bash

##### Nebula information #####
# Nebula main directory
nebuladir=$PWD
# Nebula common directory
commondir=$nebuladir/common
# Nebula network directory
networkdir=$nebuladir/models/networks
# Nebula network models directory
networkdir=$nebuladir/models/networks
# Nebula layer models directory
layerdir=$nebuladir/models/layers
# Nebula library directory
libdir=$nebuladir/library
# Nebula library
lib=$libdir/libnebula.a


# Nebula benchmark directory
benchdir=$nebuladir/benchmarks
# Nebula version
ver=1.0.0


##### Nebula build options #####
# C++ compiler
cc=g++
# CUDA compiler
cu=nvcc
# Linker
lc=$cc
# Makefile options
mopt="CC=$cc CU=$cu DIR=$nebuladir LIB=$lib"
# Compiler options
ccopt="-Wall -fPIC -I$commondir -I$networkdir -I$layerdir -DUSE_BLAS"
# Linker options
ldopt="-L$libdir"
libopt="-lnebula -lopenblas -lpthread `pkg-config --libs opencv`"
# C++ version
stdc="-std=c++11"
# CUDA support
ptx="compute_61"
cubin="sm_61"

##### Nebula banner #####
print_banner() {
    echo -e " _____________________________________________________________ "
	echo -e "|                                                             |"
	echo -e "|    * *       #   #  #####  ###     #   #  #      ##     *   |"
	echo -e "|        *    ##  #  #      #   #   #   #  #      # #   *   * |"
	echo -e "|     *      # # #  #####  #####   #   #  #      ####     *   |"
	echo -e "|   *       #  ##  #      #    #  #   #  #      #   #      *  |"
	echo -e "|      *   #   #  #####  ######  #####  #####  #    #    *    |"
	echo -e "|                                                             |"
	echo -e "|                                                             |"
	echo -e "|                                                             |"
    echo -e "|  Nebula: A Neural network framework                         |"
    echo -e "|  Intelligent Computing System Lab(ICSL)                     |"
    echo -e "|  School of Electrical Engineering, Yonsei University        |"
    echo -e "|  Version: 0.1                                               |"
    echo -e "|_____________________________________________________________|"
}

##### Nebula usage #####
print_help() {
    echo -e "Usage: $0 [build/clean/test/train] [target]"
    echo -e "Options: -load_weight  : resume training from input.wgh"
    exit 0
}


# Print usage help.
if [[ "$#" -lt 2 || $1 = '-h' || $1 = '--help' ]]; then
	print_help
fi

##### Default arguments #####
# Build, clean, test, or training
action=$1; shift
if [[ $action != build && $action != clean && $action != test && $action != train ]]; then
	echo -e "Error: unknown action $action"
	print_help
fi

# Read lowercase of network type.
network=${1,,}; shift
target=$network

# Read lowercase of size option.
size=${1,,}; shift

# Concat the size option to the benchmark.
if [[ $target != 'lib' && $target != 'all' ]]; then
    if [[ $size = 'small' ]]; then
        target+=_small
    elif [[ $size = 'medium' ]]; then
        target+=_medium
    elif [[ $size = 'large' ]]; then
        target+=_large
    else
        echo -e "Error: Nebula benchmark does not support the size option $size"
        echo -e "Usage: $0 $action $network <size>"
        echo -e "<size> options: large, medium, small"
        exit 1
    fi
fi


##### Optional arguments #####
# Use gdb for debug.
use_debug=0
# Use GPU.
gpu_enabled=0
# Use custom blas.
custom_blas=0
# Do not load weight for training by default.
load_weight=0

# Parse optional arguments for load weight when training.
while [[ "$1" != '' ]]; do
	case $1 in
		-load_weight )
			load_weight=1
			;;
		*)
			echo -e "Error: unknown option $1"
			exit 1
	esac
	shift
done

# Append Makefile options when debug.
if [[ $debug_enabled -eq 1 ]]; then
	ccopt+=" -g"
fi

# Append Makefile options when GPU is enabled.
if [[ $gpu_enabled -eq 1 ]]; then
	lc=$cu
	ccopt+=" -DGPU_ENABLED"
    cuarch="CUARCH=\"-gencode arch=$ptx,code=[$cubin,$ptx]\""
    libopt+=" -lcuda -lcudart -lcurand -lcublas"
    mopt+=" CUSRC=\"$cusrc\" CUOBJ=\"$cuobj\" GPU_ENABLED=\"$gpu_enabled\""
fi

# Append Makefile options when select custom blas.
if [[ $custom_blas -eq 1 ]]; then
	ccopt+=" -DCUSTOM_BLAS"
fi

# Makefile MFLAG
mflag="$mopt LC=$lc"
# Makefile CCFLAG
ccflag="CCFLAG=\"$ccopt\""
# Makefile LDFLAG
ldflag="LDFLAG=\"$ldopt\""
# Makefile LIBFLAG
libflag="LIBFLAG=\"$libopt\""
# Makefile STD
std="STD=\"$stdc\""

# Clean the Nebula benchmark
if [[ $action = 'clean' ]]; then
	# Clean everything.
	if [[ $target = 'all' ]]; then
		# Clean Nebula library.
		echo -e "\n# Cleaning Nebula library"
		cd $libdir; eval $mflag make clean
		# Clean all Nebula benchmarks.
		for t in $benchdir/*; do
			target=$(basename $t)
			echo -e "\n# Cleaning Nebula benchmark $target"
			cd $t; eval EXE=$target make clean
		done
	# Clean Nebula library.
	elif [[ $target = 'lib' ]]; then 
		echo -e "\n# Cleaning Nebula library"
		cd $libdir; eval $mflag make clean
	# Clean the specified Nebula benchmark.
	else 
		# Executable file of specified Nebula benchmark does not exist.
		if [[ ! -d $benchdir/$target ]]; then
			echo -e "Error: Nebula benchmark $target does not exist"
			exit 1
		fi
		echo -e "\n# Cleaning Nebula benchmark $target"
		cd $benchdir/$target; eval EXE=$target make clean
	fi

# Build the Nebula benchmark
elif [[ $action = 'build' ]]; then
	# Build everything.
	if [[ $target = 'all' ]]; then
		# Build Nebula library.
		echo -e "\n# Building Nebula library v$ver"
		cd $libdir; eval $mflag $ccflag $cuarch $std make -j$(nproc)
		# Build Nebula benchmarks.
		for t in $benchdir/*; do
			target=$(basename $t)
			echo -e "\n# Building Nebula benchmark $target"
			cd $t; eval $mflag $ccflag $cuarch $ldflag $libflag $std EXE=$target make -j$(nproc)
		done
	# Build Nebula library.
	elif [[ $target = 'lib' ]]; then
		echo -e "\n# Building Nebula library v$ver"
		cd $libdir; eval $mflag $ccflag $cuarch $std make -j$(nproc) 
	# Build the specific Nebula benchmark.
	else
		if [[ ! -d $benchdir/$target ]]; then 
			echo -e "Error: Nebula benchmark $target does not exist"
			exit 1
        # Build the particular benchmark.
		else 
			#check if Nebula library was built.
			if [[ ! -f $lib ]]; then
				read -p "Build Nebula library first? [Y/N] " ans
				if [[ ${ans,,} = 'y' ]]; then
					echo -e "Build Nebula library"
					cd $libdir; eval $mflag $ccflag $cuarch $std make -j$(nproc)
				else 
					echo -e "Error: $lib does not exist"
					exit 1
				fi
			fi
			# Build Nebula benchmark target.
			echo -e "\n# Building Nebula benchmark $target"
			cd $benchdir/$target; eval $mflag $ccflag $cuarch $ldflag $libflag $std EXE=$target make -j$(nproc)
		fi
	fi
# Training the Nebula benchmark
elif [[ $action = 'train' ]]; then
	print_banner
	# Executable file does not exist.
	if [[ ! -f $benchdir/$target/$target ]]; then 
        read -p "Build benchmark first? [Y/N] " ans
        if [[ ${ans,,} = 'y' ]]; then
            if [[ ! -f $lib ]]; then 
                read -p "Build Nebula library first? [Y/n] " ans
                if [[ ${ans,,} = 'y' ]]; then
					echo -e "Build Nebula library"
					cd $libdir; eval $mflag $ccflag $cuarch $std make -j$(nproc)
                else 
                    echo -e "Error: $lib does not exist"
                    exit 1
                fi
            fi
            echo -e "\n# Building Nebula benchmark $target"
			cd $benchdir/$target; eval $mflag $ccflag $cuarch $ldflag $libflag $std EXE=$target make -j$(nproc)
        else 
            echo -e "Error: Executable file $target does not exist"
            exit 1
        fi
	fi
    # Training the Nebula benchmark.
	echo -e "\n# Training Nebula benchmark $target ..."
	cd $benchdir/$target
	# Training the nebula benchmark target.
	if [[ $load_weight -eq 0 ]]; then
		eval ./$target train network.cfg data.cfg \"\" input.wgh
	else 
		eval ./$target train network.cfg data.cfg input.wgh input.wgh
	fi

# Inference the Nebula benchmark
elif  [[ $action = 'test' ]]; then
	print_banner
	# Executable file does not exist.
	if [[ ! -f $benchdir/$target/$target ]]; then 
        read -p "Build benchmark first? [Y/N] " ans
        if [[ ${ans,,} = 'y' ]]; then
            if [[ ! -f $lib ]]; then 
                read -p "Build Nebula library first? [Y/n] " ans
                if [[ ${ans,,} = 'y' ]]; then
					echo -e "Build Nebula library"
					cd $libdir; eval $mflag $ccflag $cuarch $std make -j$(nproc)
                else 
                    echo -e "Error: $lib does not exist"
                    exit 1
                fi
            fi
            echo -e "\n# Building Nebula benchmark $target"
			cd $benchdir/$target; eval $mflag $ccflag $cuarch $ldflag $libflag $std EXE=$target make -j$(nproc)
        else 
            echo -e "Error: Executable file $target does not exist"
            exit 1
        fi
	fi
	cd $benchdir/$target
	# Download weight from google drive if the benchmark doesn't have input weight.
	if [[ ! -f $benchdir/$target/input.wgh ]]; then
		read -p "Want to download weight? [Y/N] " ans
		if [[ $ans = 'y' || $ans = 'Y' ]]; then
			cd $nebuladir
			./weight.sh $network $size
			cd $benchdir/$target
			echo -e "Downloading the weight is done."
		else 
			echo -e "Input weight does not exist."
			exit 1
		fi
	fi
	# Inference the Nebula benchmark target.
	./$target test network.cfg data.cfg input.wgh
fi
