#!/bin/bash

# Nebula main directory
nebuladir=$PWD
# benchmark directory
benchdir=$PWD/benchmarks/$1

##### Translate google drive ID of Nebula benchmark #####
get_target_id() {
	echo -e "Get file ID of $1"
	case $1 in 
		lenet )
			FILEID="1i1FvFqeyjlTAqyh0Dji9FUo97uhbFXxR" ;;
		alexnet_S )
			echo -e "Does not support weight file of $1"
			exit 1;;
		alexnet_M )
			echo -e "Does not support weight file of $1"
			exit 1;;
		alexnet_L )
			FILEID="1CJyYVci0vgjZAf3kl_i1_VJaowFkZ6qV" ;;
		vgg_S )
            FILEID="1L6GzG0Je43jd6sVWICFC8oCVEtP507ee" ;;
		vgg_M )
            FILEID="19gUDTtHQInK12y0PUgyOPtxomkwlLJhl" ;;
		vgg_L )
            FILEID="1IfKa3pgt5W9kMtj1OVeuV1bNW4jfxhlD" ;;
		mlp_S )
			FILEID="1Rvx5gzy-ActdGjHWWiz5uJTfV5AD-oQ_" ;;
		mlp_M )
			FILEID="1Re0I0q0_ngt2NRcAd22AQthpxCtoeTi6" ;;
		mlp_L )
			FILEID="1l8POYfrLT2ZEIRbljt-7ub7-dXpEA1l2" ;;
		dbn_S )
			FILEID="1eT8bN0DPPtQNLumF92IEHQI31vTLV9gx" ;;
		dbn_M )
			FILEID="1dertTo4oNPxb8u4I3RVP9g1absIyAV78" ;;
		dbn_L )
            echo -e "Does not support weight file of $1" 
		    exit 1 ;;
		resnet_L )
			FILEID="1XYuSRsPm1HlDXQRLtvTx9sVmXbgXTCG2" ;;
		resnet_M )
            FILEID="1KzzvRJkYE4Qu5n7kjwBwbfKyAYYbROui" ;;
		resnet_S )
            FILEID="1DlERgUr2dOPZbPUP16EZ7y2EWjxR0Qy5" ;;
		rnn_L )
			echo -e "Does not support weight file of $1"
			exit 1 ;;
		rnn_M )
			echo -e "Does not support weight file of $1"
			exit 1 ;;
		rnn_S )
			echo -e "Does not support weight file of $1"
			exit 1 ;;
		lstm_L )
			echo -e "Does not support weight file of $1"
			exit 1 ;;
		lstm_M )
			echo -e "Does not support weight file of $1"
			exit 1 ;;
		lstm_S )
			echo -e "Does not support weight file of $1"
			exit 1 ;;
		* )
			echo -e "Error: unsupported weight"
			exit 1
	esac
} 

##### get weight usage #####
print_help() {
	echo -e "Usage: $0 <target name>"
}

# Print usage help
if [[ "$#" -lt 1 ]]; then
	print_help
	exit 1
fi

# Get google drive ID of nebula benchmark target.
get_target_id $1

# Download Nebula benchmark weight from google drive.
# store the weight to each benchmark directory with the name of <input.wgh>.
echo -e "$benchdir"
cd $benchdir

# Get weight from google drive.
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O input.wgh && rm -rf /tmp/cookies.txt
