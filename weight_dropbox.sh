#!/bin/bash

print_help() {
    echo -e "Usage: $0 <network> <size>"
    echo -e ""
    echo -e "<network> options: alexnet, dbn, mlp, resnet, vgg"
    echo -e "<size> options: large, medium, small"
    exit 0
}

if [[ "$#" -lt 2 || $1 = '-h' || $1 = '--help' ]]; then
    print_help
fi

### Default arguments ###
# Read lowercase of network type.
network=${1,,}; shift

# Read lowercase of size options.
size=${1,,}; shift

### Concatenate size option to network type.
if [[ $size = 'large' ]]; then
    network+=_large
elif [[ $size = 'medium' ]]; then
    network+=_medium
elif [[ $size = 'small' ]]; then
    network+=_small
else
    echo -e "Error : Wrong size option $size"
    exit 1
fi

# Nebula main directory
nebuladir=$PWD

##### Translate google drive ID of Nebula benchmark #####
weight_ID() {
	case $1 in 
		lenet )
			FILEID="1i1FvFqeyjlTAqyh0Dji9FUo97uhbFXxR" ;;
		alexnet_large )
			FILEID="aq8rxjj5w3a4akkaqy8n7"
            RLKEY="3c4y0yuc6366c1e6enj55o1bq"
            ;;
		alexnet_medium )
            FILEID="1MYFsiV-LHt4sK-OumUlbfWooOcsFze7Y" ;;
		alexnet_small )
            FILEID="1fjJPJnJ914w8BsS7yDe-m9Cre3KZVOZi" ;;
		vgg_large )
            FILEID="1IfKa3pgt5W9kMtj1OVeuV1bNW4jfxhlD" ;;
		vgg_medium )
            FILEID="19gUDTtHQInK12y0PUgyOPtxomkwlLJhl" ;;
		vgg_small )
            FILEID="1L6GzG0Je43jd6sVWICFC8oCVEtP507ee" ;;
		mlp_large )
			FILEID="1l8POYfrLT2ZEIRbljt-7ub7-dXpEA1l2" ;;
		mlp_medium )
			FILEID="1Re0I0q0_ngt2NRcAd22AQthpxCtoeTi6" ;;
		mlp_small )
			FILEID="1Rvx5gzy-ActdGjHWWiz5uJTfV5AD-oQ_" ;;
		dbn_large )
			FILEID="1b1Fhhxsn4SRTmjLtii6Bxmjh_i71bZmr" ;;
		dbn_medium )
			FILEID="1dertTo4oNPxb8u4I3RVP9g1absIyAV78" ;;
		dbn_small )
			FILEID="1eT8bN0DPPtQNLumF92IEHQI31vTLV9gx" ;;
		resnet_large )
			FILEID="1XYuSRsPm1HlDXQRLtvTx9sVmXbgXTCG2" ;;
		resnet_medium )
            FILEID="1KzzvRJkYE4Qu5n7kjwBwbfKyAYYbROui" ;;
		resnet_small )
            FILEID="1DlERgUr2dOPZbPUP16EZ7y2EWjxR0Qy5" ;;
		rnn_large )
			echo -e "Does not support weight file of $1"
			exit 1 ;;
		rnn_medium )
			echo -e "Does not support weight file of $1"
			exit 1 ;;
		rnn_small )
			echo -e "Does not support weight file of $1"
			exit 1 ;;
		lstm_large )
			echo -e "Does not support weight file of $1"
			exit 1 ;;
		lstm_medium )
			echo -e "Does not support weight file of $1"
			exit 1 ;;
		lstm_small )
			echo -e "Does not support weight file of $1"
			exit 1 ;;
		* )
			echo -e "Error: unsupported weight"
			exit 1
	esac
} 

weight_ID $network

wget --no-check-certificate "https://www.dropbox.com/scl/fi/$FILEID/$network.wgh?rlkey=$RLKEY"
mv $network.wgh?rlkey=$RLKEY input.wgh 
