#!/bin/bash

print_help() {
    echo -e "Usage: $0 <network> <size>"
    echo -e ""
    echo -e "<network> options: alexnet, dbn, mlp, resnet50, vgg"
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
            FILEID="ww49ab035d0a9dnr6a0yg"
            RLKEY="0d6pyldbyyvve201gj8p6ysjd"
            ;;
		alexnet_small )
            FILEID="xftc2qad6qzkoahpl7bps"
            RLKEY="ldx036geqe9imw77a36abblci"
            ;;
		vgg_large )
            FILEID="7qtpkr5qgn419x4bfmhio"
            RLKEY="b5b2ec277y3qq1264etnxthxi"
            ;;
		vgg_medium )
            FILEID="io1heo0u8ytbtrnelm98e"
            RLKEY="xapo2k06q8mhd246vejikjysy"
            ;;
		vgg_small )
            FILEID="2adpc22v9cfu121wqngbn"
            RLKEY="89j5mi9hvai16u9nzlrhutpuf"
            ;;
		mlp_large )
			FILEID="v7hrjk3yslmihlkqnv0df"
            RLKEY="5i4r8qlmwi4mre7nj9n652x7o"
            ;;
		mlp_medium )
			FILEID="oc1kf9d770hrdhqtuh8qb"
            RLKEY="pxn0yv56ph4wbf853iockpmsp"
            ;;
		mlp_small )
			FILEID="2afw3bq5m0hp0cszo6pk3"
            RLKEY="2xc9ospokaxp3imqbunhsf6ts"
            ;;
		dbn_large )
			FILEID="qc5k6fw2071fb5kzcl36q"
            RLKEY="az50l7os9dttgf5d1q7muz2rb"
            ;;
		dbn_medium )
			FILEID="cmrjlqf27qsrfsosjkj6s"
            RLKEY="7ou81c9t3cls8vzj7s96jwnrp"
            ;;
		dbn_small )
			FILEID="3w2ner7n4peq8w9w3i7ij"
            RLKEY="zhgxz461oesu1gjdquidm8026"
            ;;
		resnet_large )
			FILEID="9iotd5dh3yenn1v1gonee" 
            RLKEY="mj4dfjtoyi7v353r44n8uxipk"
            ;;
		resnet_medium )
            FILEID="m6n4rpfi1tgzcrdw56zxb"
            RLKEY="x12u1k60nqnlqov89iomtar4o"
            ;;
		resnet_small )
            FILEID="92z9ygdf63zczwtxfxbb3"
            RLKEY="9ow8xjuidm928ao4wnd5by4ro"
            ;;
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

cd weights

wget --no-check-certificate "https://www.dropbox.com/scl/fi/$FILEID/$network.wgh?rlkey=$RLKEY"
mv $network.wgh?rlkey=$RLKEY benchmarks/$network/input.wgh 
