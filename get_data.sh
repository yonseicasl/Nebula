#!/bin/bash

nebuladir=$PWD
datadir=$nebuladir/data

print_help() {
    echo -e "The supported datasets are : "
    echo -e "ImageNet-L, ImageNet-M, ImageNet-S"
    echo -e "NIST-L, NIST-M, NIST-S"
    echo -e "PTB-L, PTb-M, and PTB-S"
    exit 0
}


if [[ $1 = '-h' || $1 = '--help' ]]; then
    print_help
fi

##### Translate google drive link ID #####
get_data_ID() {
	echo -e "Get data ID of $1"
	case $1 in 
		ImageNet_L )
            DATAID="1FAOrcFcpRbR0mMaWv5vzUq-PeyIJHsFv" ;;
		ImageNet_M )
            DATAID="1UeFJa35DOhwR_iaxbX15_e30ETFSo4O8" ;;
		ImageNet_S )
			DATAID="1zCTB82sCmBcf0tCapOa0cd7tAcPp9ADe" ;;
		NIST_L )
			DATAID="1BiZkpBSoT562Qfacy95KHP7Z1-MoZwX8" ;;
		NIST_M )
			DATAID="1grcL1Ktahh1dgUbKqLIOv7BR7hzJmjo8" ;;
		NIST_S )
			DATAID="1pRW2Uovd8r1Fd-qKb6mMjFbFnXGd-JMu" ;;
		PTB_L )
            echo -e "Does not support dataset of $1" 
			exit 1 ;;
		PTB_M )
            echo -e "Does not support dataset of $1" 
			exit 1 ;;
		PTB_S )
            echo -e "Does not support dataset of $1" 
			exit 1 ;;
		MNIST )
            echo -e "Does not support dataset of $1" 
			exit 1 ;;
		* )
			echo -e "Error: unsupported dataset"
			exit 1
	esac
}

if [[ ! -d $datadir ]]; then
	mkdir $datadir
fi

# Choose a database type.
read -p "Which dataset? [ImageNet[I] / NIST[N] / MNIST[M]] / PTB[P] " ans
if [[ $ans = "ImageNet" || $ans = "I" ]]; then
	dataset=ImageNet
elif [[ $ans = "NIST" || $ans = "N" ]]; then
	dataset=NIST
elif [[ $ans = "MNIST" || $ans = "M" ]]; then
	dataset=MNIST
elif [[ $ans = "PTB" || $ans = "P" ]]; then
	dataset=PTB
else
	echo -e "Wrong option for dataset"
	exit 1
fi

# Choose a database size.
read -p "Which size? [Large[L] / Medium[M] / Small[S]] " ans
if [[ $ans = "O" || $ans = "o" ]]; then
	dataset=$dataset
elif [[ $ans = "L" || $ans = "l" ]]; then
	dataset+=_L
elif [[ $ans = "M" || $ans = "m" ]]; then
	dataset+=_M
elif [[ $ans = "S" || $ans = "s" ]]; then
	dataset+=_S
else 
	echo -e "Wrong option"
	exit 1
fi

if [[ ! -d $datadir/$dataset ]]; then
	mkdir $datadir/$dataset
fi

# Get google drive ID of Nebula dataset.
get_data_ID $dataset
# Download Nebula dataset from google drive.
# Store the dataset to data directory
cd $datadir/$dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$DATAID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$DATAID" -O $dataset.tar && rm -rf /tmp/cookies.txt

# Unzip the dataset and make list(label list, test list, and train list)
tar xf $dataset.tar && rm $dataset.tar
sh list.sh
