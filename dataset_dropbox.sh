#!/bin/bash

# Nebula main directory
nebuladir=$PWD
# Dataset directory
datadir=$nebuladir/datasets

print_help() {
    echo -e "Usage: $0 <dataset> <size>"
    echo -e ""
    echo -e "<dataset> options: imagenet, nist, ptb"
    echo -e "<size> options: large, medium, small"
    exit 0
}


if [[ "$#" -lt 2 || $1 = '-h' || $1 = '--help' ]]; then
    print_help
fi

### Default arguments ###
# Read lowercase of dataset type.
dataset=${1,,}; shift

# Read lowercase of size options.
size=${1,,}; shift

### Concatenate size option to dataset type.
if [[ $size = 'large' ]]; then
    dataset+=_large
elif [[ $size = 'medium' ]]; then
    dataset+=_medium
elif [[ $size = 'small' ]]; then
    dataset+=_small
else
    echo -e "Error : Wrong size option $size"
    exit 1
fi

##### Translate google drive link ID #####
data_ID() {
	case $dataset in 
		imagenet_large )
            DATAID="o1yo5az42nsfj8gezb5d1"
            RLKEY="2nebshox1jbjy7xlosqkgqhxg"
            ;;
		imagenet_medium )
            DATAID="1RspmHUeV3qyOYnqtbramv61mCDiu1B7r" ;;
		imagenet_small )
			DATAID="1ueT5X7ICjuBuC9DkVGWm8j6u7hWakCNn" ;;
		nist_large )
			DATAID="1TO1cIXUQtNeb3InQf4iDV9hDbA97MJGx" ;;
		nist_medium )
			DATAID="1d_Wue5Dype3KucgWZnqQbhqGcGd20o8E" ;;
		nist_small )
			DATAID="1ufIjaGJTJNdyAdY-jZkx9PUSshqYeEIX" ;;
		ptb_large )
            DATAID="1ErHMRMyxTNqaEKk4tuPrq-f-EVpW0enK" ;;
		ptb_medium )
            DATAID="164c-EBWwPmcfUUeauBwCS5tx1CCO8uXQ" ;;
		ptb_small )
            DATAID="1WImqQjk1hC4ZtNWci29zl8f8Zfmv4RbX" ;;
		mnist )
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

if [[ ! -d $datadir/$dataset ]]; then
	mkdir $datadir/$dataset
fi

# Get Dropbox ID of Nebula dataset
data_ID $dataset

# Download Nebula dataset from Dropbox

wget --no-check-certificate "https://www.dropbox.com/scl/fi/$DATAID/$dataset.tar?rlkey=$RLKEY"
mv $dataset.tar?rlkey=$RLKEY $dataset.tar

# Unzip the dataset and make list(label list, test list, and train list)
tar xf $dataset.tar && rm $dataset.tar
sh list.sh && rm list.sh 
