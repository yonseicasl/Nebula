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
            DATAID="dk7dqy27equ0qbxsx71so"
            RLKEY="v0zt350i6swoph5dhb0yue7r6"
            ;;
		imagenet_medium )
            DATAID="o1yo5az42nsfj8gezb5d1"
            RLKEY="2nebshox1jbjy7xlosqkgqhxg"
            ;;
		imagenet_small )
            DATAID="7560yxeurqs5qcvyhadrx"
            RLKEY="6vy9q68zgz26lwp1v580mgt63"
            ;;
		nist_large )
            DATAID="ww9hoxm3rdyyr5edvdhbb"
            RLKEY="8s4kgzb4sftwy2d7mscn8d5np"
            ;;
		nist_medium )
            DATAID="c1d2hkksij0ghv863tqwd"
            RLKEY="6s4wh8i368xwiz1s96b2o58lu"
            ;;
		nist_small )
            DATAIS="wstw38ckcv0d0b9cbfei9"
            RLKEY="kwu05v94gnagrw3lzf7nz59js"
            ;;
		ptb_large )
            echo -e "Does not support dataset of $1" 
			exit 1 ;;
		ptb_medium )
            echo -e "Does not support dataset of $1" 
			exit 1 ;;
		ptb_small )
            echo -e "Does not support dataset of $1" 
			exit 1 ;;
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

cd $datadir/$dataset

wget --no-check-certificate "https://www.dropbox.com/scl/fi/$DATAID/$dataset.tar?rlkey=$RLKEY"
mv $dataset.tar?rlkey=$RLKEY $dataset.tar

# Unzip the dataset and make list(label list, test list, and train list)
tar xf $dataset.tar && rm $dataset.tar

./list.sh 
