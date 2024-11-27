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
            DATAID="hh5x16pdokes2wmbwos54"
            RLKEY="ce6nspqvyrylknemr69s6rsr9"
            ;;
		imagenet_small )
            while true; do
                read -p "Do you have ImageNet file? (y/n) " yn
                case $yn in 
                    [yY]* ) 
                        DATAID="i3zy0tai78kfzrwbjga8s"
                        RLKEY="ui33vsy7twg24fik5wcxa4106"
                        break
                        ;;
                    [nN]* )
                        DATAID="wwnpvkhp05ab5fvidccfy"
                        RLKEY="2oiprelet8e3decgxy8fah9qi"
                        break
                        ;;
                    * ) echo -e "Invalid response"
                        ;;
                esac
            done
            ;;
		nist_large )
            DATAID="4n2tqnmz6z67iemky0vfg"
            RLKEY="ikz2l8bpdza10uuhg5s8hznsh"
            ;;
		nist_medium )
            DATAID="vw2kli048c0u2g5ba5si3"
            RLKEY="09yjc2esgbrgfzy6cei28l8sn"
            ;;
		nist_small )
            DATAID="6n3prj8gtf1yjghsls740"
            RLKEY="fiwqracaf1aep73upk4mc3dba"
            ;;
		ptb_large )
            DATAID="5g3whqx7wu91u4oabwihg"
            RLKEY="jeyenm1qr4t1ctgkhlby9j2qj"
            ;;
		ptb_medium )
            DATAID="5krv13vfjiu2snfbv8csk"
            RLKEY="5wnpae1s3zco7gw91oz68yjvf"
            ;;
		ptb_small )
            DATAID="fcbv4uyeb9vd1h194livu"
            RLKEY="lyom6acdjhpz6vxm85uxjgre6"
            ;;
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
tar jxf $dataset.tar && rm $dataset.tar
#tar xf $dataset.tar && rm $dataset.tar

./list.sh 
