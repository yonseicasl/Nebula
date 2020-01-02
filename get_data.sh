#!/bin/bash

nebuladir=$PWD
datadir=$nebuladir/data

##### Translate google drive link ID #####
get_data_ID() {
	echo -e "Get data ID of $1"
	case $1 in 
		ImageNet )
			DATAID="1_6XD4mIjcTrzJTmRnVgj6ME-nahJ8pCW" ;;
		ImageNet_L )
			DATAID="1s0f7kVMYfopFG40QuVX4hDVsEY7d_70B" ;;
		ImageNet_M )
			DATAID="1ed4h30oumph7xdyq4cZUUTFA_WEsZuSE" ;;
		ImageNet_S )
			DATAID="1GU9nEOxrmhMsJhXyL2j13PqEXLoh8QcA" ;;
		NIST_L )
			DATAID="1L16EtnjaE6f1OHo6ndMJ6eWIfzWDgY5o" ;;
		NIST_M )
			DATAID="1ssCfVl4pkxQH6EmkdojN3U5sHqqZ8EqR" ;;
		NIST_S )
			DATAID="1L9hBx7zaXrpF9OW3KruMplDkptJIUCjH" ;;
		MNIST )
			DATAID="1Ya8Ts8-VjJsV0RcQabGtx359OcJJBVLe" ;;
		* )
			echo -e "Error: unsupported dataset"
			exit 1
	esac
}

if [[ ! -d $datadir ]]; then
	mkdir $datadir
fi

# Choose a database type.
read -p "Which dataset? [ImageNet[I] / NIST[N] / MNIST[M]] " ans
if [[ $ans = "ImageNet" || $ans = "I" ]]; then
	dataset=ImageNet
elif [[ $ans = "NIST" || $ans = "N" ]]; then
	dataset=NIST
elif [[ $ans = "MNIST" || $ans = "M" ]]; then
	dataset=MNIST
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
