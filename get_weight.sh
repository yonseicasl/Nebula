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
			FILEID="1URezybxLRa7P-Aazhs8jAI0ocjbgz-Yx" ;;
		vgg_S )
			FILEID="1yPVjxbpiuSrW1Qi9Q8Saj5TsNwIkhCzL" ;;
		vgg_M )
			FILEID="1f4JDDC3aH_MtuVK2zh_5E2ffJzHq6vzC" ;;
		vgg_L )
			FILEID="1DZoMJZBKHeUyFiLEABRsoWfwBhDTF6Z9" ;;
		vgg )
			FILEID="1v9MlkPA9oNen_zHRaL8IjOktbIWFz5WH" ;;
		mlp_S )
			FILEID="1yP1wIk25tisGAQ006fw3UUboUtGZqFEq" ;;
		mlp_M )
			FILEID="1nIPvpNSVxkguRKSW9zOjBqUT4W05DHJP" ;;
		mlp_L )
			FILEID="1cyHNOJxTj1fsbSI1JPvvNjPhlbfSvosW" ;;
		resnet )
			exit 1 ;;
		resnet_L )
			exit 1 ;;
		resnet_M )
			exit 1 ;;
		resnet_S )
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
