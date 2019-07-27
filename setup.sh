#!/usr/bin/env bash

echo Installing system dependencies...
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev unzip

echo Installing Miniconda...
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -f -p $HOME/miniconda
rm -f miniconda.sh

echo Creating conda environment...
conda env create -f environment.yml

echo Getting EMNIST dataset...
mkdir emnist_data
cd emnist_data
wget http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
unzip gzip.zip
mv gzip/* .
rmdir gzip
cd ..

echo Done!
