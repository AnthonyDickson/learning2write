#!/usr/bin/env bash

echo Installing system dependencies...
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev unzip xvfb python-opengl

echo Installing Miniconda...
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh 
rm -f miniconda.sh

echo Getting EMNIST dataset...
mkdir emnist_data
cd emnist_data
wget http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
unzip gzip.zip
mv gzip/* .
rmdir gzip
rm gzip.zip
cd ..

echo Done!
