# Learning2write
Teaching a neural network how to write characters.
The environment provided has three sets of patterns (mostly letters and digits) 
that an agent can be trained on:
- A set of simple 3x3 patterns 
- A set of 5x5 patterns 
- Letters and digits from EMNIST (MNIST with both digits and letters).

The goal is for the agent to fill in the squares in a grid to reproduce the pattern
that it has been presented as accurately as possible (see below for an example).

**Fig 1.** Example of an ACKTR RL agent trained on 5x5 patterns. 

![An example of a trained agent in a 5x5 environment](./5x5_demo.gif)

## Getting started
There is a convenience script `setup.sh` which assumes a UNIX based system and automates most of the setup process. 
If you use this script then restart your terminal after successfully running it (to get conda set up correctly) and 
skip to step 3. 

1.  Install the required system packages:
    ```bash
    sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev unzip xvfb python-opengl
    ```
    
    See the prerequisites section of [stable-baselines.readthedocs.io](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites) 
    for instructions for other operating systems.

2.  Set up the python environment using conda:
    ```bash
    conda env create -f environment.yml
    ```
    or if you are not using conda, then make sure you have a python environment
    set up with all of the packages listed in the file `environment.yml`.
    
3.  Activate the conda environment:
    ```bash
    conda activate learning2write
    ```
    
4.  Train a model:
    ```bash
    python train.py -updates 1000000 -n-workers=4
    ```
    
5.  Test a previously trained model:
    ```bash
    python test.py -model-path models/acktor_mlp_5x5.pkl
    ```
    
6.  You can see the help text for these scripts by adding the flag `-h` or `--help`.

## The EMNIST Dataset
[EMNIST](https://www.nist.gov/node/1298471/emnist-dataset) is an image dataset that is a superset of MNIST and also include alphabetic characters.
This dataset is needed to train CNN-based RL models (the handmade patterns are too small).

**Fig 2.** Sample of images from the EMNIST dataset. 
![Sample of EMNIST Images](./EMNIST_sample.png)

To get the dataset run the following:
```bash
mkdir emnist_data
cd emnist_data
wget http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
unzip gzip.zip
mv gzip/* .
rmdir gzip
cd ..
```
