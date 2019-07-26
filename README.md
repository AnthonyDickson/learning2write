# learning_to_write
Teaching a neural network how to write characters.

## Getting started
1.  Install the required system packages:
    ```bash
    $ sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
    ```
    
    See the prequisites section of [stable-baselines.readthedocs.io](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites) 
    for instructions for other operating systems.

2.  Set up the python environment using conda:
    ```bash
    $ conda env create -f environment.yml
    ```
    or if you are not using conda, then make sure you have a python environment
    set up with all of the packages listed in the file `environment.yml`.
    
3.  Activate the conda environment:
    ```bash
    $ conda activate learning_to_write