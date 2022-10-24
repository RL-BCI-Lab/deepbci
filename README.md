# About
*WARNING: This repository is highly volatile and frequent reworks are often conducted!*

Deep-bci is a repository for running deep learning algorithms on a biosensing data. The current focus of the project is on EEG data, deep neural networks, and deep reinforcement learning. 

The goal of this repository is to allow for biosensing datasets to be combined and utilized by deep learning algorithms. If you know anything about biosensing datasets or EEG datasets you should know how each dataset is usually uniquely formatted. One of the main goals of this projects is to equalize all the diverse datasets within the biosensing field so that they can be used in conjunction with one anther. The target audience for this project are those who "wish to be close to the code". This repository should be interacted with and edited. 

# Requirements 
To use this repository you will need to setup the following conda environment (or simply acquire the listed dependencies below). The below are the contents of a yaml file for automatically creating a conda environment. Refer to conda's official [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for creating a conda environment from a yaml file.

Alternatively, you can use a pre-built Docker containers (see the Docker section below).
```
name: mlenv
channels:
  - conda-forge
  - plotly
  - anaconda
dependencies:
  - python=3.9
  - pip
  - conda-forge::opencv
  - conda-forge::pyyaml
  - conda-forge::numpy
  - conda-forge::pandas=1.5.2
  - conda-forge::matplotlib=3.5.3
  - conda-forge::seaborn
  - conda-forge::jupyterlab
  - conda-forge::jupyter_contrib_nbextensions
  - conda-forge::scipy=1.9.1
  - conda-forge::scikit-learn=1.1.3 
  - conda-forge::mpi4py
  - conda-forge::dill
  - conda-forge::psutil
  - conda-forge::rpy2
  - conda-forge::cmake
  - plotly::plotly-orca
  - plotly::plotly
  - conda-forge::dash
  - conda-forge::dash-daq
  - plotly::jupyter-dash
  - conda-forge::mne
  - conda-forge::tqdm
  - pip:
    - tensorflow==2.9.0
    - tensorboard
    - hydra-core
    - pygame
    - zepid==0.9.0
```
# Docker

If you are using Linux or Windows Linux Subsystem (WSL), it is highly recommended that you use the pre-built Docker containers or build your own Docker contain. Docker contains will contain all the necessary packages and requirements for running the code and make dependency handling vastly easier!

If you wish to utilize a local GPU you will need to use a version of Linux where [nvidia-docker](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html) is supported. The nvidia-docker allows users to mount their GPUs into a docker container, which is crucial for deep learning. WSL has support for GPUs only on Windows 10 insider builds and Windows 11.

The main Docker image is the `deep-bci:tf` which contains [TensorFlow with GPU](https://hub.docker.com/repository/docker/bpoole908/deep-bci). Even if you don't have a GPU this is still the image you should work with. Note, that none of the `deep-bci` images contain the deep-bci repository code. Thus, you will need to mount your deep-bci directory into the container. If you make your container by sourcing the environment script  `docker/env.sh`, these alias commands will mount your repository automatically. 

## Building Containers
There are easy to access commands that can be used by sourcing the environment script located at `./deep-bci/docker/env.sh` script. There are various commands you can pass to this environment script which will help determine how the container is made (privileges) and which image tags to download.

*Note: All of the following commands should be ran from the root directory of deep-bci!*

First, we must source the `env.sh` script to get access to the alias build commands.
```
source docker/env.sh
```

Once done, we can run one of the aliases to temporarily build a container with no GPU. We can do soo by running the below command which will build, attach, and then destroy said container upon shutdown. 

```
dbci-attach
```

To make a temporary container with a GPU, simply use the following command.

```
dbci-gpu-attach
```

If you want a persistent container you can use the following command. This command will build a docker container that will not be destroyed when shutdown. 

```
# No GPU
dbci-make

# GPU
dbci-gpu-make
```

Upon running this command the container will start running. However, you will have to manually attach to the container using the following command.

*Note: If you are not using the `tf` image tag you need to replace `tf` with whatever the image tag name is!*

```
# No GPU
docker attach dbci-tf

# GPU
docker attach dbci-gpu-tf
```

# Brief overview
Below is a brief overview of what each directory or module does within (and without) the deep-bci repository.

## deepbci/ 
Within this directory is all the code the relates to the deepbci module. Any code outside the deep-bci directory can not be accessed through the imported deepbci module.

### deepbci/games/
The `games/` directory contains tasks that are paired with EEG recording for data collection. These tasks can be used for deep reinforcement learning as well.

### deepbci/models/
The `models/` directory contains all the code related to machine learning algorithms. If you want to add new algorithms then you can do so by adding them here. Make sure to you implement the necessary base classes (abstract classes) if you wish to do so.

Currently, the code base ONLY supports the Sklearn and TensorFlow libraries. There are plans to support PyTorch in the future.

### deepbci/data_utils/
The `data_utils/` directory contains all the code loading, grouping, and transforming/pre-processing (we call mutating) BCI data. If you wish to add new datasets that can be loaded and utilized by this repository here is where you would do so. Make sure to you implement the necessary base classes (abstract classes) if you wish to do so.

### deepbci/utils
The `utils/` directory contains all code considered to be "utilities". This directory contains a wide verity of code and is currently where most of the helper functions are put. *WARNING: this code is highly unorganized and will be reworked greatly at a later date.*

## scripts/
This directory is not contained within the deep-bci repository but examples of how to use the repository to train classifiers, visualize BCI data, build the our data, and conduct BCI experiments. *WARNING: this code is highly unstable and frequently changes!* See the `./deep-bci/experiments/classification/readme.md` for more details for conducting you own experiments (no coding necessary, just editing of config files).

