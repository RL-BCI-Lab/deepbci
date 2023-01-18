# CIBCI 2022

This branch is dedicated to reproducing the results reported in our paper [*Error-related Potential Variability: Exploring the Effects on Classification and Transferability*](https://arxiv.org/abs/2301.06555).

# Quick Start
To simply download all the pre-built results including plots, statistical analysis results, and classification results from the paper, use the following download given below links (links to replicated analysis are provided as well). If you wish to build the plots from scratch or perform classification from scratch see the below sections.

- Paper
  - [Classification results](https://drive.google.com/file/d/14IctHzHOm1UTeq4fO8DIep0UoVJP8dUY/view?usp=share_link)
  - [Statistical analysis results and plots](https://drive.google.com/file/d/1CFKI6Yyc3X_7l4rASvD1AcJgPUKEnHoI/view?usp=sharing)
- Replication
  - [Classification results](https://drive.google.com/file/d/11GIog1bpSASE-sUFjvq24MWcQa1h2bdl/view?usp=sharing)
  - [Statistical analysis results and plots](https://drive.google.com/file/d/1LcDOnzln4BFzB7RvLZMvEA-wG7p9GdFk/view?usp=share_link) 

# Environment

## Conda Environment 
*Warning: These requirements may diverge from the main branch requirements in the future!*

To use this branch of the repository you will need to setup the conda environment or simply install the dependencies listed. All dependencies are provided in the conda yaml file below. Refer to conda's official [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for creating a conda environment from a yaml file.

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
## Docker Environment
Alternatively, if you are using Linux or Windows Linux Subsystem (WSL), it is highly recommended that you use the pre-built Docker containers or build your own Docker contain. Docker contains will contain all the necessary packages and requirements for running the code and make dependency handling vastly easier!

If you wish to utilize a local GPU you will need to use a version of Linux where [nvidia-docker](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html) is supported. The nvidia-docker allows users to mount their GPUs into a docker container, which is crucial for deep learning. WSL has support for GPUs only on Windows 10 insider builds and Windows 11.

The Docker image for the CIBCI branch is called `deep-bci:cibci` and can be manually download from [DockerHub](https://hub.docker.com/repository/docker/bpoole908/deep-bci) if needed. Note, that none of the `deep-bci` images contain the deep-bci repository code. Thus, you will need to mount your deep-bci directory into the container. If you follow the recommended instructions below, you'll download the required image and create your container by sourcing the environment script  `docker/env.sh`. The below instructions will also automatically mount the deep-bci repository into your container. 

### Running the Container
To quickly pull the docker image and create a running container it is recommended that you utilize the existing `deep-bci/docker/env.sh` environment script. By sourcing this script, various aliases will be added to your command line. Depending on your OS and setup, you may have to manipulate some arguments within this file. For instance, it is possible to change the UID and GID if your system does not default to `1000:10`.

*Note: All of the following commands should be ran from the root directory of the deep-bci repository.*

To source the environment script to access the various alias build commands, enter the following command.
```
source docker/env.sh
```

Once done, you can run one of the aliases to temporarily build a container with no GPU. You can do soo by running the below command which will build, automatically attach you to the container, and then destroy said container upon exiting the container by running the `dbci-attach` alias command. 

```
dbci-attach
```

To make a temporary container with a GPU, simply use the following command.

```
dbci-gpu-attach
```

If you want a persistent container you can use the following command. This command will build a docker container that will not be destroyed after you exit the container. 

```
# No GPU
dbci-make

# GPU
dbci-gpu-make
```

Upon running a variation of the `-make` command the container will start running. However, you will have to manually attach to the container using the following command.

```
# No GPU
docker attach dbci-cibci

# GPU
docker attach dbci-gpu-cibci
```

### Building an Image
To build a Docker image from scratch you can run the following command. It is highly recommended you use the the pre-built image unless you are very familiar with Docker. 

```
docker build -f docker/Dockerfile -t bpoole908/deep-bci:cibci --force-rm --no-cache --build-arg VERSION=tf .
```

# Running Classification
*Coming soon: we have not yet released the data but are working on doing so. Until then, users will have to download our classification results.*

# Running Statistical Analysis and Building Plots

All code for building plots is contained within jupyter notebooks. The notebooks for building statistical analysis plots are located in `deep-bci/scripts/classification` directory. There are two notebooks for building the result plots: `significance_analysis.ipynb` and `significance_visualization.ipynb`. More detailed instructions about the code are given within the notebooks. It is important to note that these notebooks provide code for automatically downloading the classification results.

The `significance_analysis.ipynb` notebook contains all the code for generating the statistical analysis results. The `significance_visualization.ipynb` notebook contains all the code for visualizing and plotting the  statistical analysis results. To build and visualize the results from the paper simply run the all cells in `significance_analysis.ipynb` first and then run all cells in `significance_visualization.ipynb`. These notebooks can also be modified to use your own classification results if they were produced by this repository.

The `plotly_signal_vis_gui.ipynb` notebook is used for building the grand average plots and is located within the `deep-bci/scripts/data-exploration` directory. *Warning: this notebook requires access to the raw EEG data which is not available yet.**

# Citation

```
@inproceedings{poole_errpvar_2022,
    title = {Error-related Potential Variability: Exploring the Effects on Classification and Transferability},
    author = {Poole, Benjamin and Lee, Minwoo},
    booktitle={IEEE Symposium Series on Computational Intelligence in Brain-Computer Interfaces},
    year = {2022},
    organization={IEEE}
}
```