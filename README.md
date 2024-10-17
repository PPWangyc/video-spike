
# IBL-Video Project

## Environment Setup

  

To get started, ensure you have git-lfs and anaconda installed. If git-lfs is not installed, please follow the installation instructions [here](https://git-lfs.github.com/).

If you are using a HPC system, try to use `module load` or `module spide`r to find git-lfs.

Execute the following commands to set up the project environment:

  

```bash

cd  script

# make sure you installed git-lfs and anaconda

# The code and scripts are tesed on Linux (Ubuntu 20.04)

source  setup.sh

```

---

This setup script will automatically configure the required conda environment and clone the dataset from [Hugging Face](https://huggingface.co/datasets/PPWangyc/ibl-video). The source data originates from the [International Brain Lab](https://int-brain-lab.github.io/iblenv/notebooks_external/data_release_repro_ephys.html).

## Train an End-to-End Model

Navigate to the `script` directory and execute the training script:

```bash
cd script
source train.sh
```

This script will train a Linear model to predict neural spikes based on raw behavioral video data.