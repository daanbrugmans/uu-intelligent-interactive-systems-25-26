# Installation Instructions

These instructions apply both to the Crash Course and all the labs. You only need to do this process once.

## Install Python

[Python](https://www.python.org/) is the programming language of choice for this course.

Check if you have a Python 3 installation available. The easiest way is to run

```shell
> python --version
Python X.X.XX
```

In some systems, `python` might refer to a Python 2 installation (version `2.X.XX`). In this case, you might want to check the following:

```shell
> python3 --version
Python 3.X.XX
```

If you don't have Python 3 available, follow the system-specific instructions on [the official webpage](https://www.python.org/). 

**The material in this course was tested on Python 3.12**.

### Important Notes

*__Windows:__ Make sure to tick the box "Add Python to your path" during the installation process.*

*__MacOS:__ Make sure to run the command `Install Certificates.command` after you install Python.*

## Creating a Python Virtual Environment

While not strictly required, this is highly recommended as it will ensure the correct versions of the libraries.

* Step 1: Install `virtualenv`
```bash
pip install virtualenv
```

* Step 2: Create the Virtual Environment
```bash
python3 -m venv env
```
Replace `env` with your desired environment name.

* Step 3: Activate the Virtual Environment
- On **Linux/macOS**:
```bash
source env/bin/activate
```

- On **Windows**:
```bash
env\Scripts\activate
```

After activation, you should see the environment name in your terminal prompt.

## Install CUDA for PyTorch

In some of the Labs and your Final Project you will use PyTorch, a GPU-accelerated Machine Learning framework. While it can run on both CPU and GPU, running on the GPU makes it significantly faster i.e. it will make a *big* difference in how much time you spend staring at your computer while it thinks. To ensure that it can access your GPU, you need to follow some manual installation steps:

1. Make sure you have a [CUDA-enabled](https://developer.nvidia.com/cuda-gpus) or [ROCm-enabled](https://github.com/ROCm/ROCm.github.io/blob/master/hardware.md) GPU.

2. If your GPU is ROCm-enabled, install it following the procedure here: [ROCm](https://rocmdocs.amd.com/en/latest/). If your GPU is CUDA-enabled, download and install its latest version from here: [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)


**Windows users (this is not required for linux/macos users):** Before you install the various libraries you need to install PyTorch manually with the following:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
If you do this, then also remember to remove torch from the `requirements.txt` file!

## Install the Requirements

Download `requirements.txt` from Canvas. This file indicates which libraries we need from the Python Package Index (PyPI), and which version we would like. To install it, run:

```shell
pip install -r requirements.txt
```

*__Note:__ If you are using Poetry, Conda, or another version control system, check their specific instructions on how to install `requirements.txt`.*

## Download OpenFace pre-trained models

To download pre-trained models for OpenFace, run:

```shell
# cannot install openface with the rest due to dependency problems
pip install --no-deps openface-test==0.1.26

openface download
```

## Troubleshoot

* You can run shell (command line) commands from within jupyter notebook file itself. Use '!' symbol at the start of the line.
For example you may install packages into the environment where your jupyter notebook or VSCode is running, by executing a cell like "!pip install -r requirements.txt" You may also need to use a full path to requirements.txt in this command if it is not in the same folder as your notebook file is.

* Some issues in previous years were caused by different python installations, where the required modules were installed in one and jupyter notebook or VSCode was actually using another one. Few commands that may help you to diagnose it from within jupyter notebook (to run in a code cell):
"!python --version"  (which will show the version of python in the environment, does not necessarily match the version of kernel that is being running)
"!pip list --version" (which will list the versions of modules installed in the environment, it should match the versions in the requirements.txt)
"import sys
sys.version" (two lines. This will show the version of the kernel that is running, it should match the version of the environment python)
