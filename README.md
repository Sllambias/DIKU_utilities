# Mads Nielsen Group compliance guide and tools

Structure and content heavily inspired by the DTU MLOPS course (https://skaftenicki.github.io/dtu_mlops/).

The purpose of this guide is twofold. Primarily it serves to encourage reproducibility and compliance to appropriate standards. This is an end-to-end task requiring information from the earliest stages of environment creation to the final hyperparameter selection. Secondly, it aims to ease knowledge sharing through the use of shared tools.

# Environment 
It is generally encouraged to use virtual environments and environment management software such as miniconda.
To install and use miniconda in Cluster/Hendrix run the following commands in the terminal
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh
```

Required: The current python version

> Python 3.7.15

Obtained: 
In the correct virtual environment run
```
python -V
```

Reproduced:
```
conda create -n myenv python=3.9
```
# Dependencies
Required: 
The list of dependencies required to succesfully run a project in the format of a requirements.txt file

> ...
> batchgenerators==0.24
> matplotlib==3.5.3
> numpy==1.21.6
> Pillow==9.3.0
> SimpleITK==2.2.0
> ...

Obtained: 
To retrieve and save the list of installed packages in a requirements.txt file run the following in the correct virtual environment
```
python -m pip freeze > requirements.txt
```

Reproduced:
```
pip install -r requirements.txt
```









