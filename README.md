# Mads Nielsen Group compliance guide and tools

Structure and content heavily inspired by the DTU MLOPS course (https://skaftenicki.github.io/dtu_mlops/).

The purpose of this guide is twofold. Primarily it serves to encourage reproducibility and compliance to appropriate standards. This is an end-to-end task requiring information from the earliest stages of environment creation to the final hyperparameter selection. Secondly, it aims to ease knowledge sharing through the use of shared tools.

# Environment 
The required python version should always be reported

#### Required: 

> Python 3.7.15

### Obtained:  
in the correct environment run
```
python -V
```

### Miniconda
Managing working enviroments is important, and often multiple environments are necessary to avoid package conflicts, e.g. between PyTorch and Tensorflow installations. To this end virtual environments are practical and can be initialized using the miniconda environment management software.

To install miniconda in Cluster/Hendrix run the following commands in the terminal
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh
```

To create a virtual environment using miniconda the first command initializes an environement with the latest version of python while the second initializes it with a specific version
```
conda create -n ENV_NAME
conda create -n ENV_NAME python=3.9
```
The environment can then be activated
```
conda activate ENV_NAME
```


# Dependencies
The list of dependencies required to succesfully run a project should always be reported in e.g. the format of a requirements.txt file
Required:   

> batchgenerators==0.24   
> matplotlib==3.5.3   
> numpy==1.21.6  
> Pillow==9.3.0  
> SimpleITK==2.2.0   
> ...

To retrieve and save the list of installed packages in a requirements.txt file run the following in the correct environment
Obtained:   
```
python -m pip freeze > requirements.txt
```









