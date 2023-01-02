# Mads Nielsen Group compliance guide and tools
Structure and content heavily inspired by the DTU MLOPS course (https://skaftenicki.github.io/dtu_mlops/).

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Environment](#environment)
- [Dependencies](#dependencies)
- [Data Management](#data-management)
- [Conding Practices](#coding-practices)


The purpose of this guide is twofold. Primarily it serves to encourage reproducibility and compliance to appropriate standards. This is an end-to-end task requiring information from the earliest stages of environment creation to the final hyperparameter selection. Secondly, it aims to ease knowledge sharing through the use of shared tools.

All the sections are structured similarly. First **Required** and **Obtained** presents the information that must be reported for compliance and reproduciblity along with how that information can be obtained. Then additional tools and tips are provided for managing the subject in a reasonable and sustainable way. For the seasoned programmer the **Required** parts are often sufficient and can serve as a check list.

# Environment 
The python version used in the project should always be reported

#### Required: 

> Python 3.7.15

#### Obtained:  
in the correct environment run
```
python -V
```

## _Miniconda_
Managing working enviroments is important, and often multiple environments are necessary to avoid package conflicts, e.g. between PyTorch and Tensorflow installations. To this end virtual environments are practical and can be initialized using the miniconda environment management software.

To install miniconda in Cluster/Hendrix run the following commands in the terminal
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh
```

To create a virtual environment using miniconda with either 
(1) the latest version of python or
(2) a specific version of python
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
#### Required:   

> batchgenerators==0.24   
> matplotlib==3.5.3   
> numpy==1.21.6  
> Pillow==9.3.0  
> SimpleITK==2.2.0   
> ...

To retrieve and save the list of installed packages in a requirements.txt file run the following in the correct environment
#### Obtained:   
```
python -m pip freeze > requirements.txt
```
# Data Management
#### Required:   
#### Obtained:   

# Coding Practices
#### Required:   
#### Obtained: 

### _Hard coded variables_

### _Paths_









