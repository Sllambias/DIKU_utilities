# Mads Nielsen Group compliance guide and tools
Structure and content heavily inspired by the DTU MLOPS course (https://skaftenicki.github.io/dtu_mlops/).

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Project Requirements](#project-requirements)
- [Environment](#environment)
- [Dependencies](#dependencies)
- [Preprocessing](#preprocessing)
- [Coding Practices](#coding-practices)


The purpose of this guide is twofold. Primarily it serves to encourage reproducibility and compliance to appropriate standards. This is an end-to-end task requiring information from the earliest stages of environment creation to the final hyperparameter selection. Secondly, it aims to ease knowledge sharing through the use of shared tools.

All the sections are structured similarly. First, **Required** and **Obtained** describes the information that should always be reported for compliance and reproduciblity along with examples of how that information looks and can be obtained. Subsequently, project management tools relevant to the section are provided (e.g. [_Miniconda_](#miniconda)). For the seasoned programmer the **Required** parts are often sufficient and can serve as a check list.

# Project Requirements
Bullet point version of project requirements with links to descriptions and assistance.

- Environment & Dependencies
  - [Python version](#environment-reportables)
  - [Dependencies/Requirements](#dependency-reportables)
- Data
  - [Preprocessing Scripts](#data-reportables)

# Environment 

### _Miniconda_
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
conda create -n MY_TORCH_ENVIRONMENT
conda create -n MY_TORCH_ENVIRONMENT python=3.9
```
The environment can then be activated
```
conda activate MY_TORCH_ENVIRONMENT
```

#### Environment Reportables: 
To ensure that others can reproduce your results, they must be able to create an environment identical to yours (see Docker for an alternative solution), starting with the python version. The python version(s) used in the project should be reported as:

> Python 3.7.15

Obtained by running the following in the appropriate environment:
```
python -V
```

# Dependencies
#### Dependency Reportables: 

The list of dependencies required to succesfully run a project should always be reported in e.g. the format of a requirements.txt file

> batchgenerators==0.24   
> matplotlib==3.5.3   
> numpy==1.21.6  
> Pillow==9.3.0  
> SimpleITK==2.2.0   
> ...

To retrieve and save the list of installed packages in a requirements.txt file, run the following in the correct environment
```
python -m pip freeze > requirements.txt
```

# Preprocessing
The exact preprocessing that is performed is always extremely important to make sure results are reproducible and to enable future debugging. This could include documentation about: 
- The random seed used to select slices, splits, etc. 
- Reorientation implementations
- Resampling implementations
- Preprocessing pipelines
- How images are loaded and stored (e.g. loading NIfTI images with SimpleITK and saving them with numpy transposes images)

#### Preprocessing Reportables: 
Include the scripts used to process each dataset along with the order they are applied in.
For a project using the nnUNet framework this could include the:

- [Dataset Conversion](Example_Scripts/DatasetConversion_FakeTask.py) script preparing the data for the nnUNet framework 
- [Experiment Planner](Example_Scripts/DIKUExperimentPlanner.py) used to select the preprocessing pipeline and save relevant information  
- [Preprocessor Pipeline](Example_Scripts/DIKUPreprocessor.py) carrying out the preprocessing operations  


# Coding Practices

Very importantly once data or results based on the scripts are used anywhere: DO NOT CHANGE THEM.
Instead, if you need to change MyScript.py create MyScriptV2.py. 

#### Coding Reportables: 

### _Hard coded variables_

### _Paths_









