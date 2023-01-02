# Mads Nielsen Group compliance guide and tools

Structure and content heavily inspired by the DTU MLOPS course (https://skaftenicki.github.io/dtu_mlops/).

The purpose of this guide is twofold. Primarily it serves to encourage reproducibility and compliance to appropriate standards. This is an end-to-end task requiring information from the earliest stages of environment creation to the final hyperparameter selection. Secondly, it aims to ease knowledge sharing through the use of shared tools.

# Environment 


What: 
The current python version

> Python 3.7.15

How: 
In the correct virtual environment run
```
python -V
```

# Dependencies
What: 
The list of dependencies required to succesfully run a project in the format of a requirements.txt file

> ...
> batchgenerators==0.24
> matplotlib==3.5.3
> numpy==1.21.6
> Pillow==9.3.0
> SimpleITK==2.2.0
> ...

How: 
To retrieve and save the list of installed packages in a requirements.txt file run the following in the correct virtual environment
```
python -m pip freeze > requirements.txt
```









