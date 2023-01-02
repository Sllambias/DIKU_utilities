# Mads Nielsen Group compliance guide and tools

Structure and content heavily inspired by the DTU MLOPS course (https://skaftenicki.github.io/dtu_mlops/).

The purpose of this guide is twofold. Primarily it serves to encourage reproducibility and compliance to appropriate standards. This is an end-to-end task requiring information from the earliest stages of environment creation to the final hyperparameter selection. Secondly, it aims to ease knowledge sharing through the use of shared tools.

# Environment & Dependencies

## Environment
What: 
The current python version

> Python 3.7.15

How: 
In the correct virtual environment run
```
python -V
```

## Dependencies
What: 

run the following in the correct enviroment:
While the dependencies for a given environment can be retrieved and saved in a requirements.txt file using
```
python -m pip freeze > requirements.txt
```
