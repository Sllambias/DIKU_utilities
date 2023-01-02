# Mads Nielsen Group compliance guide and tools

Structure and content heavily inspired by the DTU MLOPS course (https://skaftenicki.github.io/dtu_mlops/).

The purpose of this guide is twofold. Primarily it serves to encourage reproducibility and compliance to appropriate standards. This is an end-to-end task requiring information from the earliest stages of environment creation to the final hyperparameter selection. Secondly, it aims to ease knowledge sharing through the use of shared tools.

# Environment & Dependencies

To retrieve the python version used for a project such as:
> Python 3.7.15
run the following in the correct enviroment:
```
python -V
```

While the dependencies for a given environment can be retrieved and saved in a requirements.txt file using
```
python -m pip freeze > requirements.txt
```
