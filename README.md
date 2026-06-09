# BearingNAS

## Install
Create a python virtual environment for BearingNAS

`python -m venv bearingnas_venv`

Activate it

`source bearingnas_venv/bin/activate`

Install tensorflow

`python -m pip install tensorflow`

Install tensorflow_model_optimization

`python -m pip install tensorflow_model_optimization`

## Build the CWRU Dataset

Download the following files and put them in the `./data` folder
[baseline](https://engineering.case.edu/sites/default/files/97.mat), [inner race 0.007"](https://engineering.case.edu/sites/default/files/105.mat), [ball 0.007"](https://engineering.case.edu/sites/default/files/118.mat), [inner race 0.014"](https://engineering.case.edu/sites/default/files/169.mat), [ball 0.014"](https://engineering.case.edu/sites/default/files/185.mat), [inner race 0.021"](https://engineering.case.edu/sites/default/files/209.mat), [ball 0.021"](https://engineering.case.edu/sites/default/files/222.mat), [outer race 0.007" Centered](https://engineering.case.edu/sites/default/files/130.mat), [outer race 0.007" Orthogonal](https://engineering.case.edu/sites/default/files/144.mat), [outer race 0.007" Opposite](https://engineering.case.edu/sites/default/files/156.mat)

Run `python build_numpy_dataset.py`

## Enjoy BearingNAS

 `python search.py`


