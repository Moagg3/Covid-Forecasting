# CSE 8803 EPI Project

This repository contains the code and documents for the project for CSE 8803 EPI Fall 2022 course. The topic of the project is "Removing Media Coverage Bias from
COVID-19 Symptom Search Trends Data". All the source code and data is organized in the [`src`](src) folder, and the [`doc`](doc) folder contains the project report and presentation slides.

## Setup

1. Create a clean conda environment with Python 10:

```
$ conda create --name epi-project --no-default-packages python=3.10
$ conda activate epi-project
```

2. Install the dependencies

```
$ pip install -r requirements.txt
```

## Update Dependencies (dev)

1. Add new packages to [`requirements.in`](requirements.in) with or without specific versions.

2. Run the following command to auto-solve the dependencies:

```
$ pip-compile requirements.in
```

## Experiments

All of the experiments are documented in notebooks in the `./src/` directory, prefixed by `experiment_`.

There are three notebooks, each corresponds to a different model:

1. [`experiments_lampos.ipynb`](src/experiments_lampos.ipynb): Baseline Lampose et al.
2. [`experiments_linear_regression.ipynb`](src/experiments_linear_regression.ipynb): Linear Regression (LR)
3. [`experiments_rnn.ipynb`](src/experiments_rnn.ipynb): Recurrent neural network (RNN)

Once you have the dependencies installed, you can run the notebooks in the order the code cells are presented to see the experimentation and results. For `example_rnn.ipynb`, we present an example training an RNN with media data and with a sequent length of 7.

## Artifacts

All the numerical results are saved in the [`./src/results`](src/results/) directory and all the plots are saved in the [`./src/plots`](src/plots/) directory. You the files are named according to the model and experiment trial name. You can find the correponding source code in the correpsonding `experiments_<model>.ipynb` notebooks in the [section](#experiments) above.

The project report is available at [`cse_8803_epi_project_final_report.pdf`](doc/cse_8803_epi_project_final_report.pdf).