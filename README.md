# Description
This repository contains an easy-to-use Python function for the ESP prediction model from our paper [A general model to predict small molecule substrates of enzymes based on machine and deep learning](https://doi.org/10.1038/s41467-023-38347-2). 


## Downloading data folder
Before you can run the ESP prediction function, you need to [download and unzip a data folder from Zenodo](https://doi.org/10.5281/zenodo.8046233). Afterwards, this repository should have the following strcuture:

    ├── code                   
    ├── data                    
    └── README.md

## How to use the ESP prediction function
There is a Jupyter notebook "Tutorial ESP prediction.ipynb" in the folder "code" that contains an example on how to use the ESP prediction function.

## Requirements

- python 3.8
- jupyter
- pandas 1.3.1
- torch 1.12.1
- numpy 1.23.1
- rdkit 2022.09.5
- fair-esm 0.4.0
- py-xgboost 1.3.3

The listed packages can be installed using micromamba (or conda or anaconda) and pip as follows:

```bash
micromamba create -n esp -c conda-forge pandas==1.3.1 python=3.8 jupyter  numpy==1.23.1 fair-esm==0.4.0 py-xgboost=1.3.3 rdkit=2022.09.5
micromamba activate esp
micromamba remove py-xgboost
pip install xgboost
```
You can use `conda` instead of `micromamba`. This method is tested on Macbook pro 2021 Intel Chip on 14.02.2024.

## Problems/Questions
If you face any issues or problems, please open an issue.

