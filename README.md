# Description
This repository contains an easy-to-use Python function for the ESP prediction model from our paper [A general model to predict small molecule substrates of enzymes based on machine and deep learning](https://doi.org/10.1038/s41467-023-38347-2). 


## Downloading data folder
Before you can run the ESP prediction function, you need to [download and unzip a data folder from Zenodo](https://doi.org/10.5281/zenodo.7981153). Afterwards, this repository should have the following strcuture:

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

The listed packages can be installed using conda and anaconda:

```bash
pip install pandas==1.3.1
pip install torch==1.12.1
pip install numpy==1.23.1
pip install fair-esm==0.4.0
conda install -c conda-forge py-xgboost=1.3.3
conda install -c rdkit rdkit=2022.09.5
```


## Problems/Questions
If you face any issues or problems, please open an issue.

