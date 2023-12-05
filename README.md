# scPriorGraph：Constructing Biosemantic Cell-Cell Graphs with Prior Gene Set Selection for Cell Type Identification from scRNA-seq Data
<img src="./figure.jpg" width="900">

We created a method scPriorGraph based on Python and R that uses `torch` to identify cell types in scRNA-seq data. This project is a simple implementation of scPriorGraph.

## Requirements
### Python requirements
+ Python == 3.9
+ torch == 1.11.0
+ rpy2 == 3.5.10

### R requirements
+ R == 4.2.2
+ GSEABase == 1.60.0
+ AUCell == 1.20.2
+ SingleCellExperiment == 1.20.1
+ SNFtool == 2.3.1

## Create environment

```
conda create -n scPriorGraph python=3.9
conda activate scPriorGraph
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
```

## Installation
You can install scPriorGraph by copying the code directly from GitHub. We recommend using PyCharm as the Python IDE to run this method. This project needs to invoke R to run the code. Please make sure that you have already installed the R environment (version 4.2.2) and have installed the required R packages on your computer before running this project. 

## Usage
### Step 1: Prepare data
The standard data format that the model can accept is a gene expression matrix where rows represent cells and columns represent genes. The cell label file should have cell indices that match those in the expression matrix. The reference dataset and the query dataset should have the same gene names and the same number of genes, and gene names should be in lowercase.

### Step 2: Training the model and Prediction

```
python ./scPriorGraph.py --ref bh --query se --pathway KEGGHuman
```

#### Input
+ `ref`: The name of the dataset used for training the model.
+ `query`: The name of the dataset used for prediction.
+ `pathway`: The pathway selected by the user.

#### Output
+ `pred.csv`: The predicted results for cell types in the query dataset.

## Datasets
You can download the sample dataset from Google Drive at the following link: https://drive.google.com/drive/folders/1sFbU3Wd9Ai1vlNeJH6_QkxLwWBI0WKSS?usp=sharing

After downloading the files, copy `L_bh.csv` and `M_bh.csv` into the `./data/ref` folder in the root directory, and copy `L_se.csv` and `M_se.csv` into the `./data/query` folder in the root directory.