# RGDA-DDI: Residual graph attention network and dual-attention based framework for drug-drug interaction prediction

## Install

```
conda create -n RGDA python=3.7
source activate RGDA
git clone git@github.com:zhangxincode/RGDA.git
cd RGDA  
pip install -r requirement.txt 
```

## Usage

```
python run.py
  -d : Choose which dataset to use, the default is kegg
  -n :  Select the number of neighbor samples, the default is 64
  -hop : Select the depth of neighbor sampling, default is 1
  -b : The value of batchsize for each epoch of training, the default is 1024
  -lr : The value of learning rate for each epoch of training, the default is 1e-2
  -nd : The value of node feature's dimension , the default is 64
  -lc : this parameter decide whether to use Layer-wise aggregation layer,the default is 1
  -c : this parameter decide which attention layer to be used,the default is 3(LaGAT layer)
  -r: this parameter determines whether to export the test results for visualization, the default is 0
  -K: this parameter determines whether to test the generalization ability of the model in the cold start scenario, the default is 0
  -head : this parameter determines the number of Multi-head attention, the default is 1; it only takes effect when the -c parameter value is 2 (GAT layer)
```

## Dataset

We currently provide the KEGG dataset and Drugbank dataset in [raw_data](https://github.com/RGDA/tree/main/raw_data).


## Training

The training adopts `5` rounds of early-stopping strategy, the maximum number of training rounds is set not to exceed `50`, and the regularization coefficient is fixed to `1e-7`; Note that by default we randomly divide the data into 5 folds and take 5-fold cross-validation to test our model. It is also possible to use the `-K `parameter to control the use of new division folds, each fold contains only drugs that appear only in this fold, to test the model's generalization ability to drugs that do not appear in the training set.

