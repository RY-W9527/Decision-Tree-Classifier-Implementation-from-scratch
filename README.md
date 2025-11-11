# Intro2ML CW1: Training A Decision Tree Model with Cross Validation
This project implements a Decision Tree model trained using both k-fold and k fold nested cross validation (CV) method. To ensure future reusability and clean integration, the model and utilities are implemented using object-oriented class design.

## Project Background

This project was originally developed as part of a **group coursework for COMP70050 – Introduction to Machine Learning** at **Imperial College London**.  
The original repository is hosted on the Imperial College GitLab.  
This GitHub version is a **personal reconstruction** for portfolio and educational display purposes.

### Group Authors
- **Xiaoxin Deng**
- **Zhaodong Guo**
- **Ruiyang Wang** 
- **Yiming Wang**

### Personal Contribution (by Ruiyang Wang)
- Implemented the **core Decision Tree algorithm from scratch**, including:
  - Recursive tree construction logic: decision_tree_learning()  
  - Feature splitting rules for continuous attributes: find_split()
  - Information Gain (IG) and entropy-based decision criteria: find_split()  
- Contributed to the **nested k-fold cross-validation** framework for model evaluation  
- Conducted **model evaluation and pruning experiments** and provided part of the analytical report

---

## Installation
### Download the project From GitLab
Assuming you have access to the Imperial College London GitLab:
```
git clone https://gitlab.doc.ic.ac.uk/yw8025/intro2ml-cw1.git
cd intro2ml-cw1
```
### Dataset
Please ensure the directory ```wifi_db/``` is placed under the project root.

## Activate virtual Environment
```source /vol/lab/ml/intro2ml/bin/activate```

## Getting Started
Simply run:
```
python main.py
```
This will start training the model using cross validation
The framework follows these steps:
1. Load Clean and Noisy Dataset using the ```Dataloader```
2. Perform k-fold cross validation on both datasets. This will create 10 models each. After finishing CV, the plots and results are saved to ```results/k_fold```
3. Perform nested cross validation on both datasets. This will create 90 models for each dataset.  After finishing CV, the plots and results are saved to ```results/k_fold_nested```.
The process will take a bit of time since we are training $10\times2 + 90\times2$ models in series. 

## Class Overview
### Dataloader
Loads dataset files into memory from the given path. You can split data into three fixed partitions with dataloader, but this is not required in this project.

### Metric
Tracks performance metrics during cross validation:
1. ```value``` stores a dictionary of all historical metric values computed during the CV process. It is updated by calling ```update()``` class function.
2. ```average``` stores a dictionary of average of ```value```. It is computed once ```get_average_metrics``` is called.
3. Fn ```compute()``` computes all metrics or a specific metric by name.
This class wraps a metric operation module with reusable functions.

### DecisionTreeModel
Encapsulates model training, prediction, pruning, and visualization.

### DecisionTreeVisualizer
Helper class for plotting, used internally by DecisionTreeModel.

## Roadmap
Enable parallel execution of cross validation using multithreading to reduce total training time.

## Acknowledgments
This project was developed as part of the coursework for *COMP70050 Introduction to Machine Learning* at Imperial College London.
We would like to thank the course instructors and teaching assistants for their guidance.
