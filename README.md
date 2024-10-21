# ID3 Simple Decision Tree Learning Algorithm

This repository contains a simple implementation of the ID3 decision tree learning algorithm in Python. The ID3 algorithm is a popular machine learning algorithm used for building decision trees based on given data.

## Contents

1. [ID3_Tree.py](ID3_Tree.py): Python script that builds the decision tree using the ID3 algorithm.

![Decision Tree](https://github.com/zeon-X/ID3-simple-decision-tree-learning-algorithm/assets/73699852/fa8eb32a-50e9-4e2b-834e-d9b4409b8cc6)

2. [ID3_Prediction.py](ID3_Prediction.py): Python script for making predictions using the decision tree constructed by ID3_Tree.py.

## How to Use

To build a decision tree using the ID3 algorithm, you can run ID3_Tree.py. It will take your dataset and generate a decision tree based on the provided data.

To make predictions using the decision tree, you can use ID3_Prediction.py. Provide the input data, and it will predict the outcome based on the constructed decision tree.

## Algorithm Overview

ID3 (Iterative Dichotomiser 3) is a classic machine learning algorithm used for constructing decision trees. It works by selecting the best attribute to split the dataset at each step, based on information gain or entropy. The algorithm recursively creates branches in the tree until it reaches a leaf node that provides a classification or prediction.

This repository offers a basic implementation of the ID3 algorithm, which can be extended and customized for specific use cases.

