# Linear Regression with Gradient Descent

## Overview

This project involves implementing a linear regression algorithm in Python3 using gradient descent. The implementation focuses on using vectorization techniques for efficient computation. Additionally, predictions are made using the implemented linear regression on given training and test datasets.

## Files

- **linear_regression.py**: Contains the implementation of several functions for linear regression using vectorization.
- **application.py**: Contains code to make predictions using the linear regression implementation.

## Implementation

### Part 1: Implement Linear Regression with Gradient Descent

In `linear_regression.py`, several functions are implemented to perform linear regression using gradient descent. The functions are vectorized for efficient computation.

### Part 2: Make Predictions using the Implementation

In `application.py`, predictions are made using the implemented linear regression on provided training and test sets. The parameters `alpha` (learning rate) and `number of epochs` are adjusted to ensure the testing loss is smaller than 0.01.

## Results

**Relationship between Alpha and Number of Epochs**

Based on the experiments, the relationship between the learning rate and the number of epochs is observed to be crucial in optimizing model training and convergence. Higher learning rates can speed up convergence but may cause instability, whereas lower learning rates require more epochs for convergence but provide stable training.
