import pandas as pd
import numpy as np 
import seaborn as sns 
sns.set(style='ticks', palette='Set2')
sns.set_context("talk", font_scale=1.2)
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')

from sklearn import svm 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from mlxtend.plotting import plot_decision_regions
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import tensorflow as tf

import math

def perceptron_training(X, y, learning_rate, epochs, tol):
    X_train = np.insert(X, 0, -1, axis=1)

    X_train, y_train = shuffle(X_train, y, random_state = 42)

    err = tol + 0.0001
    err_array = []
    epoch = 0
    wt = np.random.uniform(size = X_train.shape[1])

    while err > tol and epoch < epochs:
        err_ds = 0
        for i, x_i in enumerate(X_train):
            y_hat =  1.0 * (np.dot(wt, x_i) >= 0)
            ei = y_train[i] - y_hat
            dw = learning_rate * ei * x_i
            wt = wt + dw
            err_ds += ei ** 2
        err = err_ds / len(X_train)
        err_array.append(err)
        epoch += 1

    return wt, epoch, err_array


def perceptron_predict(X, wt):
    X_train = np.insert(X, 0, -1, axis=1)
    y_pred = 1.0 * (np.dot(X_train, wt) >= 0)
    return y_pred


def adaline_training(X, y, learning_rate, epochs, tol):
    X_train = np.insert(X, 0, 1, axis=1)
    X_train, y_train = shuffle(X_train, y, random_state = 42)

    err = tol + 0.0001
    err_array = []
    epoch = 0
    wt = np.random.uniform(size = X_train.shape[1])

    while err > tol and epoch < epochs:
        err_ds = 0
        for i, x_i in enumerate(X_train):
            y_hat =  np.dot(wt, x_i)
            ei = y_train[i] - y_hat
            dw = learning_rate * ei * x_i
            wt = wt + dw
            err_ds += ei ** 2
        err = err_ds / len(X_train)
        err_array.append(err)
        epoch += 1

    return wt, epoch, err_array

def adaline_predict(X, wt):
    X_train = np.insert(X, 0, 1, axis=1)
    y_pred = np.dot(X_train, wt)
    return y_pred

