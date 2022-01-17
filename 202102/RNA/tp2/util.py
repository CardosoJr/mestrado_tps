from re import escape
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
from scipy.linalg import pinv2

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


def elm_train(X, y, hidden_dim = 1000):
    X_train = np.insert(X, 0, 1, axis=1)
    X_train, y_train = shuffle(X_train, y)    
    Z = np.random.normal(size = [X_train.shape[1], hidden_dim])
    H = np.tanh(np.dot(X_train, Z))
    W = np.dot(pinv2(H), y_train)
    return W, Z

def elm_predict(X, Z, W):
    X_test = np.insert(X, 0, 1, axis=1)
    H = np.tanh(np.dot(X_test, Z))
    Yhat = np.sign(np.dot(H, W))
    return Yhat

def radial_function(x, m, r):
    return np.exp(-0.5 * np.matmul((x-m).T, (x-m)) / (r **2))

def elm_rbf_train(X, y, hidden_dim = 10):
    X_train = np.insert(X, 0, 1, axis=1)
    X_train, y_train = shuffle(X_train, y) 
    Z = np.random.normal(size = [hidden_dim, X_train.shape[1]]) @ np.cov(X_train, rowvar = False) + np.mean(X_train, axis = 0)
    dists = np.sqrt(((Z[:,None,:] - Z)**2).sum(axis=2))
    r = np.mean(dists[np.triu_indices(dists.shape[0], k = 1)])  
    H = np.zeros([X_train.shape[0], hidden_dim])
    for i in range(X_train.shape[0]):
        for j in range(hidden_dim):
            val = radial_function(X_train[i], Z[j,:], r)
            H[i,j] = val

    W = np.dot(pinv2(H), y_train)
    return W, Z, r

from sklearn.cluster import KMeans

def kmeans_rbf_train(X, y, hidden_dim = 10):
    X_train = np.insert(X, 0, 1, axis=1)
    X_train, y_train = shuffle(X_train, y)
    Z = KMeans(n_clusters = hidden_dim).fit(X_train).cluster_centers_
    dists = np.sqrt(((Z[:,None,:] - Z)**2).sum(axis=2))
    r = np.mean(dists[np.triu_indices(dists.shape[0], k = 1)]) 
    H = np.zeros([X_train.shape[0], hidden_dim])
    for i in range(X_train.shape[0]):
        for j in range(hidden_dim):
            H[i,j] = radial_function(X_train[i], Z[j,:], r)

    W = np.dot(pinv2(H), y_train)
    return W, Z, r

def elm_rbf_predict(X, Z, W, r):
    X_test = np.insert(X, 0, 1, axis=1)
    hidden_dim = Z.shape[0]
    H = np.zeros([X_test.shape[0], hidden_dim])
    for i in range(X_test.shape[0]):
        for j in range(hidden_dim):
            H[i,j] = radial_function(X_test[i], Z[j,:], r)

    Yhat = np.sign(np.dot(H, W))
    return Yhat

class ELM:
    def __init__(self, hidden_dim = 1000):
        self.hidden_dim = hidden_dim
        self.mapper = None
        self.mapping = False

    def fit(self, X, y):
        classes = np.sort(np.unique(y))
        self.mapping = False
        self.mapper = {classes[0] : classes[0], classes[1] : classes[1]}
        if len(classes) > 2:
            raise Exception("Not supported multiclass")
        elif classes[0] != -1:
            self.mapper[classes[0]] = -1
            self.mapping = True
        elif classes[1] != 1:
            self.mapper[classes[1]] = 1
            self.mapping = True
        
        if self.mapping:
            print(self.mapper)
            y_train = np.vectorize(self.mapper.get)(y)
        else:
            y_train = y

        X_train = np.insert(X, 0, 1, axis=1)
        self.Z = np.random.normal(size = [X_train.shape[1], self.hidden_dim])
        H = np.tanh(np.dot(X_train, self.Z))
        self.W = np.dot(pinv2(H), y_train)

    def predict(self, X):
        X_test = np.insert(X, 0, 1, axis=1)
        H = np.tanh(np.dot(X_test, self.Z))
        Yhat = np.sign(np.dot(H, self.W))
        if self.mapping:
            inv_map = {v: k for k, v in self.mapper.items()}
            Yhat = np.vectorize(inv_map.get)(Yhat) 
        return Yhat


class RBF:
    def __init__(self, hidden_dim = 10):
        self.hidden_dim = hidden_dim

    def fit(self, X, y):
        X_train = np.insert(X, 0, 1, axis=1)
        X_train, y_train = shuffle(X_train, y)
        self.Z = KMeans(n_clusters = self.hidden_dim).fit(X_train).cluster_centers_
        dists = np.sqrt(((self.Z[:,None,:] - self.Z)**2).sum(axis=2))
        self.r = np.mean(dists[np.triu_indices(dists.shape[0], k = 1)]) 
        H = np.zeros([X_train.shape[0], self.hidden_dim])
        for i in range(X_train.shape[0]):
            for j in range(self.hidden_dim):
                H[i,j] = radial_function(X_train[i], self.Z[j,:], self.r)

        self.W = np.dot(pinv2(H), y_train)

    def predict(self, X):
        X_test = np.insert(X, 0, 1, axis=1)
        H = np.zeros([X_test.shape[0], self.hidden_dim])
        for i in range(X_test.shape[0]):
            for j in range(self.hidden_dim):
                H[i,j] = radial_function(X_test[i], self.Z[j,:], self.r)

        Yhat = np.sign(np.dot(H, self.W))
        return Yhat