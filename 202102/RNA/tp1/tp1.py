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
from mlxtend.plotting import plot_decision_regions
from scipy.special import softmax
from pathlib import Path

import tensorflow as tf


###############################################################################
#   TF Neural Network Models
###############################################################################   

def tf_build_dataset(X, y, batch_size = 1, training = True):

    if type(X) == pd.DataFrame:
        target = X.pop(y)
        dataset = tf.data.Dataset.from_tensor_slices((X.values, target.values))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if training:
        dataset = dataset.shuffle(buffer_size = 1000).repeat().batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)
        
    dataset = dataset.prefetch(buffer_size = 1)

    return dataset     


def tf_build_predict_dataset(X, batch_size = 1):
    if type(X) == pd.DataFrame:
        dataset = tf.data.Dataset.from_tensor_slices(X.values).batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
    return dataset   


def tf_build_nn_model(hidden_dimensions, 
                        activation, 
                        optimizer, 
                        lr = 0.001, 
                        loss = 'binary_crossentropy', 
                        l2_regularization = None):
    layer = []

    l2 = None
    if l2_regularization is not None:    
        l2 = tf.keras.regularizers.l2(l2_regularization)


    if activation == 'sigmoid':
        act_func = tf.nn.sigmoid
    elif activation == 'tanh':
        act_func = tf.nn.tanh   
    elif activation == 'relu':
        act_func = tf.nn.relu
    else:
        raise ValueError('Activation function not supported')

    for dim in hidden_dimensions:
        layer.append(tf.keras.layers.Dense(dim, activation = act_func, kernel_regularizer = l2))

    # last layer 
    layer.append(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    model = tf.keras.Sequential(layer)

    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(lr = lr)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(lr = lr)
    elif optimizer == 'adagrad':
        opt = tf.keras.optimizers.Adagrad(lr = lr)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(lr = lr)
    else:
        raise ValueError('Optimizer not supported')  

    model.compile(optimizer = opt, loss = loss, metrics=['accuracy', 
                                      tf.keras.metrics.BinaryAccuracy(name = 'binary_acuracy'), 
                                      tf.keras.metrics.AUC(name = 'auc'),
                                      tf.keras.metrics.AUC(name = 'auc_pr', curve = 'PR')])
    return model    


def tf_build_model(X_train, y_train, X_test, y_test, hidden_dimensions, activation, optimizer, lr = 0.001, loss = 'binary_crossentropy', l2_regularization = None, batch_size = 10, epochs = 600):
    nn = tf_build_nn_model(hidden_dimensions = hidden_dimensions, 
                            activation = activation, 
                            optimizer = optimizer, 
                            lr = lr, 
                            loss = loss,
                            l2_regularization = l2_regularization)

    steps_per_epoch = len(X_train) // (batch_size * 10) # 10 epochs for completing the dataset

    train_dataset = tf_build_dataset(X_train, 
                                            y_train,
                                            batch_size = batch_size, 
                                            training = True)

    eval_dataset = tf_build_dataset(X_test, 
                                            y_test, 
                                            batch_size = batch_size,
                                            training = False)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = epochs / 2, mode = 'auto', verbose = 1)]

    history = nn.fit(train_dataset,
                    epochs = epochs,
                    steps_per_epoch = steps_per_epoch,
                    validation_data = eval_dataset, 
                    callbacks = []) # callbacks)

    return nn, history



def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, .1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)  

    fig = plt.figure(figsize=(10, 8))
    _ = sns.scatterplot(X[:,0], X[:,1], hue = y, legend= None, alpha = 0.9)
    plt.contourf(xx, yy, Z, alpha=0.4)
    sns.despine()


from math import copysign, exp
from itertools import accumulate
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm


class VotedPerceptron(BaseEstimator, ClassifierMixin):
    """ Voted Perceptron.
    Parameters
    ----------
    kernel_parameters : dict
        Kernel parameters appropriate for the desired kernel.
        kernel_type : str
            The desired kernel {'linear'|'polynomial'|'gaussian_rbf'}
        degree : int
            Used by kernel_type: polynomial
        gamma : float
            Used by kernel_type: polynomial, rbf
        coef0 : float
            Used by kernel_type: linear, polynomial
    Attributes
    ----------
    kernel : function
        The kernel function defined by kernel_parameters.
    prediction_vector_term_coeffs : list
        The label components of the prediction_vectors.
    prediction_vector_terms : list
        The training case components of the prediction vectors.
    prediction_vector_votes : list
        The votes for the prediction vectors.
    Notes
    -----
    Each prediction vector can be written in an implicit form based
    on its contruction via training.
    A prediction vector is calculated by adding the training label times the
    training input to the previous prediction vector and since the initial
    prediction vector is the zero vector we have
    :math:`v_k = v_{k-1} + y_i x_i` for some label :math:`y_i` and training case :math:`x_i`.
    From this recurrence we see
    :math:`v_k = \sum_{j=1}^{k-1}{y_{i_j} \vec{x}_{i_j}}` for appropriate indices :math:`i_j`.
    Then the kth prediction vector is the kth partial sum of the
    element-wise product of prediction_vector_term_coeffs
    and prediction_vector_terms.
    Specifically the kth prediction_vector is the sum i=1 to i=k of
    prediction_vector_term_coeffs[i] * prediction_vector_terms[i]
    Note: To get an iterable of the prediction vectors explicitly we can do:
    accumulate(pvtc * pvt
                for pvtc, pvt in
                zip(self.prediction_vector_term_coeffs,
                self.prediction_vector_terms))
    Working with prediction vectors in their implicit form allows us to
    apply the kernel method to the voted perceptron algorithm.
    """
    def __init__(self, kernel_parameters, error_threshold=0, max_epochs=1):
        # Set the kernel function
        self.kernel_parameters = kernel_parameters
        self.error_threshold = error_threshold
        self.max_epochs = max_epochs

        self.kernel = self.kernel_function(self.kernel_parameters)

        # Initialize structures that will store the prediction vectors in their
        # implicit form.
        self.prediction_vector_term_coeffs = []
        self.prediction_vector_terms = []

        # Prediction vector votes generated during training.
        self.prediction_vector_votes = []


    def get_params(self, deep = False):
        return {
                    "kernel_parameters" : self.kernel_parameters,
                    "error_threshold" : self.error_threshold,
                    "max_epochs" : self.max_epochs
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, data, labels):
        self.kernel = self.kernel_function(self.kernel_parameters)

        # Initialize structures that will store the prediction vectors in their
        # implicit form.
        self.prediction_vector_term_coeffs = []
        self.prediction_vector_terms = []

        # Prediction vector votes generated during training.
        self.prediction_vector_votes = []

        # Ensure the data dtype is allowed.
        self.check_data_dtype(data)

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis=1)

        # If needed set the initial prediction vector.
        # Initialize the initial prediction_vector to all zeros.
        if len(self.prediction_vector_terms) == 0:
            initial_prediction_vector = np.zeros(data.shape[1], dtype=data.dtype)
            self.prediction_vector_terms.append(initial_prediction_vector.copy())
            self.prediction_vector_term_coeffs.append(1)

        # shape is tuple of dimensions of array (num_rows, num_columns); so shape[0] is
        # be the total number of input vectors.
        num_training_cases = data.shape[0]

        labels_changed = np.where(labels ==0, -1, labels)

        # Set the starting prediction_vector_vote for training to be the last
        # prediction vector vote defined by self.prediction_vector_votes.
        # Note: self.prediction_vector_votes is empty if we have not done any training yet.
        prediction_vector_vote = (self.prediction_vector_votes[-1]
                                  if len(self.prediction_vector_votes) > 0
                                  else 0)

        for _ in range(self.max_epochs):
            num_epoch_errors = 0
            for training_case, training_label in zip(data, labels_changed):
                pre_activation = sum(pvtc * self.kernel(pvt, training_case)
                                     for pvtc, pvt
                                     in zip(self.prediction_vector_term_coeffs,
                                            self.prediction_vector_terms))
                result = copysign(1, pre_activation)

                if result == training_label:
                    prediction_vector_vote += 1
                else:
                    num_epoch_errors += 1

                    # Save the prediction vector vote.
                    self.prediction_vector_votes.append(prediction_vector_vote)

                    # Save new prediction vector term and term coefficient.
                    self.prediction_vector_term_coeffs.append(training_label)
                    self.prediction_vector_terms.append(training_case.copy())

                    # Reset prediction_vector_vote.
                    prediction_vector_vote = 1

            epoch_error = num_epoch_errors / num_training_cases
            if epoch_error <= self.error_threshold:
                # Error for epoch is under the error threshold.
                break

        # Training complete.
        # Save the last prediction_vector_vote.
        self.prediction_vector_votes.append(prediction_vector_vote)

    def decision_function(self, X):
        return self.predict_proba(X)

    def predict_proba(self, X):
        results = [self.predict_proba_singular(x) for x in X]
        return softmax(np.array(results))

    def predict(self, X):
        results = [self.predict_singular(x) for x in X]
        results = np.array(results)
        return np.where(results == -1, 0, results)

    def predict_proba_singular(self, input_vector):
        """ Output of voted perceptron and given input vector.
        Parameters
        ----------
        input_vector : ndarray
            A given state of visible units.
        output_type : str
            Determines output, either 'classification' or 'score'.
                'classification': The label the voted perceptron predicts
                    for the given input_vector.
                'score': The pre_activation value the voted perceptron
                    calculates for the given input_vector.
        Returns
        -------
        float
            If output_type is 'classification' then output the classification
            the voted perceptron predicts for the given input_vector: 1 or -1.
            If output_type is 'score' then output the pre_activation value the
            voted perceptron calculates for the given input_vector.
        """
        # Insert a bias unit of 1.
        input_vector = np.insert(input_vector, 0, 1, axis=0)

        pv_pre_activations = accumulate(pvtc * self.kernel(pvt, input_vector)
                                        for pvtc, pvt
                                        in zip(self.prediction_vector_term_coeffs,
                                               self.prediction_vector_terms))

        pre_activation = sum(
            pvv
            * copysign(1, pvpa)
            for pvv, pvpa
            in zip(self.prediction_vector_votes, pv_pre_activations)
        )

        return pre_activation

    def predict_singular(self, input_vector):
        """ Output of voted perceptron and given input vector.
        Parameters
        ----------
        input_vector : ndarray
            A given state of visible units.
        output_type : str
            Determines output, either 'classification' or 'score'.
                'classification': The label the voted perceptron predicts
                    for the given input_vector.
                'score': The pre_activation value the voted perceptron
                    calculates for the given input_vector.
        Returns
        -------
        float
            If output_type is 'classification' then output the classification
            the voted perceptron predicts for the given input_vector: 1 or -1.
            If output_type is 'score' then output the pre_activation value the
            voted perceptron calculates for the given input_vector.
        """
        return copysign(1, self.predict_proba_singular(input_vector))

    def error_rate(self, data, labels):
        """ Outputs the error rate for the given data and labels.
        Parameters
        ----------
        data : ndarray
            An ndarray where each row is a input vector consisting of the
            state of the visible units.
        labels : ndarray
            An ndarray where each element is the label/classification of a
            input vector in data for binary classification.
            Valid label values are -1 and 1.
        Notes
        -----
        The elements in data must correspond in sequence to the
        elements in labels.
        Returns
        -------
        float
            The error rate of the voted perceptron for the given data
            and labels.
        """
        # Ensure the data dtype is allowed.
        self.check_data_dtype(data)

        # Generate the VotedPerceptron output/classification for each
        # input and save as a numpy array.
        predictions = np.asarray(
            [self.predict(d, 'classification') for d in data], dtype=labels.dtype
        )

        # Gather the results of the predictions; prediction_results is an ndarray corresponding
        # to the predictions and the labels for the data with True meaning the prediction matched
        # the label and False meaning it did not.
        prediction_results = (predictions == labels)
        # Note the number of incorrect prediction results
        # (i.e. the number of False entries in prediction_results).
        num_incorrect_prediction_results = np.sum(~prediction_results)
        # Note the number of results.
        num_prediction_results = prediction_results.shape[0]
        # Compute the error rate.
        error_rate = num_incorrect_prediction_results / num_prediction_results

        return error_rate

    @staticmethod
    def kernel_function(kernel_parameters):
        """ Output the chosen kernel function given the name and parameters.
        Parameters
        ----------
        kernel_parameters : dict
            Kernel parameters appropriate for the desired kernel.
            kernel_type : str
                The desired kernel {'linear'|'polynomial'|'gaussian_rbf'}
            degree : int
                Used by kernel_type: polynomial
            gamma : float
                Used by kernel_type: polynomial, gaussian_rbf
            coef0 : float
                Used by kernel_type: linear, polynomial
        Returns
        -------
        function
            The chosen kernel function with the appropriate parameters set.
        Raises
        ------
        NotImplementedError
            If strategy not in ('OVA', 'OVO')
        """
        def linear(vector_1, vector_2):
            """
            Linear Kernel
            """
            coef0 = kernel_parameters["coef0"]
            output = np.dot(vector_1, vector_2) + coef0
            return output
        def polynomial(vector_1, vector_2):
            """
            Polynomial Kernel
            """
            gamma = kernel_parameters["gamma"]
            coef0 = kernel_parameters["coef0"]
            degree = kernel_parameters["degree"]
            output = (gamma * np.dot(vector_1, vector_2) + coef0) ** degree
            return output
        def gaussian_rbf(vector_1, vector_2):
            """
            Gaussian Radial Basis Function Kernel
            """
            gamma = kernel_parameters["gamma"]
            vector_difference = vector_1 - vector_2
            output = exp(-gamma * np.dot(vector_difference, vector_difference))
            return output

        kernel_choices = {'linear': linear,
                          'polynomial': polynomial,
                          'gaussian_rbf': gaussian_rbf}

        kernel_type = kernel_parameters['kernel_type']

        if kernel_type not in kernel_choices:
            raise NotImplementedError(kernel_type)

        kernel_choice = kernel_choices[kernel_type]

        return kernel_choice

    @staticmethod
    def check_data_dtype(data):
        """ Check to see if the data dtype is a valid predesignated type.
        Parameters
        ----------
        data : ndarray
            An ndarray where each row is a input vector consisting of the
            state of the visible units.
        Raises
        ------
        TypeError
            If data.dtype is not a valid predesignated type.
        Notes
        -----
        We require data.dtype to be float32 or float64. When numpy is built
        with an accelerated BLAS the dtypes eligible for accelerated operations
        are float32, float64, complex64, complex128 with the latter 2 not relevant
        for the voted perceptron implementation here.
        Also by restricting to floating point types we minimize the possibility of
        any unanticipated overflow issues with regards to np.dot.
        By having this check a user cannot pass in say mnist data as uint8.
        While uint8 does fit mnist data it leads to unexpected np.dot calculations.
        More specifically np.dot does not upcast or warn when integer overflow occurs
        (numpy bugs 4126, 6753).
        e.g. a = np.asarray([1,128], dtype=uint8)
             b = np.asarray([0,2], dtype=uint8)
             np.dot(a,b) would return 0.
        """
        if data.dtype not in (np.float32, np.float64):
            raise TypeError('data dtype required to be float32 or float64')

    @staticmethod
    def validate_inputs(input_vector_size, data, labels):
        """ Validate inputs used for the train method of the voted perceptron.
        Parameters
        ----------
        input_vector_size: The number of visible units.
        data: An ndarray where each row is a input vector consisting of the
              state of the visible units.
        labels: An ndarray where each element is the label/classification of a
                input vector in data for binary classification.
                Valid label values are -1 and 1.
        Note the elements in data must correspond in sequence to the
        elements in labels.
        Raises
        ------
        ValueError
            If the given combination of input_vector_size, data, labels is not
            valid.
        Notes
        -----
        To be used by instantiator to double check that their input data has been
        properly preprocessed.
        """
        # Ensure the number of data items matches the number of labels.
        if len(data) != len(labels):
            raise ValueError("Number of data items does not match"
                             + " the number of labels.")

        # Ensure self.input_vector_size matches size of each item in data.
        if any(input_vector_size != len(data_item) for data_item in data):
            raise ValueError("A data item size does not match"
                             + " input_vector_size.")

        # Ensure set of label values in [-1, 1].
        if not np.all(np.in1d(labels, [-1, 1])):
            raise ValueError("Valid label values are -1 and 1;"
                             + " adjust labels accordingly when calling"
                             + " this function.")


class Perceptron(BaseEstimator, ClassifierMixin):

    def __init__(self, tol, epochs, learning_rate):
        self.tol = tol
        self.epochs = epochs
        self.learning_rate = learning_rate

    def get_params(self, deep = False):
        return {
                    "tol" : self.tol,
                    "epochs" : self.epochs,
                    "learning_rate" : self.learning_rate
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit (self, X, y):
        X_train = np.insert(X, 0, -1, axis=1)

        err = self.tol + 0.0001
        err_array = []
        epoch = 0
        wt = np.random.uniform(size = X_train.shape[1])

        while err > self.tol and epoch < self.epochs:
            err_ds = 0
            for i, x_i in enumerate(X_train):
                y_hat =  1.0 * (np.dot(wt, x_i) >= 0)
                ei = y[i] - y_hat
                dw = self.learning_rate * ei * x_i
                wt = wt + dw
                err_ds += ei ** 2
            err = err_ds / len(X_train)
            err_array.append(err)
            epoch += 1

        return wt, epoch, err_array


    def predict(self, X):
        X_train = np.insert(X, 0, -1, axis=1)
        y_pred = 1.0 * (np.dot(X_train, self.wt) >= 0)
        return y_pred

def parse_dataset_control(file_name):
    data = pd.read_csv(file_name, header = None)
    labels = ['normal'] * 100 + ['cyclic'] * 100 + ['increasing'] * 100 + ['decreasing'] * 100 + ['upward_shift'] * 100 + ['downward_shift'] * 100
    data['label'] = labels

    return data

import csv

def parse_lp_data(filename = 'lp1.data'):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile, delimiter="\t") 
        dataset = list(lines)
        labels = []
        final = []
        #print (dataset, '\n\n')
        x = 0
        while x < (len(dataset)):
            dados =[]
            #primeira informação é a classe
            classe = str(dataset[x][0])
            labels.append(classe)
            x = x+1
            d = 0
            y=1
            while d < 15:
                linha = []
                linha = dataset[x+d]
                d= d+1
                linha = linha[1:]
                i=0
                while i < len(linha):
                    linha[i] = int(linha[i])
                    i=i+1
                if len(dados) == 0:
                    dados = linha
                else:
                    dados = np.hstack((dados, linha))
            x = x+17
            dados = np.squeeze(dados)

            if len(final) == 0:
                final = dados
            else:
                final = np.vstack((final,dados))
            
        df = pd.DataFrame(final)
        df['label'] = labels

        return df