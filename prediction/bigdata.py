"""
Source Code for Homework 4.b of ECBM E6040, Spring 2016, Columbia University

Instructor: Prof. Aurel A. Lazar

"""

import numpy
import numpy as np
import csv
import pandas as pd
from collections import OrderedDict
from pandas import DataFrame as df
import theano
from theano import tensor as T
import sys

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

from utils import shared_dataset
from nn import myMLP, train_nn, HiddenLayer, LogisticRegression
    
#TODO: build and train a MLP to learn parity function 
def test_mlp_parity(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=64, n_hidden=500, n_hiddenLayers=1,
             verbose=False):
    
    reader = csv.reader(open("joint_knee.csv","rb"),delimiter=',')

    x = list(reader)
    #print x
    result = numpy.array(x)
    #print result.shape

    def score_to_numeric(x, a):
        if (x == 'Hospice - Home'):
            return 11
        if (x == 'Psychiatric Hospital or Unit of Hosp'):
            return 10
        if (x == 'Hospice - Medical Facility'):
            return 9
        if (x == 'Expired'):
            return 8
        if (x == 'Facility w/ Custodial/Supportive Care'):
            return 7
        if (x.lower() == 'left against medical advice'):
            return 6
        if (x.lower() == 'short-term hospital'):
            return 5
        if (x.lower() == 'multi-racial' or x.lower() == 'home or self care'):
            return 4
        if (x.lower() == 'other race' or x.lower() == 'emergency' or x.lower() == 'skilled nursing home' or x.lower() == 'not available'):
            return 3
        if (x.lower() == 'm' or x.lower() == 'black/african american' or x.lower() == 'urgent' or x.lower() == 'inpatient rehabilitation facility'):
            return 2
        if (x.lower() == 'f' or x.lower() == 'white' or x.lower() == 'elective' or x.lower() == 'home w/ home health services'):
            return 1
        if (a == 1):
            return int(x[:2])
        if (a == 2):
            return float(x[1:])
        else:
            return float(x)

    rownum = 0
    for row in result:
        # Save header row.
        if rownum == 0:
            rownum += 1
            header = row
            for i in range(0, len(header)):
                if header[i].lower() == 'gender':
                    gender = i
                if header[i].lower() == 'race':
                    race = i
                if header[i].lower() == 'type of admission':
                    admi = i
                if header[i].lower() == 'patient disposition':
                    disp = i
                if header[i].lower() == 'age group':
                    age = i
                if header[i].lower() == 'total charges':
                    price = i
        else:
            row[gender] = score_to_numeric(row[gender],0)
            row[race] = score_to_numeric(row[race],0)
            row[admi] = score_to_numeric(row[admi],0)
            row[disp] = score_to_numeric(row[disp],0)
            row[age] = score_to_numeric(row[age],1)
            row[price] = score_to_numeric(row[price],2)
            for i in range(0, len(row)):
                row[i] = float(row[i])
                #y = row[i].astype(numpy.float)
                #row[i] = y
                #print type(row[i])

    #print type(result)
    #result = numpy.array(result).astype('float')
    #print result[1:(len(result)),1:]
    res = result[1:(len(result)),1:].astype(numpy.float)
    for i in range(len(res)):
        for j in range(len(res[0])):
            if(j == 9):
                res[i,j] = int(round(res[i,j]/10000))
            else:
                res[i,j] = int(round(res[i,j]))

    myset = set(res[:,9])
    nout = len(myset)

    y = res[:,9]
    #print y
    x = res[:,0:9]

    iris = load_iris()
    clf = ExtraTreesClassifier()
    clf = clf.fit(x, y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(x)
    data = np.c_[X_new, y]

    totallen = len(data)
    numpy.random.shuffle(data)
    training, validation, testing = data[:totallen/2,:], data[totallen/2:(3*totallen/4),:], data[(3*totallen/4):,:]

    l = len(data[0]) - 1

    train_set = [training[:,0:l],training[:,l]]
    valid_set = [validation[:,0:l],validation[:,l]]
    test_set  = [testing[:,0:l],testing[:,l]]
    
    #print train_set
    #print valid_set
    #print test_set

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=l,
        n_hidden=n_hidden,
        n_out=len(myset),
        n_hiddenLayers=n_hiddenLayers
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    
    y_p_train = theano.function(
        inputs=[],
        outputs=[classifier.logRegressionLayer.y_pred],
        givens={
            x: train_set_x
        }
    )
    
    y_predict = theano.function(
        inputs=[],
        outputs=[classifier.logRegressionLayer.y_pred],
        givens={
            x: test_set_x
        }
    )
    y_pred1 = y_p_train()
    y_pred2 = y_predict()
    
    return y_pred1, y_pred2

if __name__ == '__main__':
    test_mlp_parity()
