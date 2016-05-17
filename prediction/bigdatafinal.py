__author__ = 'jingyiyuan'
import numpy

import csv
import numpy as np
import sklearn

from pandas.io.data import DataReader
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from collections import OrderedDict

reader = csv.reader(open("joint_hip.csv","rb"),delimiter=',')
x = list(reader)
result = numpy.array(x)
temp = numpy.array(x)

#change categorical data to numeric data
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

#res is the dataset
res = result[1:(len(result)),1:].astype(numpy.float)
for i in range(len(res)):
    for j in range(len(res[0])):
        if(j == 9):
            res[i,j] = int(round(res[i,j]))
        else:
            res[i,j] = int(round(res[i,j]))

myset = set(res[:,9])
nout = len(myset)

y = res[:,9]
x = res[:,0:9]
X_new = x

#ways of feature selection
def randomselect(x):
    print x.shape,"shape"
    clf = ExtraTreesClassifier()
    clf = clf.fit(x, y)
    print clf.feature_importances_
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(x)
    print X_new.shape,"shape"
    return X_new

def varianceselect(x):
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_new = sel.fit_transform(x)
    print X_new.shape,"xnewshape"
    return X_new

def L1select(x):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(x)
    print X_new.shape,"xnewshape"
    return X_new

def treeselect(x):
    clf_tree = ExtraTreesClassifier()
    clf_tree = clf_tree.fit(x, y)
    print clf_tree.feature_importances_
    model_tree = SelectFromModel(clf_tree, prefit=True)
    X_new = model_tree.transform(x)
    print X_new.shape,"xnewshape"
    return X_new, clf_tree.feature_importances_


X_new, importance = treeselect(x)
#print len(X_new)
#print len(y)

#data is the dataset after feature selection
data = np.c_[X_new, y]

print data.shape,"datashape"
print data,"datashape"

#use PCA to do dimension reduction
def dimensionreducPCA(data):
    data_mn = np.mean(data, 0)
    data = data - np.repeat(data_mn.reshape(1, -1), data.shape[0], 0)

    B = np.dot(data.T, data)
    [val,vec] = np.linalg.eigh(B)
    D = vec[:, np.argsort(val)[::-1]]
    c = np.dot(D.T, data.T)
    c = c[:,:1000]

    data = np.dot(c.T, D.T)
    data  = data + np.repeat(data_mn.reshape(1, -1), data.shape[0], 0)
    return data

data = dimensionreducPCA(data)
totallen = len(data)

#randomly divide the data into training and testing set
numpy.random.seed(seed=4)
numpy.random.shuffle(data)
training, testing = data[:(3*totallen/4),:], data[(totallen/4):,:]

print training.shape,'train'
print testing.shape,'test'
l = len(data[0]) - 1
print l,"this is l"

#linear regression
LR = linear_model.LinearRegression()
LR.fit (training[:,0:l], training[:,l])
res1 = LR.predict(testing[:,0:l])

#ridge regression
ridge = linear_model.Ridge(alpha = 0.1)
ridge.fit (training[:,0:l], training[:,l])
res2 = ridge.predict(testing[:,0:l])

#tree
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(training[:,0:l], training[:,l])
res3 = clf.predict(testing[:,0:l])

error1 = 0
for i in range(len(res3)):
    error1 = error1 + abs(res1[i] - testing[i,l])/testing[i,l]
mse1 = error1/len(testing[:,l])
print "average error rate res1"
print mse1


error2 = 0
for i in range(len(res3)):
    error2 = error2 + abs(res2[i] - testing[i,l])/testing[i,l]
mse2 = error2/len(testing[:,l])
print "average error rate res2"
print mse2


error3 = 0
for i in range(len(res3)):
    error3 = error3 + abs(res3[i] - testing[i,l])/testing[i,l]
mse = error3/len(testing[:,l])
print "average error rate res3"
print mse


t = np.arange(0, len(testing), 1)
plt.plot(t, testing[:,l],'r')
plt.plot(t, res3,'b')
#plt.plot(t, res2,'b')
#plt.plot(t, res3,'k')

#plt.show()

#reader = csv.reader(open("joint_hip.csv","rb"),delimiter=',')

#with open("joint_hip.csv", "rb") as source:
#    lines = [row for row in source]
#import random
#random_choice = random.sample(result, 1)

#print random_choice

print result.shape
k = np.random.randint(1, 1000)
input = temp[k,:]
input = ['1442707', '50 to 69', 'M', 'White', '4', 'Elective', 'Home w/ Home Health Services', '203', '153', '1', '$94325.40']
print k,"k"
print temp[k,:]
print result[k,:]

print "INPUT DATA"
print "Age: ",input[1]
print "Gender: ",input[2]
print "Race: ",input[3]
print "Length of Stay: ",input[4]
print "Type of Admission: ",input[5]
print "Patient Disposition: ",input[6]
print "CCS Diagnosis Code: ",input[7]
print "CCS Procedure Code: ",input[8]
print "APR Severity of Illness Code: ",input[9]
#print "Predicted Charges for General Hospital: "

index = importance.argsort()[-l:][::-1]
testdata = result[k,:][index]
costspe = clf.predict(testdata)
print "Predicted Charges for Specialty Hospital: ", costspe
print input[10]