

import numpy as np
import pandas as pd
import csv
import random
readerTrain = np.loadtxt(
    #had to get rid of major to be able to use the other values as floats, skips row one (headers)
    open("dataset/train_reghanData_noMajor.csv", "r"), delimiter=",", skiprows=1)
x = list(readerTrain)
resultTrain = np.array(x).astype("double")
#print('train', '\n', resultTrain)
readerTest = np.loadtxt(
    #had to get rid of major to be able to use the other values as floats
    open("dataset/test_reghanData_noMajor.csv", "r"), delimiter=",", skiprows=1)
x = list(readerTest)
resultTest = np.array(x).astype("double")
#print('test', '\n', resultTest)
'''for i in range(len(resultTrain)):
        for j in range(len(resultTrain[i])):
            data=[]
'''

def first_array(small_array):
    deep_process = []
    attributes = []
    target = []
    for i in range(len(small_array)):
        if i < 3:
            attributes.append(small_array[i])
        else: 
            target.append(small_array[i]) 
    print("      ")
    deep_process.append(attributes)
    deep_process.append(target)
    
    return deep_process
'''
train = pd.read_csv("dataset/train_reghanData.csv")
test = pd.read_csv("dataset/test_reghanData.csv")
#Combined both Train and Test Data set to do preprocessing together # and set flag for both as well
train['Type'] = 'Train'
'''
small_array = [1,23,43,8]
print(first_array(small_array))

big_data = []
for a in range(len(resultTrain)):
    big_data.append(first_array(resultTrain[a]))    
    
print(big_data)

