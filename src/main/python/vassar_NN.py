# -*- coding: utf-8 -*-
"""

@author: roshan94
"""

import csv
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt

### Set which Neural Net to train
### 'LowInstr' - trains the Neural Net pertaining to lower instrument count architectures
### 'MedInstr' - trains the Neural Net pertaining to uniformly distributed instument architectures
ArchSampleType = 'MedInstr'

def read_csv(ArchSampleType):
    if (ArchSampleType=='MedInstr'):
        ### Read data from vassar_data_uniform_10000.csv and store into different arrays 
        with open('C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\vassar_data_medinstr_train.csv',newline='')as csvfile:
            data = [row for row in csv.reader(csvfile)]
            architectures = ["" for x in range(len(data))]
            science_vals = ["" for x in range(len(data))]
            cost_vals = ["" for x in range(len(data))]
            for x in range(len(data)):
                architectures[x] = data[x][0]
                science_vals[x] = data[x][1]
                cost_vals[x] = data[x][2]
    elif (ArchSampleType=='LowInstr'):
        ### Read data from vassar_data_lessarchs_10000.csv and store into different arrays 
        with open('C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\vassar_data_lowinstr_train.csv',newline='')as csvfile:
            data = [row for row in csv.reader(csvfile)]
            architectures = ["" for x in range(len(data))]
            science_vals = ["" for x in range(len(data))]
            cost_vals = ["" for x in range(len(data))]
            for x in range(len(data)):
                architectures[x] = data[x][0]
                science_vals[x] = data[x][1]
                cost_vals[x] = data[x][2]
    return architectures, science_vals, cost_vals
                
archs, science, cost = read_csv(ArchSampleType)

### Print 100-th row for sanity check
# print(archs[100], " = ", science[100], " and ", cost[100])

### Defining Neural Net training parameters
batch_size = 128
num_epochs_science = 50
num_epochs_cost = 100

### Separate data into training and testing datasets        
num_data = len(archs)-1
archs_train = archs[1:round(num_data*0.8)]
science_train = science[1:round(num_data*0.8)]
cost_train = cost[1:round(num_data*0.8)]

archs_test = archs[(round(num_data*0.8)+1):]
science_test = science[(round(num_data*0.8)+1):]
cost_test = cost[(round(num_data*0.8)+1):]

print(len(archs_train), 'train sequences')
print(len(archs_test), 'test sequences')

# print(archs_train[1])

### Preprocessing data for input to Neural Net
archArray_train = np.empty([len(archs_train),60])
archArray_test = np.empty([len(archs_test),60])
for x in range(len(archs_train)):
    current_arch = archs_train[x]
    # print(current_arch)
    for y in range(60):
        archArray_train[x][y] = int(current_arch[y])
    # print(type(science_train[x]))
    science_train[x] = float(science_train[x])
    cost_train[x] = float(cost_train[x])

#print(archArray_train[0,:])
for x in range(len(archs_test)):
    current_arch = archs_test[x]
    for y in range(60):
        archArray_test[x][y] = int(current_arch[y])
    science_test[x] = float(science_test[x])
    cost_test[x] = float(cost_test[x])


### Training and testing the Science Neural Net
print('Building Science Model....')
ScienceModel = Sequential()
ScienceModel.add(Dense(100, input_dim=60))
ScienceModel.add(Activation('relu'))
ScienceModel.add(Dropout(0.3))
ScienceModel.add(Dense(50))
ScienceModel.add(Activation('relu'))
ScienceModel.add(Dropout(0.3))
ScienceModel.add(Dense(25))
ScienceModel.add(Activation('relu'))
ScienceModel.add(Dropout(0.3))
ScienceModel.add(Dense(1))

ScienceModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

ScienceHistory = ScienceModel.fit(archArray_train, science_train, batch_size=batch_size, epochs=num_epochs_science, validation_split=0.1)

ScienceScore = ScienceModel.evaluate(archArray_test, science_test, batch_size=batch_size)

ScienceTrainScore = ScienceModel.evaluate(archArray_train, science_train, batch_size=batch_size)

plt.figure(1)
plt.plot(ScienceHistory.history['mse'], label='train')
plt.plot(ScienceHistory.history['val_mse'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error for Science Neural Net Training')
plt.legend(loc='upper right')
plt.show()

print('Science Model Test MSE:', ScienceScore[1])

print('Science Model Test on Train MSE:', ScienceTrainScore[1])

### Saving the trained Science Neural Net 
if (ArchSampleType=='MedInstr'):
    print('Saving the Science Model for Uniform type Architectures')
    ScienceModel.save('Science_NN_uniform.h5')
elif (ArchSampleType=='LowInstr'):
    print('Saving the Science Model for LessInstruments type Architectures')
    ScienceModel.save('Science_NN_lessinstruments.h5')

### Training and testing the Cost Neural Net
print('Building Cost Model....')
CostModel = Sequential()
CostModel.add(Dense(100, input_dim=60))
CostModel.add(Activation('relu'))
CostModel.add(Dropout(0.3))
CostModel.add(Dense(75))
CostModel.add(Activation('relu'))
CostModel.add(Dropout(0.3))
CostModel.add(Dense(50))
CostModel.add(Activation('relu'))
CostModel.add(Dropout(0.3))
CostModel.add(Dense(1))

CostModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

CostHistory = CostModel.fit(archArray_train, cost_train, batch_size=batch_size, epochs=num_epochs_cost, validation_split=0.1)

CostScore = CostModel.evaluate(archArray_test, cost_test, batch_size=batch_size)

CostTrainScore = CostModel.evaluate(archArray_train, cost_train, batch_size=batch_size)

plt.figure(2)
plt.plot(CostHistory.history['mse'], label='train')
plt.plot(CostHistory.history['val_mse'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error for Cost Neural Net Training')
plt.legend(loc='upper right')
plt.show()

print('Cost Model Test MSE:', CostScore[1])

print('Cost Model Test on Train MSE:', CostTrainScore[1])

### Saving the trained Cost Neural Net 
#if (ArchSampleType=='MedInstr'):
    #print('Saving the Cost Model for Uniform type Architectures')
    #CostModel.save('Cost_NN_medInstr.h5')
#elif (ArchSampleType=='LowInstr'):
    #print('Saving the Cost Model for LessInstruments type Architectures')
    #CostModel.save('Cost_NN_lowInstr.h5')