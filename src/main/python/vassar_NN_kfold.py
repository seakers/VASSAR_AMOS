# -*- coding: utf-8 -*-
"""

@author: roshan94
"""

import csv
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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

### Defining Neural Net training parameters
batch_size = 128
num_epochs_science = 100
num_epochs_cost = 100

### Science Neural Net
def create_science_model():
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
    return ScienceModel

### Cost Neural Net
def create_cost_model():
    print('Building Cost Model....')
    CostModel = Sequential()
    #CostModel.add(Dense(1, input_dim=60))
    CostModel.add(Dense(100, input_dim=60))
    CostModel.add(Activation('sigmoid'))
    #CostModel.add(Dropout(0.3))
    CostModel.add(Dense(20))
    CostModel.add(Activation('sigmoid'))
    #CostModel.add(Dropout(0.3))
    CostModel.add(Dense(5))
    CostModel.add(Activation('sigmoid'))
    #CostModel.add(Dropout(0.3))
    CostModel.add(Dense(1))

    CostModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return CostModel

### Train-Test split
arch_training, arch_testing, science_training, science_testing = train_test_split(archs, science, test_size=0.2)#, random_state=21)
arch_training, arch_testing, cost_training, cost_testing = train_test_split(archs, cost, test_size=0.2)#, random_state=20)

#print(np.array(arch_training).shape)
#print(np.array(science_training).shape)
#print(np.array(arch_testing).shape)
#print(np.array(science_testing).shape)
#print(np.array(cost_training).shape)
#print(np.array(cost_testing).shape)

### Preprocessing data for input to Neural Net
science_train_scaler = StandardScaler()
science_test_scaler = StandardScaler()
cost_train_scaler = StandardScaler()
cost_test_scaler = StandardScaler()

archArray_train = np.empty([len(np.array(arch_training)),60])
archArray_test = np.empty([len(np.array(arch_testing)),60])
science_train = np.empty([len(np.array(arch_training))])
cost_train = np.empty([len(np.array(arch_training))])
science_test = np.empty([len(np.array(arch_testing))])
cost_test = np.empty([len(np.array(arch_testing))])
for x in range(len(archArray_train)):
    current_arch = arch_training[x]
    if (current_arch == 'Architecture' or science_training[x] == 'Science Benefit' or cost_training[x] == 'Cost'):
        continue
    # print(current_arch)
    for y in range(60):
        archArray_train[x][y] = int(current_arch[y])
    # print(type(science_train[x]))
    science_train[x] = float(science_training[x])
    cost_train[x] = float(cost_training[x])

#print(archArray_train[0,:])
for x in range(len(arch_testing)):
    current_arch = arch_testing[x]
    if (current_arch == 'Architecture' or science_testing[x] == 'Science Benefit' or cost_testing[x] == 'Cost'):
        continue
    for y in range(60):
        archArray_test[x][y] = int(current_arch[y])
    science_test[x] = float(science_testing[x])
    cost_test[x] = float(cost_testing[x])

science_train_scalerfit = science_train_scaler.fit(science_train.reshape(-1,1))
print(science_train_scaler.mean_)
cost_train_scalerfit = cost_train_scaler.fit(cost_train.reshape(-1,1))
print(cost_train_scaler.mean_)
science_test_scalerfit = science_test_scaler.fit(science_test.reshape(-1,1))
print(science_test_scaler.mean_)
cost_test_scalerfit = cost_test_scaler.fit(cost_test.reshape(-1,1))
print(cost_test_scaler.mean_)

#science_train_scaled = science_train_scaler.transform(science_train.reshape(-1,1))
#cost_train_scaled = cost_train_scaler.transform(cost_train.reshape(-1,1))
#science_test_scaled = science_test_scaler.transform(science_test.reshape(-1,1))
#cost_test_scaled = cost_test_scaler.transform(cost_test.reshape(-1,1))

science_train_scaled = science_train/np.amax(science_train)
cost_train_scaled = cost_train/np.amax(cost_train)
science_test_scaled = science_test/np.amax(science_test)
cost_test_scaled = cost_test/np.amax(cost_test)

### Save normalization constants to csv file
print('Saving normalization constants to csv file')
if (ArchSampleType=='MedInstr'):
    with open('normalization_constants_medinstr.csv', mode='w') as instr_num_file:
        instr_num_writer = csv.writer(instr_num_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        instr_num_writer.writerow(['Dataset','Value'])
        instr_num_writer.writerow(['Science Training', str(np.amax(science_train))])
        instr_num_writer.writerow(['Cost Training', str(np.amax(cost_train))])
        instr_num_writer.writerow(['Science Testing', str(np.amax(science_test))])
        instr_num_writer.writerow(['Cost Testing', str(np.amax(cost_test))])
elif (ArchSampleType=='LowInstr'):
    with open('normalization_constants_lowinstr.csv', mode='w') as instr_num_file:
        instr_num_writer = csv.writer(instr_num_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        instr_num_writer.writerow(['Dataset','Value'])
        instr_num_writer.writerow(['Science Training', str(np.amax(science_train))])
        instr_num_writer.writerow(['Cost Training', str(np.amax(cost_train))])
        instr_num_writer.writerow(['Science Testing', str(np.amax(science_test))])
        instr_num_writer.writerow(['Cost Testing', str(np.amax(cost_test))])
print('Normalizatioin constants saved')

### k-fold cross validation
n_folds = 10
best_science_model = None
best_cost_model = None
best_modelfold_science = 0
best_modelfold_cost = 0
best_trainscore_science = 1
best_trainscore_cost = 1

### For Science
Sc_Score = np.empty([n_folds,2])
Sc_TrainScore = np.empty([n_folds,2])
Sc_TestScore = np.empty([n_folds,2])
skf_sc = KFold(n_folds, shuffle=True)
count = 0
for (train_ind, test_ind) in skf_sc.split(archArray_train, science_train_scaled):
    print("Running Fold ",str(count+1),"/",str(n_folds)," for science")
    arch_train_fold = archArray_train[train_ind]
    science_train_fold = science_train_scaled[train_ind]
    arch_val_fold = archArray_train[test_ind]
    science_val_fold = science_train_scaled[test_ind]
    #print(arch_train_fold.shape)
    #print(science_train_fold.shape)
    #print(arch_val_fold.shape)
    #print(science_val_fold.shape)
    Sc_model = None
    Sc_model = create_science_model()
    Sc_History = Sc_model.fit(arch_train_fold, science_train_fold, batch_size=batch_size, epochs=num_epochs_science)
    Sc_Score[count,:] = Sc_model.evaluate(arch_val_fold, science_val_fold, batch_size=batch_size)
    Sc_TrainScore[count,:] = Sc_model.evaluate(arch_train_fold, science_train_fold, batch_size=batch_size)
    Sc_TestScore[count,:] = Sc_model.evaluate(archArray_test, science_test_scaled, batch_size=batch_size)
    #print(Sc_TrainScore[count,:])
    if (all(i < best_trainscore_science for i in list(Sc_TrainScore[count,:]))):
        best_trainscore_science = np.amin(Sc_TrainScore[count,:])
        best_science_model = Sc_model
        best_modelfold_science = count
    count += 1
    
print('The fold corresponding to the best science model is ', str(best_modelfold_science))
plt.figure(1)
plt.plot(Sc_Score[:,1], label='val')
plt.plot(Sc_TrainScore[:,1], label='train')
plt.plot(Sc_TestScore[:,1], label='test')
plt.xlabel('Fold Number')
plt.ylabel('Mean Squared Error')
plt.title('K-Fold cross validation for science')
plt.legend(loc='upper right')
plt.show()

### For cost
Cost_Score = np.empty([n_folds,2])
Cost_TrainScore = np.empty([n_folds,2])
Cost_TestScore = np.empty([n_folds,2])
skf_cost = KFold(n_folds, shuffle=True)
count = 0
for (train_ind, test_ind) in skf_cost.split(archArray_train, cost_train_scaled):
    print("Running Fold ",str(count+1),"/",str(n_folds)," for cost")
    arch_train_fold = archArray_train[train_ind]
    cost_train_fold = cost_train_scaled[train_ind]
    arch_val_fold = archArray_train[test_ind]
    cost_val_fold = cost_train_scaled[test_ind]
    #print(arch_train_fold.shape)
    #print(cost_train_fold.shape)
    #print(arch_test_fold.shape)
    #print(cost_test_fold.shape)
    Cost_model = None
    Cost_model = create_cost_model()
    Cost_History = Cost_model.fit(arch_train_fold, cost_train_fold, batch_size=batch_size, epochs=num_epochs_cost)
    Cost_Score[count,:] = Cost_model.evaluate(arch_val_fold, cost_val_fold, batch_size=batch_size)
    Cost_TrainScore[count,:] = Cost_model.evaluate(arch_train_fold, cost_train_fold, batch_size=batch_size)
    Cost_TestScore[count,:] =  Cost_model.evaluate(archArray_test, cost_test_scaled, batch_size=batch_size)
    if (all(i < best_trainscore_cost for i in list(Cost_TrainScore[count,:]))):
        best_trainscore_cost = np.amin(Cost_TrainScore[count,:])
        best_cost_model = Cost_model
        best_modelfold_cost = count
    count += 1

print('The fold corresponding to the best cost model is ', str(best_modelfold_cost))
plt.figure(2)
plt.plot(Cost_Score[:,1], label='val')
plt.plot(Cost_TrainScore[:,1], label='train')
plt.plot(Cost_TestScore[:,1], label='test')
plt.xlabel('Fold Number')
plt.ylabel('Mean Squared Error')
plt.title('K-Fold cross validation for cost')
plt.legend(loc='upper right')
plt.show()

# Saving the best science and cost neural nets
if (ArchSampleType=='MedInstr'):
    print('Saving the Science Model for Binomial type Architectures')
    best_science_model.save('Science_NNkfold_medInstr.h5')
    print('Saving the Cost Model for Binomial type Architectures')
    best_cost_model.save('Cost_NNkfold_medInstr.h5')
elif (ArchSampleType=='LowInstr'):
    print('Saving the Science Model for Poisson type Architectures')
    best_science_model.save('Science_NNkfold_lowInstr.h5')
    print('Saving the Cost Model for Poisson type Architectures')
    best_cost_model.save('Cost_NNkfold_lowInstr.h5') 