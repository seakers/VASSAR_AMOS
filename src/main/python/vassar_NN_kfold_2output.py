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
#archs2 = np.empty([len(archs)-1])
#scores = np.empty([len(archs)-1,2])
#for i in range(len(archs)-1):
    #archs2[i,:] = archs[i+1,:]
    #scores[i,:] = [float(science[i+1]), float(cost[i+1])]

### Defining Neural Net training parameters
batch_size = 128
num_epochs = 100

### Create Neural Net
def create_model():
    print('Building Model....')
    SCModel = Sequential()
    SCModel.add(Dense(100, input_dim=60))
    SCModel.add(Activation('relu'))
    SCModel.add(Dropout(0.3))
    SCModel.add(Dense(50))
    SCModel.add(Activation('relu'))
    SCModel.add(Dropout(0.3))
    SCModel.add(Dense(25))
    SCModel.add(Activation('relu'))
    SCModel.add(Dropout(0.3))
    SCModel.add(Dense(2))

    SCModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return SCModel

### Train-Test split
arch_training, arch_testing, science_training, science_testing, cost_training, cost_testing = train_test_split(archs[1:], science[1:], cost[1:], test_size=0.2)#, random_state=10)

### Preprocessing data for input to Neural Net
science_train_scaler = StandardScaler()
science_test_scaler = StandardScaler()
cost_train_scaler = StandardScaler()
cost_test_scaler = StandardScaler()

archArray_train = np.empty([len(np.array(arch_training)),60])
archArray_test = np.empty([len(np.array(arch_testing)),60])
#scores_train = np.empty([len(np.array(arch_training))])
#scores_test = np.empty([len(np.array(arch_testing))])

for x in range(len(archArray_train)):
    current_arch = archs[x]
    if (current_arch == 'Architecture'):
        continue
    # print(current_arch)
    for y in range(60):
        archArray_train[x][y] = int(current_arch[y])

#print(archArray_train[0,:])
for x in range(len(archArray_test)):
    current_arch = archs[x]
    if (current_arch == 'Architecture'):
        continue
    for y in range(60):
        archArray_test[x][y] = int(current_arch[y])

#science_train = scores_train[1,:]
#cost_train = scores_train[2,:]
#science_test = scores_test[1,:]
#cost_test = scores_test[2,:]
        
science_train_num = list(map(float,science_training))
science_test_num = list(map(float,science_testing))
cost_train_num = list(map(float,cost_training))
cost_test_num = list(map(float,cost_testing))

science_train_scalerfit = science_train_scaler.fit(np.array(science_train_num).reshape(-1,1))
print(science_train_scaler.mean_)
cost_train_scalerfit = cost_train_scaler.fit(np.array(cost_train_num).reshape(-1,1))
print(cost_train_scaler.mean_)
science_test_scalerfit = science_test_scaler.fit(np.array(science_test_num).reshape(-1,1))
print(science_test_scaler.mean_)
cost_test_scalerfit = cost_test_scaler.fit(np.array(cost_test_num).reshape(-1,1))
print(cost_test_scaler.mean_)

#science_train_scaled = science_train_scaler.transform(science_train.reshape(-1,1))
#cost_train_scaled = cost_train_scaler.transform(cost_train.reshape(-1,1))
#science_test_scaled = science_test_scaler.transform(science_test.reshape(-1,1))
#cost_test_scaled = cost_test_scaler.transform(cost_test.reshape(-1,1))

science_train_scaled = science_train_num/np.amax(science_train_num)
cost_train_scaled = cost_train_num/np.amax(cost_train_num)
scores_train_scaled = np.empty([len(science_train_num),2])
for i in range(len(science_training)):
    scores_train_scaled[i,:] = [science_train_scaled[i], cost_train_scaled[i]]

science_test_scaled = science_test_num/np.amax(science_test_num)
cost_test_scaled = cost_test_num/np.amax(cost_test_num)
scores_test_scaled = np.empty([len(science_test_num),2])
for i in range(len(science_testing)):
    scores_test_scaled[i,:] = [science_test_scaled[i], cost_test_scaled[i]]

### Save normalization constants to csv file
print('Saving normalization constants to csv file')
if (ArchSampleType=='MedInstr'):
    with open('normalization_constants_medinstr.csv', mode='w') as instr_num_file:
        instr_num_writer = csv.writer(instr_num_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        instr_num_writer.writerow(['Dataset','Value'])
        instr_num_writer.writerow(['Science Training', str(np.amax(science_train_num))])
        instr_num_writer.writerow(['Cost Training', str(np.amax(cost_train_num))])
        instr_num_writer.writerow(['Science Testing', str(np.amax(science_test_num))])
        instr_num_writer.writerow(['Cost Testing', str(np.amax(cost_test_num))])
elif (ArchSampleType=='LowInstr'):
    with open('normalization_constants_lowinstr.csv', mode='w') as instr_num_file:
        instr_num_writer = csv.writer(instr_num_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        instr_num_writer.writerow(['Dataset','Value'])
        instr_num_writer.writerow(['Science Training', str(np.amax(science_train_num))])
        instr_num_writer.writerow(['Cost Training', str(np.amax(cost_train_num))])
        instr_num_writer.writerow(['Science Testing', str(np.amax(science_test_num))])
        instr_num_writer.writerow(['Cost Testing', str(np.amax(cost_test_num))])
print('Normalization constants saved')

### k-fold cross validation
n_folds = 10
best_model = None
best_modelfold = 0
best_trainscore = 1


### Find best model
SC_Score = np.empty([n_folds,2])
SC_TrainScore = np.empty([n_folds,2])
SC_TestScore = np.empty([n_folds,2])
skf_sc = KFold(n_folds, shuffle=True)
count = 0
for (train_ind, test_ind) in skf_sc.split(archArray_train, scores_train_scaled):
    print("Running Fold ",str(count+1),"/",str(n_folds)," for science")
    arch_train_fold = archArray_train[train_ind,:]
    scores_train_fold = scores_train_scaled[train_ind,:]
    arch_val_fold = archArray_train[test_ind,:]
    scores_val_fold = scores_train_scaled[test_ind,:]
    #print(arch_train_fold.shape)
    #print(science_train_fold.shape)
    #print(arch_val_fold.shape)
    #print(science_val_fold.shape)
    SC_model = None
    SC_model = create_model()
    SC_History = SC_model.fit(arch_train_fold, scores_train_fold, batch_size=batch_size, epochs=num_epochs)
    SC_Score[count,:] = SC_model.evaluate(arch_val_fold, scores_val_fold, batch_size=batch_size)
    SC_TrainScore[count,:] = SC_model.evaluate(arch_train_fold, scores_train_fold, batch_size=batch_size)
    SC_TestScore[count,:] = SC_model.evaluate(archArray_test, scores_test_scaled, batch_size=batch_size)
    #print(Sc_TrainScore[count,:])
    if (all(i < best_trainscore for i in list(SC_TrainScore[count,:]))):
        best_trainscore = np.amin(SC_TrainScore[count,:])
        best_model = SC_model
        best_modelfold = count
    count += 1
    
print('The fold corresponding to the best model is ', str(best_modelfold))
plt.figure(1)
plt.plot(SC_Score[:,1], label='val')
plt.plot(SC_TrainScore[:,1], label='train')
plt.plot(SC_TestScore[:,1], label='test')
plt.xlabel('Fold Number')
plt.ylabel('Mean Squared Error')
plt.title('K-Fold cross validation')
plt.legend(loc='upper right')
plt.show()

# Saving the best science and cost neural nets
if (ArchSampleType=='MedInstr'):
    print('Saving the Model for Binomial type Architectures')
    best_model.save('NNkfold_2op_medInstr.h5')
elif (ArchSampleType=='LowInstr'):
    print('Saving the Model for Poisson type Architectures')
    best_model.save('NNkfold_2op_lowInstr.h5')
