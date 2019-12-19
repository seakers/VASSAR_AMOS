# -*- coding: utf-8 -*-
"""

@author: roshan94
"""
import numpy as np
import csv
import matplotlib.pyplot as plt

### Set which Neural Net to train
### 'LessInstruments' - trains the Neural Net pertaining to lower instrument count architectures
### 'Uniform' - trains the Neural Net pertaining to uniformly distributed instument architectures
ArchSampleType = 'LowInstr'

def read_csv(ArchSampleType):
    if (ArchSampleType=='Uniform'):
        ### Read data from vassar_data_uniform_10000.csv and store into different arrays 
        with open('C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\vassar_data_uniform_10000.csv',newline='')as csvfile:
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

### Counting instruments in the architectures
archArray = np.empty([len(archs),60])
science_val = np.zeros([len(archs)])
cost_val = np.zeros([len(archs)])
instr_count = np.zeros(len(archs))
for x in range(1,len(archs)):
    current_arch = archs[x]
    # print(current_arch)
    science_val[x] = float(science[x])
    cost_val[x] = float(cost[x])
    for y in range(60):
        archArray[x][y] = int(current_arch[y])
        if archArray[x][y] == 1:
            instr_count[x] += 1
    # print(type(science_train[x]))
   
archs_vec = np.linspace(1,len(archs),len(archs))
plt.figure(1)
plt.plot(archs_vec,instr_count)
plt.xlabel('Architecture number')
plt.ylabel('Instrument count')
plt.title('Instrument count for each architecture')

science_sorted = sorted(science_val, key=float)
plt.figure(2)
plt.plot(archs_vec,science_sorted)
plt.xlabel('Number')
plt.ylabel('Science')
plt.title('Science sorted in ascending order for each architecture')

cost_sorted = sorted(cost_val, key=float)
plt.figure(3)
plt.plot(archs_vec,cost_sorted)
plt.xlabel('Number')
plt.ylabel('Cost')
plt.title('Cost sorted in ascending order for each architecture')
