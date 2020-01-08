# -*- coding: utf-8 -*-
"""

@author: roshan94
"""
import csv 
import numpy as np
import matplotlib.pyplot as plt

### Read csv file
ArchSampletype = 'MedInstr_2op'

if (ArchSampletype == 'MedInstr'):
    csv_location = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\src\\main\\python\\NN_kfold_binomial2\\bestNN_kfold_predictions_medinstr.csv' 
elif (ArchSampletype == 'LowInstr'):
    csv_location = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\src\\main\\python\\NN_kfold_binomial2\\bestNN_kfold_predictions_medinstr.csv' 
elif (ArchSampletype == 'MedInstr_2op'):
    csv_location = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\src\\main\\python\\NN_kfold_binomial2_2op\\bestNN_kfold_predictions_2op_medinstr.csv' 
elif (ArchSampletype == 'LowInstr_2op'):
    csv_location = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\src\\main\\python\\NN_kfold_binomial2_2op\\bestNN_kfold_predictions_2op_medinstr.csv' 
   
with open(csv_location,newline='')as csvfile:
           data = [row for row in csv.reader(csvfile)]
           architectures = ["" for x in range(len(data))]
           true_science_vals = ["" for x in range(len(data))]
           pred_science_vals = ["" for x in range(len(data))] 
           true_cost_vals = ["" for x in range(len(data))]
           pred_cost_vals = ["" for x in range(len(data))] 
           for x in range(len(data)):
               architectures[x] = data[x][0]
               true_science_vals[x] = data[x][1]
               pred_science_vals[x] = data[x][2]
               true_cost_vals[x] = data[x][5]
               pred_cost_vals[x] = data[x][6]
               
### Calculate error for each architecture 
true_science_floats = np.empty([len(data)-1])
pred_science_floats = np.empty([len(data)-1])
true_cost_floats = np.empty([len(data)-1])
pred_cost_floats = np.empty([len(data)-1])
for x in range(len(data)-1):
    true_science_floats[x] = float(true_science_vals[x+1])
    pred_science_floats[x] = float(pred_science_vals[x+1])
    true_cost_floats[x] = float(true_cost_vals[x+1])
    pred_cost_floats[x] = float(pred_cost_vals[x+1])
    
archArray = np.empty([len(true_science_floats),60])
instrument_count = np.empty([len(true_science_floats)])
science_abs_error = np.empty([len(true_science_floats)])
cost_abs_error = np.empty([len(true_science_floats)])
for x in range(len(true_science_floats)):
    current_arch = architectures[x+1]
    instr_count = 0
    for y in range(60):
        archArray[x][y] = int(current_arch[y])
        if (int(current_arch[y]) == 1):
            instr_count += 1
    instrument_count[x] = instr_count
    science_abs_error[x] = (abs(true_science_floats[x] - pred_science_floats[x])/true_science_floats[x])*100  
    cost_abs_error[x] = (abs(true_cost_floats[x] - pred_cost_floats[x])/true_cost_floats[x])*100  

### Plot values
plt.figure(1)
plt.scatter(instrument_count, science_abs_error)
plt.xlabel('Instrument count')
plt.ylabel('% Error')
plt.title('Percentage Error for Science Model')
plt.show()
plt.figure(2)
plt.scatter(instrument_count, cost_abs_error)
plt.xlabel('Instrument count')
plt.ylabel('% Error')
plt.title('Percentage Error for Cost Model')
plt.show()

### Find number of architectures with science and cost error less than a threshold
percent_threshold = 20
science_arch_count = 0
cost_arch_count = 0
science_err_archs = []
cost_err_archs = []
for i in range(len(true_science_floats)):
    archArray_int = map(int, archArray[i])
    if (science_abs_error[i] <= percent_threshold):
        science_arch_count += 1
        science_err_archs.append(list(archArray_int))
    if (cost_abs_error[i] <= percent_threshold):
        cost_arch_count += 1
        cost_err_archs.append(list(archArray_int))
        
print(str(science_arch_count) + ' architectures out of ' + str(len(true_science_floats)) + ' have science error percentage less than or equal to ' + str(percent_threshold) + ' %')
print(str(cost_arch_count) + ' architectures out of ' + str(len(true_science_floats)) + ' have cost error percentage less than or equal to ' + str(percent_threshold) + ' %')

### Historgram of science errors
plt.figure(3)
plt.hist(science_abs_error, bins=np.arange(min(science_abs_error),max(science_abs_error),5))
plt.xlabel('Science %')
plt.ylabel('Count')
plt.title('Science Error Histogram')
plt.show()
plt.figure(4)
plt.hist(cost_abs_error, bins=np.arange(min(cost_abs_error),max(cost_abs_error),5))
plt.xlabel('Cost %')
plt.ylabel('Count')
plt.title('Cost Error Histogram')
plt.show()