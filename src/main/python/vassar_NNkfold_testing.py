# -*- coding: utf-8 -*-
"""

@author: roshan94
"""
import csv
import os.path
from keras.models import load_model
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

### Set type of data to test
ArchSampletype = 'MedInstr'

def get_test_files(ArchSampleType):
    if(ArchSampleType=='MedInstr'):
        ### Loading the trained Science and Cost Models from the h5 files
        SModel = load_model('.\\NN_kfold_binomial2\\Science_NNkfold_medInstr.h5')
        CModel = load_model('.\\NN_kfold_binomial2\\Cost_NNkfold_medInstr.h5')

        ### Finding number of csv data files for testing
        dir_path = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\NN test data\\Incorrect Datasets\\Uniform\\'
        n_testfiles = len([f for f in os.listdir(dir_path)if os.path.isfile(os.path.join(dir_path,f))])
        #print(num_testfiles)

        file_loc = ['' for x in range(n_testfiles)]
        for i in range(n_testfiles):
            file_loc[i] = dir_path + 'vassar_data_uniform_test' + str(i+1) + '.csv' 
            #print(file_path)
            
        norm_file_loc = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\src\\main\\python\\NN_kfold_binomial2\\normalization_constants_medinstr.csv'
        
    elif(ArchSampleType=='LowInstr'):
        ### Loading the trained Science and Cost Models from the h5 files
        SModel = load_model('Science_NN_lowInstr.h5')
        CModel = load_model('Cost_NN_lowInstr.h5')

        ### Finding number of csv data files for testing
        dir_path = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\NN test data\\Incorrect Datasets\\LessArchs\\'
        n_testfiles = len([f for f in os.listdir(dir_path)if os.path.isfile(os.path.join(dir_path,f))])
        #print(num_testfiles)

        file_loc = ['' for x in range(n_testfiles)]
        for i in range(n_testfiles):
            file_loc[i] = dir_path + 'vassar_data_lessarchs_test' + str(i+1) + '.csv' 
            #print(file_path)
            
        norm_file_loc = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\src\\main\\python\\NN_kfold_poisson1\\normalization_constants_lowinstr.csv'
        
    return SModel, CModel, n_testfiles, file_loc, norm_file_loc

ScienceModel, CostModel, num_testfiles, file_path, norm_file_path = get_test_files(ArchSampletype)

### Read data from csv file and store into different arrays 
def read_csv(file_path):
    with open(file_path,newline='') as csvfile:
        data = [row for row in csv.reader(csvfile)]
        archs = ["" for x in range(len(data)-1)]
        science = ["" for x in range(len(data)-1)]
        cost = ["" for x in range(len(data)-1)]
        for x in range(len(data)-1):
            archs[x] = data[x+1][0]
            science[x] = data[x+1][1]
            cost[x] = data[x+1][2]
    return [archs, science, cost]

### Read normalization constants from csv file
def read_norm_constants(norm_filepath):
    with open(norm_filepath,newline='' )as csvfile:
        data = [row for row in csv.reader(csvfile)]
        norm_constants = ["" for x in range(len(data)-1)]
        for x in range(len(data)-1):
            norm_constants[x] = float(data[x+1][1])
    return norm_constants # norm_constants = [science_training, cost_training, science_testing, cost_testing]

norm_const = read_norm_constants(norm_file_path)

### Use normalization constants on output of the neural networks
def denormalize(normalization_constants, science_norm_vals, cost_norm_vals):
    science_train_norm_const = normalization_constants[0]
    cost_train_norm_const = normalization_constants[1]
    science_denorm_vals = science_norm_vals*science_train_norm_const
    cost_denorm_vals = cost_norm_vals*cost_train_norm_const
    return science_denorm_vals, cost_denorm_vals

### Defining Neural Net evaluation batch size
n_batch = 128

def NN_evaluation(archs, science, cost, num_batch):
    ### Converting archs, science and cost to arrays to be input to the Neural Net
    n_archs = len(archs)
    archs_array = np.empty([n_archs ,60])
    science_array = np.empty([n_archs])
    cost_array = np.empty([n_archs])
    for x in range(n_archs):
        current_arch = archs[x]
        for y in range(60):
            archs_array[x][y] = int(current_arch[y])
        science_array[x] = float(science[x])
        cost_array[x] = float(cost[x])
    ### Evaluate the Neural Net using available data  
    science_metrics = ScienceModel.evaluate(archs_array, science_array, batch_size=num_batch, verbose=1)
    cost_metrics = CostModel.evaluate(archs_array, cost_array, batch_size=num_batch, verbose=1)
    return [science_metrics, cost_metrics]

def NN_prediction(archs, num_batch):
    ### Converting archs, science and cost to arrays to be input to the Neural Net
    n_archs = len(archs)
    archs_array = np.empty([n_archs ,60])
    for x in range(n_archs):
        current_arch = archs[x]
        for y in range(60):
            archs_array[x][y] = int(current_arch[y])
    ### Get the Neural Net forward pass predictions for the architectures
    sc_pred = ScienceModel.predict(archs_array, batch_size=num_batch)
    c_pred = CostModel.predict(archs_array, batch_size=num_batch)
    return sc_pred, c_pred

def normalize(vec):
    vec_int = list(map(float, vec))
    val_max = np.amax(vec_int)
    vec_norm = vec_int/val_max
    return vec_norm

metrics = []
science_met = [[0 for i in range(2)] for j in range(num_testfiles)] 
cost_met = [[0 for i in range(2)] for j in range(num_testfiles)]
#science_ref = [[] for i in range(num_testfiles)]
#cost_ref = [[] for i in range(num_testfiles)]
#science_pred = [[] for i in range(num_testfiles)]
#cost_pred = [[] for i in range(num_testfiles)]
archs_list = []
science_ref_norm = []
science_ref = []
science_pred_norm = []
science_pred = []
cost_ref_norm = []
cost_ref = []
cost_pred_norm = []
cost_pred = []
num_data = np.empty([num_testfiles])
for i in range(num_testfiles):
    file = file_path[i]
    data = read_csv(file)
    archs_eval = data[0]
    science_eval = data[1]
    cost_eval = data[2]
    num_data[i] = len(archs_eval)
    science_eval_norm = normalize(science_eval)
    cost_eval_norm = normalize(cost_eval)
    science_ref.extend(science_eval)
    cost_ref.extend(cost_eval)
    archs_list.extend(archs_eval)
    science_ref_norm.extend(science_eval_norm)
    cost_ref_norm.extend(cost_eval_norm)
    #science_pred_file = np.zeros([int(num_data[i])])
    #cost_pred_file = np.zeros([int(num_data[i])])
    science_pred_file = []
    cost_pred_file = []
    metrics = NN_evaluation(archs_eval, science_eval, cost_eval, n_batch)
    science_met[i] = metrics[0]
    cost_met[i] = metrics[1]
    print('For data from test file ' + str(i+1) + ' ,science metric = ' + str(science_met[i]) + ' and cost metric = ' + str(cost_met[i]))
    science_pred_file, cost_pred_file = NN_prediction(archs_eval, n_batch)
    science_pred_norm.extend(science_pred_file)
    cost_pred_norm.extend(cost_pred_file)
    science_pred_denorm, cost_pred_denorm = denormalize(norm_const, science_pred_file, cost_pred_file)
    science_pred.extend(science_pred_denorm)
    cost_pred.extend(cost_pred_denorm)

#print(science_pred[0])
#print(cost_pred[0])
print('The evaluation metrics shown are ' + str(ScienceModel.metrics_names) + ' for the Science Neural Net and ' + str(CostModel.metrics_names) + ' for the Cost Neural Net.')
num_archs_total = np.sum(num_data)

### Write to csv file 
print('Writing to csv file')
if (ArchSampletype=='MedInstr'):
    line = 0
    with open('bestNN_kfold_predictions_medinstr.csv', mode='w') as instr_num_file:
        instr_num_writer = csv.writer(instr_num_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        instr_num_writer.writerow(['Architecture','Reference Science','Predicted Science','Normalized Reference Science','Normalized Predicted Science','Reference Cost','Predicted Cost','Normalized Reference Cost','Normalized Predicted Cost'])
        while line < num_archs_total:
            instr_num_writer.writerow([archs_list[line], str(science_ref[line]), str(science_pred[line][0]), str(science_ref_norm[line]), str(science_pred_norm[line][0]), str(cost_ref[line]), str(cost_pred[line][0]), str(cost_ref_norm[line]), str(cost_pred_norm[line][0])])
            line += 1
elif (ArchSampletype=='LowInstr'):
    line = 0
    with open('bestNN_kfold_predictions_lowinstr.csv', mode='w') as instr_num_file:
        instr_num_writer = csv.writer(instr_num_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        instr_num_writer.writerow(['Architecture','Reference Science','Predicted Science','Normalized Reference Science','Normalized Predicted Science','Reference Cost','Predicted Cost','Normalized Reference Cost','Normalized Predicted Cost'])
        while line < num_archs_total:
            instr_num_writer.writerow([archs_list[line], str(science_ref[line]), str(science_pred[line][0]), str(science_ref_norm[line]), str(science_pred_norm[line][0]), str(cost_ref[line]), str(cost_pred[line][0]), str(cost_ref_norm[line]), str(cost_pred_norm[line][0])])
            line += 1
print('File Writing done')

### LINEAR REGRESSION ANALYSIS
#regression_science = linear_model.LinearRegression()
#regression_cost = linear_model.LinearRegression()

#sc_ref_full = []
#cost_ref_full = []
#sc_pred_full = []
#cost_pred_full = []
#index = 0

#for i in range(num_testfiles):
    #num_data_file = int(num_data[i])
    ##print(type(num_data_file))
    #sc_ref_full[index+1:index+num_data_file] = science_ref[i] 
    #cost_ref_full[index+1:index+num_data_file] = cost_ref[i]
    #sc_pred_full[index+1:index+num_data_file] = science_pred[i]
    #cost_pred_full[index+1:index+num_data_file] = cost_pred[i]
    #index = index + num_data_file

#sc_ref_full = np.array(sc_ref_full).astype('float64')
#cost_ref_full = np.array(cost_ref_full).astype('float64')
#sc_pred_full = np.array(sc_pred_full).astype('float64')
#cost_pred_full = np.array(cost_pred_full).astype('float64')

#regression_science.fit(sc_ref_full.reshape(-1,1),sc_pred_full.reshape(-1,1))
#regression_cost.fit(cost_ref_full.reshape(-1,1),cost_pred_full.reshape(-1,1))

##print(regression_science.coef_,regression_science.intercept_)
#sc_pred_full_lin = []
#cost_pred_full_lin = []

#sc_pred_full_lin = regression_science.predict(sc_ref_full.reshape(-1,1))
#cost_pred_full_lin = regression_cost.predict(cost_ref_full.reshape(-1,1))

### The Regression coefficients
#print('Science Linear Regression Coefficient: ', regression_science.coef_, 'and Intercept: ', regression_science.intercept_)
#print('Cost Linear Regression Coefficient: ', regression_cost.coef_, 'and Intercept: ', regression_cost.intercept_)

### The mean squared error
#print("Mean squared error for Science Linear Regression: %.2f"
      #% mean_squared_error(sc_pred_full, sc_pred_full_lin))
#print("Mean squared error for Cost Linear Regression: %.2f"
      #% mean_squared_error(cost_pred_full, cost_pred_full_lin))

### Explained variance score: 1 is perfect prediction
#print('Variance score for Science Linear Regression: %.2f' % r2_score(sc_pred_full, sc_pred_full_lin))
#print('Variance score for Cost Linear Regression: %.2f' % r2_score(cost_pred_full, cost_pred_full_lin))

### Plot outputs
#plt.figure(1)
#plt.scatter(sc_ref_full, sc_pred_full,  color='black', label='True Value')
#plt.plot(sc_ref_full, sc_pred_full_lin, color='blue', linewidth=3, label='Linear Regression Prediction')
#plt.xlabel('True Science Value')
#plt.ylabel('Predicted Science Value')
#plt.title('Science Linear Regression Analysis')
#plt.legend(loc='upper left')
#plt.show()

#plt.figure(2)
#plt.scatter(cost_ref_full, cost_pred_full,  color='black', label='True Value')
#plt.plot(cost_ref_full, cost_pred_full_lin, color='blue', linewidth=3, label='Linear Regression Prediction')
#plt.xlabel('True Cost Value')
#plt.ylabel('Predicted Cost Value')
#plt.title('Cost Linear Regression Analysis')
#plt.legend(loc='upper left')
#plt.show()
