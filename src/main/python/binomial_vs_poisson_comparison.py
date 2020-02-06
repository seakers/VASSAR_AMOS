# -*- coding: utf-8 -*-
"""

@author: roshan94
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import os.path
from keras.models import load_model

SModel_med = load_model('.\\NN_kfold_binomial2\\Science_NNkfold_medInstr.h5')
CModel_med = load_model('.\\NN_kfold_binomial2\\Cost_NNkfold_medInstr.h5')

norm_filepath_med = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\src\\main\\python\\NN_kfold_binomial3_sc5\\normalization_constants_medinstr.csv'
        
SModel_low = load_model('.\\NN_kfold_poisson1\\Science_NNkfold_lowInstr.h5')
CModel_low = load_model('.\\NN_kfold_poisson1\\Cost_NNkfold_lowInstr.h5')
            
norm_filepath_low = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\src\\main\\python\\NN_kfold_poisson1\\normalization_constants_lowinstr.csv'

### Finding and storing csv data file locations for testing
dir_path_uniform = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\NN test data\\Incorrect Datasets\\Uniform\\'
n_testfiles_uniform = len([f for f in os.listdir(dir_path_uniform)if os.path.isfile(os.path.join(dir_path_uniform,f))])
#print(num_testfiles_uniform)

file_loc_uniform = ['' for x in range(n_testfiles_uniform)]
for i in range(n_testfiles_uniform):
    file_loc_uniform[i] = dir_path_uniform + 'vassar_data_uniform_test' + str(i+1) + '.csv' 
    #print(file_loc_uniform[i])

dir_path_lessarchs = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\NN test data\\Incorrect Datasets\\LessArchs\\'
n_testfiles_lessarchs = len([f for f in os.listdir(dir_path_lessarchs)if os.path.isfile(os.path.join(dir_path_lessarchs,f))])
#print(num_testfiles)

file_loc_lessarchs = ['' for x in range(n_testfiles_lessarchs)]
for i in range(n_testfiles_lessarchs):
    file_loc_lessarchs[i] = dir_path_lessarchs + 'vassar_data_lessarchs_test' + str(i+1) + '.csv' 
    #print(file_loc_lessarchs[i])
    
n_testfiles_all = n_testfiles_uniform + n_testfiles_lessarchs
file_path_all = file_loc_uniform + file_loc_lessarchs
    
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

norm_const_med = read_norm_constants(norm_filepath_med)
norm_const_low = read_norm_constants(norm_filepath_low)

def normalize(vec):
    vec_int = list(map(float, vec))
    val_max = np.amax(vec_int)
    vec_norm = vec_int/val_max
    return vec_norm

### Use normalization constants on output of the neural networks
def denormalize(normalization_constants, science_norm_vals, cost_norm_vals):
    science_train_norm_const = normalization_constants[0]
    cost_train_norm_const = normalization_constants[1]
    science_denorm_vals = science_norm_vals*science_train_norm_const
    cost_denorm_vals = cost_norm_vals*cost_train_norm_const
    return science_denorm_vals, cost_denorm_vals

### Defining Neural Net evaluation batch size
n_batch = 128

def NN_prediction(ScienceModel, CostModel, archs, num_batch):
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

def get_scores_from_files (n_files, file_paths):
    archs_list = []
    science_ref_norm = []
    science_ref = []
    cost_ref_norm = []
    cost_ref = []
    num_data = np.empty([n_files])
    for i in range(n_files):
        file = file_paths[i]
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
    return archs_list, science_ref, cost_ref, science_ref_norm, cost_ref_norm 

def get_instr_count (arch):
    n_instr = 0
    for bit in arch:
        if (int(bit) == 1):
            n_instr += 1
    return n_instr

### Categorize architectures into low instrument and high instrument architectures
archs_medinstr, science_ref_medinstr, cost_ref_medinstr, science_ref_norm_medinstr, cost_ref_norm_medinstr = get_scores_from_files(n_testfiles_uniform, file_loc_uniform)
archs_lowinstr, science_ref_lowinstr, cost_ref_lowinstr, science_ref_norm_lowinstr, cost_ref_norm_lowinstr = get_scores_from_files(n_testfiles_lessarchs, file_loc_lessarchs)

n_archs_medinstr = len(archs_medinstr)
n_archs_lowinstr = len(archs_lowinstr)

archs_more = []
archs_less = []
archs_instrcount_more = []
archs_instrcount_less = []

science_ref_more = []
cost_ref_more = []
science_ref_norm_more = []
cost_ref_norm_more = []

science_ref_less = []
cost_ref_less = []
science_ref_norm_less = []
cost_ref_norm_less = []

instrcount_thresh = 18
for i in range(n_archs_medinstr):
    current_arch = archs_medinstr[i]
    science_ref_arch = science_ref_medinstr[i]
    cost_ref_arch = cost_ref_medinstr[i]
    science_ref_norm_arch = science_ref_norm_medinstr[i]
    cost_ref_norm_arch = cost_ref_norm_medinstr[i]
    instr_count = get_instr_count(current_arch)
    if (instr_count <= instrcount_thresh):
        archs_less.append(current_arch)
        archs_instrcount_less.append(instr_count)
        science_ref_less.append(science_ref_arch)
        cost_ref_less.append(cost_ref_arch)
        science_ref_norm_less.append(science_ref_norm_arch)
        cost_ref_norm_less.append(cost_ref_norm_arch)
    elif (instr_count > instrcount_thresh):
        archs_more.append(current_arch)
        archs_instrcount_more.append(instr_count)
        science_ref_more.append(science_ref_arch)
        cost_ref_more.append(cost_ref_arch)
        science_ref_norm_more.append(science_ref_norm_arch)
        cost_ref_norm_more.append(cost_ref_norm_arch)
        
for i in range(n_archs_lowinstr):
    current_arch = archs_lowinstr[i]
    science_ref_arch = science_ref_lowinstr[i]
    cost_ref_arch = cost_ref_lowinstr[i]
    science_ref_norm_arch = science_ref_norm_lowinstr[i]
    cost_ref_norm_arch = cost_ref_norm_lowinstr[i]
    instr_count = get_instr_count(current_arch)
    if (instr_count <= instrcount_thresh):
        archs_less.append(current_arch)
        archs_instrcount_less.append(instr_count)
        science_ref_less.append(science_ref_arch)
        cost_ref_less.append(cost_ref_arch)
        science_ref_norm_less.append(science_ref_norm_arch)
        cost_ref_norm_less.append(cost_ref_norm_arch)
    elif (instr_count > instrcount_thresh):
        archs_more.append(current_arch)
        archs_instrcount_more.append(instr_count)
        science_ref_more.append(science_ref_arch)
        cost_ref_more.append(cost_ref_arch)
        science_ref_norm_more.append(science_ref_norm_arch)
        cost_ref_norm_more.append(cost_ref_norm_arch)

### Remove duplicates
def remove_duplicates(archs_list, science_ref, cost_ref, science_ref_norm, cost_ref_norm):
    print(len(archs_list))
    print(len(science_ref_norm))
    print(len(cost_ref_norm))
    archs_list_unique = list(set(archs_list))
    science_ref_unique = []
    cost_ref_unique = []
    science_ref_norm_unique = []
    cost_ref_norm_unique = []
    n_instr_unique = []
    for i in range(len(archs_list_unique)):
        index = archs_list.index(archs_list_unique[i])
        #print(index)
        science_ref_unique.append(science_ref[index])
        cost_ref_unique.append(cost_ref[index])
        science_ref_norm_unique.append(science_ref_norm[index])
        cost_ref_norm_unique.append(cost_ref_norm[index])
        n_instr_unique.append(get_instr_count(archs_list_unique[i]))
    print(len(archs_list_unique))
    print(len(science_ref_norm_unique))
    print(len(cost_ref_norm_unique))
    return archs_list_unique, science_ref_unique, cost_ref_unique, science_ref_norm_unique, cost_ref_norm_unique, n_instr_unique

archs_unique_more, science_ref_unique_more, cost_ref_unique_more, science_ref_norm_unique_more, cost_ref_norm_unique_more, instr_count_unique_more = remove_duplicates(archs_more, science_ref_more, cost_ref_more, science_ref_norm_more, cost_ref_norm_more)
archs_unique_less, science_ref_unique_less, cost_ref_unique_less, science_ref_norm_unique_less, cost_ref_norm_unique_less, instr_count_unique_less = remove_duplicates(archs_less, science_ref_less, cost_ref_less, science_ref_norm_less, cost_ref_norm_less)

### Compute the prediction using binomial and poisoon Neural Nets
science_pred_norm_more_bin, cost_pred_norm_more_bin = NN_prediction(SModel_med, CModel_med, archs_unique_more, n_batch)
science_pred_norm_more_pois, cost_pred_norm_more_pois = NN_prediction(SModel_low, CModel_low, archs_unique_more, n_batch)

science_pred_norm_less_bin, cost_pred_norm_less_bin = NN_prediction(SModel_med, CModel_med, archs_unique_less, n_batch)
science_pred_norm_less_pois, cost_pred_norm_less_pois = NN_prediction(SModel_low, CModel_low, archs_unique_less, n_batch)

science_pred_more_bin, cost_pred_more_bin = denormalize(norm_const_med, science_pred_norm_more_bin, cost_pred_norm_more_bin)
science_pred_more_pois, cost_pred_more_pois = denormalize(norm_const_low, science_pred_norm_more_pois, cost_pred_norm_more_pois)

science_pred_less_bin, cost_pred_less_bin = denormalize(norm_const_med, science_pred_norm_less_bin, cost_pred_norm_less_bin)
science_pred_less_pois, cost_pred_less_pois = denormalize(norm_const_low, science_pred_norm_less_pois, cost_pred_norm_less_pois)

### Calculate error for each architecture 
def compute_score_errors(science_true, science_pred, cost_true, cost_pred):
    science_abs_error = np.empty([len(science_true)])
    cost_abs_error = np.empty([len(science_true)])
    for i in range(len(science_true)):
        science_true_float = float(science_true[i])
        science_pred_float = float(science_pred[i])
        cost_true_float = float(cost_true[i])
        cost_pred_float = float(cost_pred[i])
        science_abs_error[i] = abs(science_true_float - science_pred_float)*100/(science_true_float+1e-6)  
        cost_abs_error[i] = abs(cost_true_float - cost_pred_float)*100/(cost_true_float+1e-6) 
    return science_abs_error, cost_abs_error

science_err_more_bin, cost_err_more_bin = compute_score_errors(science_ref_unique_more, science_pred_more_bin, cost_ref_unique_more, cost_pred_more_bin)
science_err_more_pois, cost_err_more_pois = compute_score_errors(science_ref_unique_more, science_pred_more_pois, cost_ref_unique_more, cost_pred_more_pois)
science_err_less_bin, cost_err_less_bin = compute_score_errors(science_ref_unique_less, science_pred_less_bin, cost_ref_unique_less, cost_pred_less_bin)
science_err_less_pois, cost_err_less_pois = compute_score_errors(science_ref_unique_less, science_pred_less_pois, cost_ref_unique_less, cost_pred_less_pois)

### Plot values
plt.figure(1)
plt.subplot(211)
plt.scatter(instr_count_unique_more, science_err_more_bin)
plt.xlabel('Instrument count')
plt.ylabel('% Science Error')
plt.title('% Error vs Instrument Count (Binomial Neural Net, High Instr)')
plt.subplot(212)
plt.scatter(instr_count_unique_more, cost_err_more_bin)
plt.xlabel('Instrument count')
plt.ylabel('% Cost Error')
#plt.title('% Error vs Instrument Count for Cost Model (Binomial Neural Net, High Instr)')
plt.show()

plt.figure(2)
plt.subplot(211)
plt.scatter(instr_count_unique_more, science_err_more_pois)
plt.xlabel('Instrument count')
plt.ylabel('% Science Error')
plt.title('% Error vs Instrument Count (Poisson Neural Net, High Instr)')
plt.subplot(212)
plt.scatter(instr_count_unique_more, cost_err_more_pois)
plt.xlabel('Instrument count')
plt.ylabel('% Cost Error')
# plt.title('% Error vs Instrument Count for Cost Model (Poisson Neural Net, High Instr)')
plt.show()

plt.figure(3)
plt.subplot(211)
plt.scatter(instr_count_unique_less, science_err_less_bin)
plt.xlabel('Instrument count')
plt.ylabel('% Science Error')
plt.title('% Error vs Instrument Count (Binomial Neural Net, Low Instr)')
plt.subplot(212)
plt.scatter(instr_count_unique_less, cost_err_less_bin)
plt.xlabel('Instrument count')
plt.ylabel('% Cost Error')
#plt.title('% Error vs Instrument Count for Cost Model (Binomial Neural Net, Low Instr)')
plt.show()

plt.figure(4)
plt.subplot(211)
plt.scatter(instr_count_unique_less, science_err_less_pois)
plt.xlabel('Instrument count')
plt.ylabel('% Science Error')
plt.title('% Error vs Instrument Count (Poisson Neural Net, Low Instr)')
plt.subplot(212)
plt.scatter(instr_count_unique_less, cost_err_less_pois)
plt.xlabel('Instrument count')
plt.ylabel('% Cost Error')
#plt.title('% Error vs Instrument Count for Cost Model (Poisson Neural Net, Low Instr)')
plt.show()

### Find number of architectures with science and cost error less than a threshold
percent_threshold = 20
def error_threshold(thresh, science_error, cost_error, dist, instr_num_type):
    science_arch_count = 0
    cost_arch_count = 0
    for i in range(len(science_error)):
        if (science_error[i] <= thresh):
            science_arch_count += 1
        if (cost_error[i] <= thresh):
            cost_arch_count += 1
    print(str(science_arch_count) + ' architectures out of ' + str(len(science_error)) + ' have science error percentage less than or equal to ' + str(thresh) + ' % , Neural Net: ' + dist + ' ' + instr_num_type + ' Instr')
    print(str(cost_arch_count) + ' architectures out of ' + str(len(science_error)) + ' have cost error percentage less than or equal to ' + str(thresh) + ' % , Neural Net: ' + dist + ' ' + instr_num_type + ' Instr')

error_threshold(percent_threshold, science_err_more_bin, cost_err_more_bin, 'Binomial', 'High')
error_threshold(percent_threshold, science_err_more_pois, cost_err_more_pois, 'Poisson', 'High')
error_threshold(percent_threshold, science_err_less_bin, cost_err_less_bin, 'Binomial', 'Low')
error_threshold(percent_threshold, science_err_less_pois, cost_err_less_pois, 'Poisson', 'Low')

### Historgram of errors
plt.figure(5)
plt.subplot(211)
plt.hist(science_err_more_bin)#, bins=np.arange(min(science_err_more_bin),max(science_err_more_bin),5))
plt.xlabel('Science %')
plt.ylabel('Count')
plt.title('Error Histogram (Binomial Neural Net, High Instr)')
plt.subplot(212)
plt.hist(cost_err_more_bin)#, bins=np.arange(min(cost_err_more_bin),max(cost_err_more_bin),5))
plt.xlabel('Cost %')
plt.ylabel('Count')
#plt.title('Cost Error Histogram (Binomial Neural Net, High Instr)')
plt.show()

plt.figure(6)
plt.subplot(211)
plt.hist(science_err_more_pois)#, bins=np.arange(min(science_err_more_pois),max(science_err_more_pois),5))
plt.xlabel('Science %')
plt.ylabel('Count')
plt.title('Error Histogram (Poisson Neural Net, High Instr)')
plt.subplot(212)
plt.hist(cost_err_more_pois)#, bins=np.arange(min(cost_err_more_pois),max(cost_err_more_pois),5))
plt.xlabel('Cost %')
plt.ylabel('Count')
#plt.title('Cost Error Histogram (Poisson Neural Net, High Instr)')
plt.show()

plt.figure(7)
plt.subplot(211)
plt.hist(science_err_less_bin)#, bins=np.arange(min(science_err_less_bin),max(science_err_less_bin),5))
plt.xlabel('Science %')
plt.ylabel('Count')
plt.title('Error Histogram (Binomial Neural Net, Low Instr)')
plt.subplot(212)
plt.hist(cost_err_less_bin)#, bins=np.arange(min(cost_err_less_bin),max(cost_err_less_bin),5))
plt.xlabel('Cost %')
plt.ylabel('Count')
#plt.title('Cost Error Histogram (Binomial Neural Net, Low Instr)')
plt.show()

plt.figure(8)
plt.subplot(211)
plt.hist(science_err_less_pois)#, bins=np.arange(min(science_err_lesS_pois),max(science_err_less_pois),5))
plt.xlabel('Science %')
plt.ylabel('Count')
plt.title('Error Histogram (Poisson Neural Net, Low Instr)')
plt.subplot(212)
plt.hist(cost_err_less_pois)#, bins=np.arange(min(cost_err_less_pois),max(cost_err_less_pois),5))
plt.xlabel('Cost %')
plt.ylabel('Count')
#plt.title('Cost Error Histogram (Poisson Neural Net, Low Instr)')
plt.show()

### Error Box Plots
# Remove statistical outliers
quartile_thresh = 1.2
def remove_outliers(scores, archs):
    third_quartile = np.percentile(scores, 75)
    archs_outlier = []
    outlier_thresh = quartile_thresh*third_quartile
    scores_list = list(scores)
    index = 0
    for value in scores_list:
        #value = scores_list[i]
        if (value >= outlier_thresh):
            scores_list.remove(value)
            archs_outlier.append(archs[index])
        index += 1
    return scores_list, archs_outlier

science_err_more_bin_mod, outliers_science_more_bin_mod = remove_outliers(science_err_more_bin, archs_unique_more)
cost_err_more_bin_mod, outliers_cost_more_bin_mod = remove_outliers(cost_err_more_bin, archs_unique_more)

science_err_more_pois_mod, outliers_science_more_pois_mod = remove_outliers(science_err_more_pois, archs_unique_more)
cost_err_more_pois_mod, outliers_cost_more_pois_mod = remove_outliers(cost_err_more_pois, archs_unique_more)

science_err_less_bin_mod, outliers_science_less_bin_mod = remove_outliers(science_err_less_bin, archs_unique_less)
cost_err_less_bin_mod, outliers_cost_less_bin_mod = remove_outliers(cost_err_less_bin, archs_unique_less)

science_err_less_pois_mod, outliers_science_less_pois_mod = remove_outliers(science_err_less_pois, archs_unique_less)
cost_err_less_pois_mod, outliers_cost_less_pois_mod = remove_outliers(cost_err_less_pois, archs_unique_less)
    
### Finding number of instruments in outlier architectures
def num_instr_outliers (outliers, datatype_string, fig_num):
    n_archs_outliers = []
    for architecture in outliers:
        n_archs_outliers.append(get_instr_count(architecture))
    plt.figure(fig_num)
    plt.hist(n_archs_outliers)
    plt.xlabel('Number of instruments')
    plt.ylabel('Number of architectures')
    plt.title('Outlier Histogram '+datatype_string)
    plt.show()
    return n_archs_outliers

num_instr_outliers_science_more_bin = num_instr_outliers(outliers_science_more_bin_mod, "Science - Binomial - High Instr", 9)
num_instr_outliers_cost_more_bin = num_instr_outliers(outliers_cost_more_bin_mod, "Cost - Binomial - High Instr", 10)

num_instr_outliers_science_more_pois = num_instr_outliers(outliers_science_more_pois_mod, "Science - Poisson - High Instr", 11)
num_instr_outliers_cost_more_pois = num_instr_outliers(outliers_cost_more_pois_mod, "Cost - Poisson - High Instr", 12)

num_instr_outliers_science_less_bin = num_instr_outliers(outliers_science_less_bin_mod, "Science - Binomial - Low Instr", 13)
num_instr_outliers_cost_less_bin = num_instr_outliers(outliers_cost_less_bin_mod, "Cost - Binomial - Low Instr", 14)

num_instr_outliers_science_less_pois = num_instr_outliers(outliers_science_less_pois_mod, "Science - Poisson - Low Instr", 15)
num_instr_outliers_cost_less_pois = num_instr_outliers(outliers_cost_less_pois_mod, "Cost - Poisson - Low Instr", 16)

plt.figure(17)
plt.subplot(211)
#plt.boxplot(science_err_more_bin_mod)
plt.boxplot(science_err_more_bin, showfliers=False)
plt.xlabel('')
plt.ylabel('Science Error %')
plt.title('Error Box Plot (Binomial Neural Net, High Instr)')
plt.subplot(212)
#plt.boxplot(cost_err_more_bin_mod)
plt.boxplot(cost_err_more_bin, showfliers=False)
plt.xlabel('')
plt.ylabel('Cost Error %')
plt.show()

plt.figure(18)
plt.subplot(211)
#plt.boxplot(science_err_more_pois_mod)
plt.boxplot(science_err_more_pois, showfliers=False)
plt.xlabel('')
plt.ylabel('Science Error %')
plt.title('Error Box Plot (Poisson Neural Net, High Instr)')
plt.subplot(212)
#plt.boxplot(cost_err_more_pois_mod)
plt.boxplot(cost_err_more_pois, showfliers=False)
plt.xlabel('')
plt.ylabel('Cost Error %')
plt.show()

plt.figure(19)
plt.subplot(211)
#plt.boxplot(science_err_less_bin_mod)
plt.boxplot(science_err_less_bin, showfliers=False)
plt.xlabel('')
plt.ylabel('Science Error %')
plt.title('Error Box Plot (Binomial Neural Net, Low Instr)')
plt.subplot(212)
#plt.boxplot(cost_err_less_bin_mod)
plt.boxplot(cost_err_less_bin, showfliers=False)
plt.xlabel('')
plt.ylabel('Cost Error %')
plt.show()

plt.figure(20)
plt.subplot(211)
#plt.boxplot(science_err_less_pois_mod)
plt.boxplot(science_err_less_pois, showfliers=False)
plt.xlabel('')
plt.ylabel('Science Error %')
plt.title('Error Box Plot (Poisson Neural Net, Low Instr)')
plt.subplot(212)
#plt.boxplot(cost_err_less_pois_mod)
plt.boxplot(cost_err_less_pois, showfliers=False)
plt.xlabel('')
plt.ylabel('Cost Error %')
plt.show()

### Writing outliers to text file
print('Saving outlier architectures to text file.....')
def write_outliers_to_file(filename, outliers):
    outliers_file = open(filename,"w")
    outliers_file.write("Outliers (>"+str(quartile_thresh)+"*third_quartile) \n")
    for outlier_arch in outliers:
        outliers_file.write(outlier_arch+" \n")
    outliers_file.close()
    
filename_science_err_more_bin = "outliers science binomial highinstr.txt"
write_outliers_to_file(filename_science_err_more_bin, outliers_science_more_bin_mod)
filename_cost_err_more_bin = "outliers cost binomial highinstr.txt"        
write_outliers_to_file(filename_cost_err_more_bin, outliers_cost_more_bin_mod)

filename_science_err_more_pois = "outliers science poisson highinstr.txt"
write_outliers_to_file(filename_science_err_more_pois, outliers_science_more_pois_mod)
filename_cost_err_more_pois = "outliers cost poisson highinstr.txt"        
write_outliers_to_file(filename_cost_err_more_pois, outliers_science_more_pois_mod)

filename_science_err_less_bin = "outliers science binomial lowinstr.txt"
write_outliers_to_file(filename_science_err_less_bin, outliers_science_less_bin_mod)
filename_cost_err_less_bin = "outliers cost binomial lowinstr.txt"        
write_outliers_to_file(filename_cost_err_less_bin, outliers_cost_less_bin_mod)

filename_science_err_less_pois = "outliers science poisson lowinstr.txt"
write_outliers_to_file(filename_science_err_less_pois, outliers_science_less_pois_mod)
filename_cost_err_less_pois = "outliers cost poisson lowinstr.txt"        
write_outliers_to_file(filename_cost_err_less_pois, outliers_cost_less_pois_mod)