# -*- coding: utf-8 -*-
"""

@author: roshan94
"""
import pygmo as pg
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

file_path_medinstr = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\vassar_data_medinstr_train.csv'
file_path_lowinstr = 'C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\vassar_data_lowinstr_train.csv'

archs_medinstr, science_medinstr, cost_medinstr = read_csv(file_path_medinstr)
archs_lowinstr, science_lowinstr, cost_lowinstr = read_csv(file_path_lowinstr)

archs_all = archs_medinstr + archs_lowinstr
science_all = science_medinstr + science_lowinstr
cost_all = cost_medinstr + cost_lowinstr

def get_instr_count (arch):
    n_instr = 0
    for bit in arch:
        if (int(bit) == 1):
            n_instr += 1
    return n_instr

### Remove duplicates
def remove_duplicates(archs_list, science_ref, cost_ref):
    print(len(archs_list))
    print(len(science_ref))
    print(len(cost_ref))
    archs_list_unique = list(set(archs_list))
    science_ref_unique = []
    cost_ref_unique = []
    n_instr_unique = []
    for i in range(len(archs_list_unique)):
        index = archs_list.index(archs_list_unique[i])
        #print(index)
        science_ref_unique.append(science_ref[index])
        cost_ref_unique.append(cost_ref[index])
        n_instr_unique.append(get_instr_count(archs_list_unique[i]))
    print(len(archs_list_unique))
    print(len(science_ref_unique))
    print(len(cost_ref_unique))
    return archs_list_unique, science_ref_unique, cost_ref_unique, n_instr_unique

archs_unique, science_all_unique, cost_all_unique, num_instr = remove_duplicates(archs_all, science_all, cost_all) 

### Find the non-dominated architectures
scores = np.empty([len(archs_unique),2])
for i in range(len(archs_unique)):
    scores[i] = [-float(science_all_unique[i]), float(cost_all_unique[i])]
    
#science_floats = map(float, science_all_unique)
#cost_floats = map(float, cost_all_unique)    

#non_dominated_points = pg.non_dominated_front_2d(scores)
ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(scores)
#domination_bools = pg.pareto_dominance(science_floats, cost_floats)

### Plot the pareto fronts
def plot_pareto_fronts(non_dom_front, archs, science, cost, n_fronts_plot):
    colors = iter(cm.rainbow(np.linspace(0, 1, n_fronts_plot)))
    archs_pareto = []
    science_pareto = []
    cost_pareto = []
    archs_instr_pareto = []
    n_archs = 0
    for i in range(n_fronts_plot):
        archs_rank = []
        science_rank = []
        cost_rank = []
        archs_instr_rank = []
        pareto_rank = non_dom_front[i]
        #indices_par_rank = len(pareto_rank)
        for index in pareto_rank:
            archs_rank.append(archs[index])
            science_rank.append(float(science[index]))
            cost_rank.append(float(cost[index]))
            archs_instr_rank.append(get_instr_count(archs[index]))
            n_archs += 1
        archs_pareto.append(archs_rank)
        science_pareto.append(science_rank)
        cost_pareto.append(cost_rank)
        archs_instr_pareto.append(archs_instr_rank)        
    plt.figure(1)
    for i in range(n_fronts_plot):
        plt.scatter(science_pareto[i],cost_pareto[i],color=next(colors),label='pareto rank '+str(i))
    plt.xlabel('Science')
    plt.ylabel('Cost')
    plt.title('Pareto Fronts')
    #plt.legend(loc='upper right')
    plt.show()
    return archs_instr_pareto, n_archs 

n_instruments_pareto, n_archs_total = plot_pareto_fronts(ndf, archs_unique, science_all_unique, cost_all_unique, 5)

def instrument_histogram(n_instr_par):
    n_instr_list = []
    for n_instr_rank in n_instr_par:
        n_instr_list.append(n_instr_rank)
    plt.figure(2)
    plt.hist(n_instr_list)
    plt.xlabel('Number of instruments')
    plt.ylabel('Number of Architectures')
    plt.title('Histogram of number of instruments for promising architectures')
    plt.show()
    
instrument_histogram(n_instruments_pareto)

### Creating the training dataset
archs_count = 0
n_archs_pareto = 5071 # first 25 pareto ranks
#archs_labels = np.array([[0,1],]*(2*n_archs_pareto)) # for 50-50 promising, non-promising split
archs_labels = np.array([[0,1],]*len(archs_unique))
archs_prom = []
for par_rank in ndf:
    instr_par_rank = len(par_rank)
    for index in par_rank:
        archs_prom.append(archs_unique[index])
    archs_count += instr_par_rank
    if (archs_count >= n_archs_pareto):
        break

### Change labels for the promising architectures
for i in range(archs_count):
    archs_labels[i] = [1,0]

### Find the dominated architectures and append with non-dominated architectures to create training set
archs_dominated = []
n_archs_dom = 0
for architecture in archs_unique:
    if (architecture in archs_prom):
        continue
    elif (architecture not in archs_prom):
        archs_dominated.append(architecture)
    n_archs_dom += 1
    #if (n_archs_dom == n_archs_pareto): # for 50-50 promising, non-promising split
        #break

archs_train_list = archs_prom + archs_dominated
        
### Classifier Neural Net
def create_classifier():
    print('Building Classifier....')
    Classifier = Sequential()
    Classifier.add(Dense(100, input_dim=60))
    Classifier.add(Activation('relu'))
    Classifier.add(Dropout(0.3))
    Classifier.add(Dense(50))
    Classifier.add(Activation('relu'))
    Classifier.add(Dropout(0.3))
    #Classifier.add(Dense(25))
    #Classifier.add(Activation('relu'))
    #Classifier.add(Dropout(0.3))
    Classifier.add(Dense(2))
    Classifier.add(Activation('sigmoid'))
    Classifier.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['binary_accuracy'])
    return Classifier
        
### Train-Test split
arch_training, arch_testing, label_training, label_testing = train_test_split(archs_train_list, archs_labels, test_size=0.2)#, random_state=21)

def arch_list_to_array (arch_list):
    arch_array = np.empty([len(arch_list),60])
    for i in range(len(arch_list)):
        current_arch = arch_list[i]
        for j in range(60):
            arch_array[i][j] = int(current_arch[j])
    return arch_array

archArray_train = arch_list_to_array(arch_training)
archArray_test = arch_list_to_array(arch_testing)

### k-fold cross validation
batch_size = 128
num_epochs = 100

n_folds = 10
best_classifier = None
best_modelfold = 0
best_train_accuracy = 0

val_acc = np.empty([n_folds,2])
train_acc = np.empty([n_folds,2])
test_acc = np.empty([n_folds,2])
kf_class = KFold(n_folds, shuffle=True)
count = 0
for (train_ind, test_ind) in kf_class.split(archArray_train, label_training):
    print("Running Fold ",str(count+1),"/",str(n_folds))
    arch_train_fold = archArray_train[train_ind]
    label_train_fold = label_training[train_ind]
    arch_val_fold = archArray_train[test_ind]
    label_val_fold = label_training[test_ind]
    #print(arch_train_fold.shape)
    #print(label_training.shape)
    #print(arch_val_fold.shape)
    #print(label_training.shape)
    classifier_model = None
    classifier_model = create_classifier()
    classifier_History = classifier_model.fit(arch_train_fold, label_train_fold, batch_size=batch_size, epochs=num_epochs)
    val_acc[count,:] = classifier_model.evaluate(arch_val_fold, label_val_fold, batch_size=batch_size)
    train_acc[count,:] = classifier_model.evaluate(arch_train_fold, label_train_fold, batch_size=batch_size)
    test_acc[count,:] = classifier_model.evaluate(archArray_test, label_testing, batch_size=batch_size)
    #print(train_acc[count,:])
    if (train_acc[count,1] > best_train_accuracy):
        best_train_accuracy = train_acc[count,1]
        best_classifier = classifier_model
        best_modelfold = count
    count += 1
    
print('The fold corresponding to the best classifier is ', str(best_modelfold))
plt.figure(3)
plt.plot(val_acc[:,1], label='val')
plt.plot(train_acc[:,1], label='train')
plt.plot(test_acc[:,1], label='test')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('K-Fold cross validation for classifier')
plt.legend(loc='upper right')
plt.show()

print('Saving the Classifier......')
best_classifier.save('best_kfold_classifier.h5')