# -*- coding: utf-8 -*-
"""

@author: roshan94
"""
import numpy as np
import csv
#from scipy.stats import poisson
import matplotlib.pyplot as plt

### Sample Poisson Distribution
np.random.seed(0)
s = np.random.poisson(1.5, 3000)
plt.figure(1)
count, bins, patches = plt.hist(s, 59, density=False)
plt.show()

### Determine number of architectures using poisson distribution
# Since np.random.poisson only shows number of architectures for specific number
# of instruments, an interpolation function is defined to determine the number of 
# architectures at the intermediate values of number of instruments 
def interpolator(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    slope = (y2-y1)/(x2-x1)
    interp_points = np.linspace(x1,x2,x2-x1+1)
    n_archs_interp = np.empty(len(interp_points))
    for i in range(len(interp_points)):
        n_archs_interp[i] = round(np.dot(slope,(interp_points[i]-x1)) + y1)
    return interp_points, n_archs_interp

n_nonzero = np.count_nonzero(count)
nonzero_indices = np.nonzero(count)
#print(nonzero_indices)
p1 = np.empty(2)
p2 = np.empty(2)
n_instr = []
n_archs_instr = []
for i in range(n_nonzero-1):
    p1[0] = nonzero_indices[0][i]
    #print(p1[0])
    p1[1] = count[int(p1[0])]
    p2[0] = nonzero_indices[0][i+1]
    p2[1] = count[int(p2[0])]
    # The number of architectures for 0 and 1 instrument(s) is fixed 
    # (1 and 60 respectively), so the first interpolation point is moved to 2 
    # and its corresponding vount value is reduced by 61
    if i==0:
        p1[0] = p1[0] + 2
        p1[1] = p1[1] - 61
    instr_num, archs_instr = interpolator(p1,p2)
    n_instr.append(instr_num)
    n_archs_instr.append(archs_instr)
    
def archs_total(archs_instr_num):
    archs_instr_array = np.array(archs_instr_num)
    n_arrays = len(archs_instr_array)
    index = 0
    archs_sum = 0
    arch_num_array = np.empty([59])
    for i in range(n_arrays):
        n_archs_current = archs_instr_array[i]
        num = len(n_archs_current)
        for j in range(num-1):
            arch_num_array[index] = n_archs_current[j]
            index += 1
        archs_sum_current = np.sum(n_archs_current)
        archs_sum += archs_sum_current
    return archs_sum, arch_num_array

sum_architectures, architecture_numbers = archs_total(n_archs_instr)
print("Total number of architectures is " + str(sum_architectures))
#print("Instrument Numbers: "+ str(n_instr))
print("Number of Architectures for each instrument number: " + str(architecture_numbers))

arch_nums_list = list(architecture_numbers)
arch_nums_list.insert(0,60)
arch_nums = np.linspace(1,60,60)
n_arch_nums = 60
plt.figure(2)
plt.bar(arch_nums,arch_nums_list)
plt.xlabel("Number of instruments per architecture")
plt.ylabel("Number of architectures to generate")

### Sample Gaussian Distribution
mean, std = 30, 10
np.random.seed(2)
s2 = np.random.normal(mean, std, 15300)
plt.figure(3)
count2, bins2, patches2 = plt.hist(s2, 59, density=False)

### Number of architectures from 0 to 60 instruments
count2[0:19] = count2[0:19] + 200
count2[41:57] = count2[41:57] + 200
count2[57:] = 60

#print(len(arch_nums))
#print(len(count2))
plt.figure(4)
plt.bar(arch_nums,count2)
print("Total number of architectures is " + str(np.sum(count2)))
print("Number of Architectures for each instrument number: " + str(count2))

### Write to csv file 
### (This csv file is used by CCNeuralNetDatasetGeneration.java to generate training datasets for the two Neural Nets)
#line = 0
#with open('instrument_numbers.csv', mode='w') as instr_num_file:
    #instr_num_writer = csv.writer(instr_num_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    #instr_num_writer.writerow(['MedInstr','LowInstr'])
    #instr_num_writer.writerow(['60','60'])
    #while line < n_arch_nums:
        #instr_num_writer.writerow([str(int(count2[line])),str(int(architecture_numbers[line]))])
        #line += 1

