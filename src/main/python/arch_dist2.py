# -*- coding: utf-8 -*-
"""

@author: roshan94
"""
import numpy as np
import matplotlib.pyplot as plt

### Sample Gaussian Distribution
mean, std = 30, 10
np.random.seed(10)
s2 = np.random.normal(mean, std, 15300)
plt.figure(1)
count2, bins2, patches2 = plt.hist(s2, 60, density=False)

### Number of architectures from 1 to 60 instruments
count2[0] = 60
count2[1:19] = count2[1:19] + 200
count2[41:59] = count2[41:59] + 200
count2[-1] = 60
arch_nums = np.linspace(1,60,60)
n_arch_nums = 59
#print(len(arch_nums))
#print(len(count2))
plt.figure(2)
plt.bar(arch_nums,count2)
plt.xlabel("Number of instruments per architecture")
plt.ylabel("Number of architectures to generate")
print("Total number of architectures is " + str(np.sum(count2)))
print("Number of Architectures for each instrument number: " + str(count2))



