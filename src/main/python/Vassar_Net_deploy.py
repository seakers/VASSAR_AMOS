# -*- coding: utf-8 -*-
"""

@author: roshan94
"""
from keras.models import load_model
import numpy as np 

# arch_listbool = ""

def NeuralNetScienceAndCost (arch_listbool):

    ### Loading the trained Science and Cost Models from the h5 files
    ScienceModel = load_model('Science_NN_uniform.h5')
    CostModel = load_model('Cost_NN_uniform.h5')

    # print(arch_listbool)
    np_list = np.array(arch_listbool)
    np_list= np_list.reshape((1,60))

    science = ScienceModel.predict(np_list, verbose=0)
    cost = CostModel.predict(np_list, verbose=0)
    
    return arch_listbool, science, cost

### Testing
# arch_test = "111100001100011100000000011100000011100011000000000000000011"

# archArray_test = np.zeros([1,60])
# for y in range(60):
    # archArray_test[0][y] = int(arch_test[y])

# scores = NeuralNetScienceAndCost(archArray_test)
# print(scores)

