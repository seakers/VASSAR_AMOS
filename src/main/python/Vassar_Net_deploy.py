# -*- coding: utf-8 -*-
"""

@author: roshan94
"""
from keras.models import load_model
#from keras.models import Model

# Loading the trained Science and Cost Models from the h5 files
ScienceModel = load_model('Science_NN.h5')
CostModel = load_model('Cost_NN.h5')

arch = ""

science = ScienceModel.predict(arch, verbose=1)
cost = CostModel.predict(arch, verbose=1)