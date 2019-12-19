# -*- coding: utf-8 -*-
"""

@author: roshan94
"""
from keras.models import load_model
from keras.utils import plot_model

ScienceModel = load_model('Science_NN_uniform.h5')
CostModel = load_model('Cost_NN_uniform.h5')

plot_model(ScienceModel, to_file='ScienceNN.png', show_shapes=True , show_layer_names=True , expand_nested=False , dpi=96)
plot_model(CostModel, to_file='CostNN.png', show_shapes=True , show_layer_names=True , expand_nested=False , dpi=96)