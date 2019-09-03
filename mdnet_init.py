# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:22:13 2019

@author: pc
"""

from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers

from mdnet_keras import create_model


    
def initialize_model():
    net = create_model()  #untrained
    net.load_weights('model_data/MD_tarined_weights.h5')  # weights loaded till fc5 layer  
    conv_net = Model(net.input, net.get_layer('flatten').output)
    #3*3*512= 4608
    fc_in = Input(shape =(4608,))
    next_input = fc_in
    for layer in net.layers[-6:]:
        next_input = layer(next_input)
    fc_net = Model(fc_in, next_input)
    
    # this bbox regression head is crated but unused
    # sklearn Ridge Regression Model is used instead
    bbrin = Input(shape =(4608,)) 
    bbrout = Dense(512, activation = 'linear', kernel_regularizer=regularizers.l2(100), name = 'reg')(bbrin)
    bbox_reg_net =  Model(bbrin, bbrout)
    
    net_body = Model(net.input, net.layers[-2].get_output_at(0))
    return net, conv_net, fc_net, net_body, bbox_reg_net

    