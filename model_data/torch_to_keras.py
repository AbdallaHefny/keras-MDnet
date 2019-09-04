# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:10:23 2019

@author: pc
"""

import torch
import numpy as np

from mdnet_keras import create_model

###############################
# Load torch model
###############################
     
states = torch.load('model_data/mdnet_vot-otb_new.pth') 
      
shared_layers = states['shared_layers']
shared = list(states['shared_layers'].values())
numpy_weights = []
for sh in shared:
    numpy_weights.append(np.array(sh))

for (i, nw) in enumerate(numpy_weights[:5]):
    if (i%2==0):
        nw = np.expand_dims(nw,-1)
        nw = np.expand_dims(nw,-1)
        nw = np.swapaxes(nw,0,-1)
        nw = np.swapaxes(nw,1,-2)
        nw = np.squeeze(nw, 0)
        nw = np.squeeze(nw, 0)
        numpy_weights[i] = nw
numpy_weights[-4] = numpy_weights[-4].T
numpy_weights[-2] = numpy_weights[-2].T

#for nw in numpy_weights:
#    print(nw.shape)
    
#####################################################################
# save numpy weights to npy file to load them to tensorflow later
#####################################################################
#np.save("model_data/np_weights_new.npy", np.array(numpy_weights), allow_pickle = True)
# to load use: np.load("np_weights.npy", allow_pickle = True)


#######################################
# create model and pass torch weights
#######################################

trained = create_model()  
 
c = 0
for i in range (len(trained.layers[:-3])):
    if (trained.layers[i].name.startswith('conv') or trained.layers[i].name.startswith('fc')):
        print(trained.layers[i].name)
        wb = [numpy_weights[c], numpy_weights[c+1]]
        trained.layers[i].set_weights(wb)
        c += 2
  
#############################################
# save weights to be loaded and used later
#############################################  
trained.save_weights('model_data/MD_tarined_weights.h5')