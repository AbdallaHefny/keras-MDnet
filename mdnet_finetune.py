# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:44:19 2019

@author: pc
"""

import numpy as np


def mdnet_finetune(fc_net, pos_data, neg_data, batch_size, iterataions):
    #  batch_size = 128 
    n_pos = len(pos_data)  
    n_neg = len(neg_data)  
    
    if (n_pos> 32*iterataions):
        idx = np.random.permutation(32*iterataions)
        pos_data = pos_data[idx]

    count_neg = 0
    count_pos= 0
    
    for t in range(iterataions):
        scores_hneg = []
        remaining = n_neg - count_neg
        if remaining > 1024:
            part_neg_data = neg_data[count_neg: count_neg+1024]
            count_neg += 1024
        else:
            part_neg_data = neg_data[count_neg: count_neg+remaining]
            count_neg = 0
            idx = np.random.permutation(n_neg)
            neg_data = neg_data[idx]
        
        num_batches = int(np.ceil(len(part_neg_data)/batch_size)) 
                                                                 
        for h in range (num_batches):
            batch_neg = part_neg_data[h*batch_size : min((h+1)*batch_size,len(part_neg_data))]
            targets_neg = np.zeros((len(batch_neg), 2))  
            targets_neg[:,0] = 1  
            
            res = fc_net.predict(batch_neg) # (batch_size, 2)numpy array  
            scores_hneg.extend(res[:,1].tolist())
        scores_hneg = np.array(scores_hneg)  #shape (len(part_neg_data),)

        idx = np.argsort(scores_hneg)
        hard_data = part_neg_data[idx[-96:]]  
                                              
                                              
              
        remaining = n_pos - count_pos
        if remaining > 32:
            part_pos_data = pos_data[count_pos: count_pos+64]
            count_pos += 32              
        else:
            part_pos_data = pos_data[count_pos: count_pos+remaining]
            count_pos = 0
            idx = np.random.permutation(n_pos)
            pos_data = pos_data[idx]
        
        batch = np.concatenate((part_pos_data, hard_data), axis = 0) 
                                                                     
        targets_whole = np.zeros((len(batch), 2))
        targets_whole[:len(part_pos_data),1] = 1
        targets_whole[len(part_pos_data):,0] = 1
        
#        tr_loss = fc_net.train_on_batch(batch, targets_whole)
        _ = fc_net.fit(batch, targets_whole, batch_size = 128, shuffle = True, verbose = 0)
 