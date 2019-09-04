import numpy as np


def mdnet_finetune(fc_net, pos_data, neg_data, batch_size, iterataions):
    """
    from: https://github.com/hyeonseobnam/py-MDNet/blob/master/tracking/run_tracker.py
    with modifications 
    """
    batch_pos = 32
    batch_neg = 96
    batch_test = 256
    batch_neg_cand = 1024
    
    targets_pos = np.zeros((32, 2))
    targets_pos[:, 1] =1
    targets_neg = np.zeros((96, 2))
    targets_neg[:, 0] =1
    
    targets = np.zeros((128, 2))
    targets[:32, 1] = 1
    targets[32:, 0] = 1
    
    pos_idx = np.random.permutation(len(pos_data))
    neg_idx = np.random.permutation(len(neg_data))
    while (len(pos_idx) < batch_pos * iterataions):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(len(pos_data))])
    while (len(neg_idx) < batch_neg_cand * iterataions):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(len(neg_data))])
    
    pos_pointer = 0
    neg_pointer = 0
    
    for i in range (iterataions):
        
        pos_next = pos_pointer + batch_pos
        batch_pos_feats = pos_data[pos_idx[pos_pointer: pos_next]]
        pos_pointer = pos_next
        
        neg_next = neg_pointer + batch_neg_cand
        batch_neg_feats = neg_data[neg_idx[neg_pointer: neg_next]]
        neg_pointer = neg_next
        
        if batch_neg_cand > batch_neg:
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                
                score = fc_net.predict(batch_neg_feats[start:end])
                if start==0:
                    neg_cand_score = score[:, 1]
                else:
                    neg_cand_score = np.concatenate((neg_cand_score, score[:, 1]), axis = 0)
                    
            top_idx = np.argsort(neg_cand_score)[-batch_neg:]
            batch_neg_feats = batch_neg_feats[top_idx]
            
        # train
        batch = np.concatenate((batch_pos_feats, batch_neg_feats), axis = 0)
        _ = fc_net.fit(batch, targets, shuffle = True, verbose = 0)
        


#     