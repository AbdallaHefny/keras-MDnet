class Options:
    batchSize_test = 64     # I changed it from 256
    
    bbreg = True
    bbreg_nSamples = 1000
    
    # learning policy
    batchSize = 128
    batch_pos = 32
    batch_neg = 96
    
    # initial training policy
    learningRate_init = 0.0001 # x10 for fc6
    maxiter_init = 30

    nPos_init = 500
    nNeg_init = 5000
    posThr_init = 0.7
    negThr_init = 0.5
    
    # update policy
    learningRate_update = 0.0003 # x10 for fc6
    maxiter_update = 10

    nPos_update = 50
    nNeg_update = 200
    posThr_update = 0.7
    negThr_update = 0.3
    
    update_interval = 10 # interval for long-term update
    
    nFrames_long = 100 # long-term period
    nFrames_short = 20 #short-term period
    
    # data gathering policy
    nFrames_long = 100 # long-term period
    nFrames_short = 20 # short-term period
    
    # cropping policy
    input_size = 107
    
    # sampling policy
    nSamples = 256  
    trans_f = 0.6 # translation std: mean(width,height)*trans_f/2
    scale_f = 1 # scaling std: scale_factor^(scale_f/2)
    