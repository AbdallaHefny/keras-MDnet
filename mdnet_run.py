import os
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt 
from matplotlib import patches
from keras import optimizers
import argparse 
import time

from utils import get_data, gen_samples, iou_score, extract_regions
from options import Options  
from mdnet_init import initialize_model
from mdnet_finetune import mdnet_finetune
from lr_multiplier import LearningRateMultiplier
from bbox_reg import BBoxRegressor

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seq', default='Car1', help='input seq')
parser.add_argument('-r', '--reg', action='store_true')
args = parser.parse_args()
seq = args.seq
regress = args.reg

##############################
# Initialization
##############################
opts = Options()
path = os.path.join(os.getcwd(), 'data', seq)
imgList, gt = get_data(path) # gt: (x, y, box-width, box-height)

nFrames = len(imgList)
img = imread(imgList[0])
if (len(img.shape) != 3):
        img = np.expand_dims(img, -1)
        img = np.concatenate((img, img, img), -1)
imgSize = img.shape

targetLoc = np.array(gt[0]) # initial bounding box 
result = reg_result = np.zeros((nFrames, 4))
result[0] = reg_result[0] = targetLoc
# result is to keep track of predictions without regression
# result will not be plot if the regressor is active
# if result is needed along with regressor results: 
# it should be plot explicitly

conv_net, fc_net, net_body= initialize_model()

 
#####################################
# Train Bounding Box Regressor 
#####################################
if regress:
    # create BBoxRegressor object
    bbox_reg = BBoxRegressor()
    
    # extract rectangles for training
    pos_examples = gen_samples('uniform', targetLoc, opts.bbreg_nSamples*10, imgSize, 0.3, 1.5, 1.1)
    scores = iou_score(pos_examples, targetLoc)  
    a = targetLoc[2]*targetLoc[3]  # ground truth area
    ar= np.prod(pos_examples[:, 2:], axis = 1)/a  # areas ratio to gt
    pos_examples = pos_examples[(scores>0.6) & (ar>0.6) & (ar<1)] 
    s = min(opts.bbreg_nSamples, len(pos_examples))
    idx = np.random.choice(np.arange(len(pos_examples)), size = s, replace = False)
    pos_examples = pos_examples[idx] 
    
    # extract conv3 features
    n = len(pos_examples)
    regions= extract_regions(img, pos_examples, opts.input_size)  #shape (n, 107, 107, 3)
    regions = regions - 128.
    nBatches = np.ceil(n/opts.batchSize_test)
    feat = np.zeros((n, 4608))
    for i in range(int(nBatches)):
        batch = regions[opts.batchSize_test*i: min(opts.batchSize_test*(i+1), n)]
        res = conv_net.predict(batch)
        feat[opts.batchSize_test*i: min(opts.batchSize_test*(i+1), n), :] = res
        
    # train the regressor
    bbox_reg.train(feat, pos_examples, targetLoc)

#############################
# FC network fine-tuning
#############################
pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_init*8, imgSize, 0.1, 1.2)
scores = iou_score(pos_examples, targetLoc)  
pos_examples = pos_examples[scores>opts.posThr_init] 
s = min(opts.nPos_init, len(pos_examples))
idx = np.random.choice(np.arange(len(pos_examples)), size = s, replace = False)
pos_examples = pos_examples[idx] 

neg_examples1 = gen_samples('uniform', targetLoc, opts.nNeg_init, imgSize, 1, 2, 1.1)
neg_examples2 = gen_samples('whole', targetLoc, opts.nNeg_init, imgSize, 0, 1.2, 1.1)
neg_examples = np.concatenate((neg_examples1, neg_examples2))
scores = iou_score(neg_examples, targetLoc)  
neg_examples = neg_examples[scores<opts.negThr_init] 
s = min(opts.nNeg_init, len(neg_examples))
idx = np.random.choice(np.arange(len(neg_examples)), size = s, replace = False)
neg_examples = neg_examples[idx] 

examples = np.concatenate((pos_examples, neg_examples))

# extract conv3 features
regions= extract_regions(img, examples, opts.input_size)  #shape (n, 107, 107, 3)
regions = regions - 128.
n = len(regions)
nBatches = np.ceil(n/opts.batchSize_test)
output_neurons = conv_net.output_shape[1]
feat = np.zeros((n, output_neurons))
for i in range(int(nBatches)):
    batch = regions[opts.batchSize_test*i: min(opts.batchSize_test*(i+1), n)]
    res = conv_net.predict(batch)
    feat[opts.batchSize_test*i: min(opts.batchSize_test*(i+1), n), :] = res

pos_data = feat[:len(pos_examples)]
neg_data = feat[len(pos_examples): len(neg_examples)]

total_pos_feats = [pos_data[:opts.nPos_update]]   # each element is a numpy array (50,4608)
total_neg_feats = [neg_data[:opts.nNeg_update]]  # each element is a numpy array (200,4608)

#fine tune FC layers
multipliers = {'fc6': 10}
opt = LearningRateMultiplier(optimizers.SGD, lr_multipliers=multipliers, lr=0.0001, 
                             clipnorm=10., decay=0.0005, momentum=0.9)
fc_net.compile(loss='categorical_crossentropy', optimizer=opt)
mdnet_finetune(fc_net, pos_data, neg_data, opts.batchSize, 30)

#plot first bbox
ax = plt.subplot(1,1,1)
ishow = ax.imshow(img)
rect_gt = patches.Rectangle((gt[0][0], gt[0][1]), gt[0][2] , gt[0][3],
                            fill=False, color='blue')    
rect_target = patches.Rectangle((targetLoc[0], targetLoc[1]), targetLoc[2] , targetLoc[3],
                            fill=False, color='red')
ax.add_patch(rect_gt)
ax.add_patch(rect_target)

plt.pause(0.01)
plt.draw()

#################################
#start tracking (Main Loop)
#################################
trans_f = opts.trans_f # 0.6
scale_f = opts.scale_f # 1
multipliers = {'fc6': 10}
opt = LearningRateMultiplier(optimizers.SGD, lr_multipliers=multipliers, lr=0.0003, 
                             clipnorm=10., decay=0.0005, momentum=0.9)
fc_net.compile(loss='categorical_crossentropy', optimizer=opt)

success_thr = 0

for f in range(1,nFrames):
    print("processing frame {} / {}".format(f+1, nFrames))
    t_begin = time.time()
    img = imread(imgList[f])
    if (len(img.shape) != 3):
        img = np.expand_dims(img, -1)
        img = np.concatenate((img, img, img), -1)
    imgSize = img.shape
    # generate samples
    samples = gen_samples('gaussian',  targetLoc, opts.nSamples, imgSize, trans_f, scale_f, valid = True)
    # fc6 features
    n = len(samples)  #256 
    regions= extract_regions(img, samples, opts.input_size)  #shape (128, 107, 107, 3)
    regions = regions - 128.
    nBatches = np.ceil(n/opts.batchSize_test)  
    feat = np.zeros((n, 2))  #shape (256, 2) #fc6 output
    for i in range(int(nBatches)):
        batch = regions[opts.batchSize_test*i: opts.batchSize_test*(i+1)]
        res = net_body.predict(batch)  # shape (64,2)
        feat[opts.batchSize_test*i: opts.batchSize_test*(i+1), :] = res
    
    idx = np.argsort(feat[:, 1])  
    scores = np.sort(feat[:,1])   
    target_score = np.mean(scores[-5:])
    targetLoc = np.rint(np.mean(samples[idx[-5:]], axis = 0)) 
      
    if target_score < success_thr:
        trans_f = min(1.5, 1.1*trans_f)
        targetLoc = result[f-1]     
    else:
       trans_f = 0.6
       
    result[f] = targetLoc
    reg_targetLoc = targetLoc  # it will be updated if regressor is active
    
    # Regress target location  
    # TODO get rid of computing again conv3 features of top 5 regions
    if regress and target_score > success_thr:
        feat = conv_net.predict(regions[idx[-5:]])   # (5, 4608)    
        reg_targetLoc = bbox_reg.predict(feat, samples[idx[-5:]])
            
    reg_result[f] = reg_targetLoc
    
    
    # save conv features for some samples to tune fc layers later   
    if target_score > success_thr:
        pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_update*4, imgSize, 0.1, 1.2)
        scores = iou_score(pos_examples, targetLoc)  
        pos_examples = pos_examples[scores>opts.posThr_update] 
        s = min(opts.nPos_update, len(pos_examples))
        idx = np.random.choice(np.arange(len(pos_examples)), size = s, replace = False)
        pos_examples = pos_examples[idx] 
        
        neg_examples = gen_samples('uniform', targetLoc, opts.nNeg_update*2, imgSize, 1.5, 1.2)
        scores = iou_score(neg_examples, targetLoc)  
        neg_examples = neg_examples[scores<opts.negThr_update] 
        s = min(opts.nNeg_update, len(neg_examples))
        idx = np.random.choice(np.arange(len(neg_examples)), size = s, replace = False)
        neg_examples = neg_examples[idx] 
        
        examples = np.concatenate((pos_examples, neg_examples))  
        
        # extract conv3 features
        n = len(examples)
        regions= extract_regions(img, examples, opts.input_size)  #shape (n, 107, 107, 3)
        regions = regions - 128.
        nBatches = np.ceil(n/opts.batchSize_test)  
        feat = np.zeros((n, 4608)) #(250, 4608)
        for i in range(int(nBatches)):
            batch = regions[opts.batchSize_test*i: min(opts.batchSize_test*(i+1), n)]
            res = conv_net.predict(batch)
            feat[opts.batchSize_test*i: min(opts.batchSize_test*(i+1), n), :] = res
        
        pos_feats = feat[:len(pos_examples)]                    #(50, 4608)
        neg_feats = feat[len(pos_examples): len(neg_examples)]  #(200,4608)
        total_pos_feats.append(pos_feats)       # list length = number of success frames
        total_neg_feats.append(neg_feats)       # list length = number of success frames
  
        
        if len(total_pos_feats) > opts.nFrames_long:  # keep length <= 100
            del total_pos_feats[0]
        if len(total_neg_feats) > opts.nFrames_short: # keep length <= 20
            del total_neg_feats[0]
    
    #short term update
    if target_score < success_thr: 
        n_posframes = min(opts.nFrames_short, len(total_pos_feats))    
        pos_data = total_pos_feats[-n_posframes].reshape((-1,4608))
        for x in total_pos_feats[-n_posframes+1:]:
          pos_data = np.concatenate((pos_data, x.reshape((-1,4608)))) 
          
        neg_data = total_neg_feats[0].reshape((-1,4608))
        for x in total_neg_feats[1:]:
          neg_data = np.concatenate((neg_data, x.reshape((-1,4608)))) 
              
        #fine tune FC layers
        mdnet_finetune(fc_net, pos_data, neg_data, opts.batchSize, 10) 
        
        
    # long term update    
    elif f % opts.update_interval == 0:      
        pos_data = total_pos_feats[0].reshape((-1,4608))
        for x in total_pos_feats[1:]:
          pos_data = np.concatenate((pos_data, x.reshape((-1,4608)))) 
          
        neg_data = total_neg_feats[0].reshape((-1,4608))
        for x in total_neg_feats[1:]:
          neg_data = np.concatenate((neg_data, x.reshape((-1,4608)))) 
        #fine tune FC layers
        mdnet_finetune(fc_net, pos_data, neg_data, opts.batchSize, 10) 
   
    # plot    
    ishow.set_data(img)
    
    rect_gt.set_xy((gt[f][0], gt[f][1]))
    rect_gt.set_width(gt[f][2])
    rect_gt.set_height(gt[f][3])
       
    rect_target.set_xy((reg_targetLoc[0], reg_targetLoc[1]))
    rect_target.set_width(reg_targetLoc[2])
    rect_target.set_height(reg_targetLoc[3])    
              
    plt.pause(.01)
    plt.draw()
    
    t_total = time.time() - t_begin
    print("time required: {:.3f} s",format(t_total))