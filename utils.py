# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:07:15 2019

@author: pc
"""

import os
import numpy as np
from PIL import Image

def get_data(path):
    """
    returns list of image paths and gt rectangles
    """
    imgpath = os.path.join(path, 'img')
    imgList = []
    for name in (os.listdir(imgpath)):
        imgList.append(os.path.join(imgpath, name))
    
    boxpath = os.path.join(path, 'groundtruth_rect.txt')
    with open(boxpath) as f:
        gt = f.readlines()
    gt = [g.strip() for g in gt]
    
    for i in range (len(gt)):
        gt[i] = [int(x) for x in gt[i].split(',')]
    
    imgList = imgList[:len(gt)]    
    return imgList, gt

def gen_samples(method, bb, n, imgSize, trans_f = 1, scale_f = 1 , aspect_f=None, valid = False):
    img_size = np.array(imgSize[0:2])   
    # bb is numpy array [x, y, box-width, box-height]
    sample = [bb[0]+bb[2]/2, bb[1]+bb[3]/2, bb[2], bb[3]] #[center_x center_y width height]
    sample = np.array(sample)
    samples = np.tile(sample, (n,1))
    
    # vary aspect ratio
    if aspect_f is not None:
        ratio = np.random.rand(n,1)*2-1
        samples[:,2:] *= aspect_f ** np.concatenate([ratio, -ratio],axis=1)
    
    if method == 'gaussian':
        samples[:,:2] += trans_f * np.mean(bb[2:]) * np.clip(0.5*np.random.randn(n,2),-1,1)
        samples[:,2:] *= scale_f ** np.clip(0.5*np.random.randn(n,1),-1,1)
    
    if method == 'uniform':
        samples[:,:2] += trans_f * np.mean(bb[2:]) * (np.random.rand(n,2)*2-1)
        samples[:,2:] *= scale_f ** (np.random.rand(n,1)*2-1)
        
        
    if method == 'whole':
        m = int(2*np.sqrt(n))
        xy = np.dstack(np.meshgrid(np.linspace(0,1,m),np.linspace(0,1,m))).reshape(-1,2)
        xy = np.random.permutation(xy)[:n]
        samples[:,:2] = bb[2:]/2 + xy * (img_size-bb[2:]/2-1)
        #samples[:,:2] = bb[2:]/2 + np.random.rand(n,2) * (self.img_size-bb[2:]/2-1)
        samples[:,2:] *= scale_f ** (np.random.rand(n,1)*2-1)

    samples[:,2:] = np.clip(samples[:,2:], 10, img_size-10)
    
    if valid:
        samples[:,:2] = np.clip(samples[:,:2], samples[:,2:]/2, img_size-samples[:,2:]/2-1)
    else:
        samples[:,:2] = np.clip(samples[:,:2], 0, img_size)
    
    # (min_x, min_y, w, h)
    samples[:,:2] -= samples[:,2:]/2

    return samples




def iou_score(bboxes, target):
    """
    Parameters:
    bboxes is numpy array of shape = (N, 4)
    which is [x, y, w, h]
    
    Returns:
    target is numpy array of shape (4, )
    """
    areas = bboxes[:, 2] * bboxes[:, 3]    # (N,) array
    a = target[2] * target[3] 
    
    
    # convert bboxess to [xmin, ymin, xmax, ymax]
    bboxes = np.copy(bboxes)
    target = np.copy(target)
    bboxes[:,2] = bboxes[:,2] + bboxes[:, 0]
    bboxes[:,3] = bboxes[:,3] + bboxes[:, 1]
    target[2] = target[2] + target[0]
    target[3] = target[3] + target[1]
       
    inter_rows = np.maximum(np.minimum(bboxes[:, 3], target[3]) - np.maximum(bboxes[:,1], target[1]), 0)
    inter_cols = np.maximum(np.minimum(bboxes[:, 2], target[2]) - np.maximum(bboxes[:,0], target[0]), 0)
    intersection = inter_rows * inter_cols  # shape (N,)
    union = areas + a - intersection # shape (N,)
    return 1.0 * intersection / union    

def extract_regions(image, boxes, final_dim):
    """
    Params:
        image is a numpy 3D array of the image
        
        boxes are all boxes regions [x y w h]
        final dim is the input dimension of the model i.e. 107
    
    Returns:
        numpy array of shape(number of boxes, final_dim, final_dim ,3)
    """
    image_obj = Image.fromarray(image)
    regions = []
    for (i, box) in enumerate (boxes):
        regions.append(np.array(image_obj.resize((final_dim, final_dim), Image.BILINEAR)))
    return np.array(regions)
        



#def gen_samples(method, bb, n, imgSize, scale_factor, trans_f = None, scale_f=None):
#    h= imgSize[0]
#    w= imgSize[1]
#    
#    # bb is numpy array [x, y, box-width, box-height]
#    sample = [bb[0]+bb[2]/2, bb[1]+bb[3]/2, bb[2], bb[3]] #[center_x center_y width height]
#    sample = np.array(sample)
#    samples = np.tile(sample, (n,1))
#    if method == 'gaussian':
#        temp = np.rint(np.mean(bb[2:])) * np.maximum(-1, np.minimum(1, (0.5*np.random.normal(size = (n,2)))))
#        samples[:,0:2]= samples[:,0:2] + trans_f * temp
#        temp = np.power(scale_factor, (scale_f* np.maximum(-1, np.minimum(1, (0.5*np.random.normal(size = (n,1)))))))
#        samples[:,2:4] = samples[:,2:4] * np.tile(temp, (1,2))
#        
#    
#    if method == 'uniform':
#        temp = np.rint(np.mean(bb[2:])) * np.random.uniform(low= -1, high = 1, size = (n,2))
#        samples[:,0:2]= samples[:,0:2] + trans_f * temp   
#        temp = np.power(scale_factor, (scale_f* np.random.uniform(low= -1, high = 1, size = (n,1)))) 
#        samples[:,2:4] = samples[:,2:4] * np.tile(temp, (1,2))
#        
#    if method == 'uniform_aspect':
#        temp = samples[:, 2:4] * np.random.uniform(low= -1, high = 1, size = (n,2))      
#        samples[:,0:2]= samples[:,0:2] + trans_f * temp   
#        samples[:,2:4] = samples[:,2:4] * np.power(scale_factor, np.random.uniform(low= -2, high = 2, size = (n,2)))         
#        temp = np.power(scale_factor, (scale_f* np.random.uniform(low= 0, high = 1, size = (n,1)))) 
#        samples[:,2:4] = samples[:,2:4] * np.tile(temp, (1,2))
#        
#        
#    if method == 'whole':
#        m = int(4*np.sqrt(n))
#        xy = np.dstack(np.meshgrid(np.linspace(0,1,m),np.linspace(0,1,m))).reshape(-1,2)
#        xy = np.random.permutation(xy)[:n]
#        samples[:,:2] = bb[2:]/2 + xy * ((w,h)-bb[2:]/2-1)
#        #samples[:,:2] = bb[2:]/2 + np.random.rand(n,2) * (self.img_size-bb[2:]/2-1)
#        samples[:,2:] *= scale_factor ** (np.random.rand(n,1)*2-1)
#        
#        
##        samples = []
##        for i in range(n):
##            portion_w = bb[2]*scale_factor**random.randint(-5,5)
##            portion_h = bb[3]*scale_factor**random.randint(-5,5)
##            minxc = random.randint(portion_w//2,max(portion_w//2+1,int(w-portion_w-1)))
##            minyc = random.randint(portion_h//2,max(portion_h//2+1,int(h-portion_h-1)))
##            samples.append([minxc, minyc, portion_w, portion_h])
##        samples = np.array(samples)    
#        
#    samples[:,2] = np.maximum(10, np.minimum(w-10, samples[:,2]))    
#    samples[:,3] = np.maximum(10, np.minimum(h-10, samples[:,3])) 
#    
#    
#    #convert back to [x, y, box-width, box-height]
#    bb_samples = np.copy(samples)
#    bb_samples[:,0] = bb_samples[:,0] - bb_samples[:,2]/2 
#    bb_samples[:,1] = bb_samples[:,1] - bb_samples[:,3]/2 
#    bb_samples[:,0] = np.maximum(1- bb_samples[:,2]/2, np.minimum(w-bb_samples[:,2]/2, bb_samples[:,0]))
#    bb_samples[:,1] = np.maximum(1- bb_samples[:,3]/2, np.minimum(h-bb_samples[:,3]/2, bb_samples[:,1]))
#    
#    bb_samples = np.rint(bb_samples)
#    return bb_samples
