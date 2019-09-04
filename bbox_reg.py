import numpy as np
from sklearn.linear_model import Ridge

class BBoxRegressor():
    def __init__(self):
        self.model = Ridge(alpha=1000)
                        
    def prepare_targets(self, boxes, gt):
        """
        boxes is a numpy array each box is [topx, topy, w, h]
        gt is list [topx, topy, w, h]
        
        returns: Y--> scale-invariant and log-space box paarams
        """
        boxes = np.copy(boxes)
        gt = np.copy(gt)
        boxes[:,:2] = boxes[:,:2] + boxes[:,2:]/2
        gt[:2] = gt[:2] + gt[2:]/2
        
        d_xy = (gt[:2] - boxes[:,:2]) / boxes[:,2:]
        log_wh = np.log(gt[2:] / boxes[:,2:])
        
        Y = np.concatenate((d_xy, log_wh), axis=1)
        return Y
    
    def train(self, X, boxes, gt):
        """
        X is a numpy (n, 4608) array
        boxes is a numpy array each box is [topx, topy, w, h]
        gt is list [topx, topy, w, h]
        """
        Y= self.prepare_targets(boxes, gt)     
        self.model.fit(X, Y)
    
     
    def predict(self, X, boxes):
        """
        X is a numpy (5, 4608) array
        boxes is a numpy array (5, 4), each is  [topx, topy, w, h]   

        returns reg_box        
        """        
        p = self.model.predict(X)  #(5, 4)  
        # use predictions to correct box
        boxes = np.copy(boxes)
        boxes[:, :2] = boxes[:, :2] + boxes[:, 2:]/2  # top to center 
        boxes[:, :2] = boxes[:, :2] + p[:, :2]*boxes[:, 2:]  # new center
        boxes[:, :2] = boxes[:, :2] - boxes[:, 2:]/2 # center to top
        boxes[:, 2:] = np.exp(p[:, 2:])*boxes[:, 2:] 
        box = np.mean(boxes, axis = 0)
        box=np.rint(box)
        
        return box
        
        
        
        