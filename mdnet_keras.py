import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Layer, Activation
from keras.models import Sequential


# Inter-channel LRN
class LocalResponseNormalization(Layer):

    def __init__(self, n=5, alpha=0.0001, beta=0.75, k=2, **kwargs):    
        self.n = n        
        self.alpha = alpha        
        self.beta = beta        
        self.k = k        
        super(LocalResponseNormalization, self).__init__(**kwargs)
    
    def build(self, input_shape):    
        self.shape = input_shape        
        super(LocalResponseNormalization, self).build(input_shape)
    
    def call(self, x, mask=None):        
        return tf.nn.local_response_normalization(x, depth_radius=5, 
                                                  bias=self.k,
                                                  alpha=self.alpha, 
                                                  beta=self.beta)
                                             
    def get_output_shape_for(self, input_shape):    
        return input_shape


def create_model():
    model =Sequential()
    #96
    model.add(Conv2D(96, (7,7), strides = 2, input_shape = (107,107,3), padding='valid', activation = 'relu', name = 'conv1'))
    # out shape (51,51,96)
    lrn = LocalResponseNormalization(input_shape = model.output_shape[1:], name = 'lrn1') 
    model.add(lrn)
    model.add(MaxPooling2D(3, strides = 2, name = 'pool1'))
    # out shape (25,25,96)
    model.add(Conv2D(256, (5,5), strides = 2, padding='valid', activation = 'relu', name = 'conv2'))
    # out shape (11,11,256)
    lrn = LocalResponseNormalization(input_shape = model.output_shape[1:], name = 'lrn2') 
    model.add(lrn)
    model.add(MaxPooling2D(3, strides = 2, name = 'pool2'))
    # out shape (5,5,256)
    model.add(Conv2D(512, (3,3), strides = 1, padding='valid', activation = 'relu', name = 'conv3'))
    # out shape (3,3,512)
    model.add(Flatten(name = 'flatten')) # 3*3*512 neurons = 4608
    
    model.add(Dense(512, activation = 'relu', name = 'fc4'))
    model.add(Dropout(rate = 0.5))
    
    model.add(Dense(512, activation = 'relu', name = 'fc5'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(2, name = 'fc6'))
    model.add(Activation('softmax'))
    return model



#  Inta Channel LRN
#class LocalResponseNormalization(Layer):
#
#    def __init__(self, n=5, alpha=0.0001, beta=0.75, k=2, **kwargs):    
#        self.n = n        
#        self.alpha = alpha        
#        self.beta = beta        
#        self.k = k        
#        super(LocalResponseNormalization, self).__init__(**kwargs)
#    
#    def build(self, input_shape):    
#        self.shape = input_shape        
#        super(LocalResponseNormalization, self).build(input_shape)
#    
#    def call(self, x, mask=None):   
#        if K.image_data_format == "channels_first":     #image_dim_ordering
#            _, f, r, c = self.shape       
#        else:        
#            _, r, c, f = self.shape          
#        squared = K.square(x)            
#        pooled = K.pool2d(squared, (self.n, self.n), strides=(1, 1),           
#        padding="same", pool_mode="avg")
#        
#        if K.image_data_format == "channels_first":        
#            summed = K.sum(pooled, axis=1, keepdims=True)            
#            averaged = self.alpha * K.repeat_elements(summed, f, axis=1)        
#        else:        
#            summed = K.sum(pooled, axis=3, keepdims=True)            
#            averaged = self.alpha * K.repeat_elements(summed, f, axis=3)            
#        denom = K.pow(self.k + averaged, self.beta)
#        
#        return x / denom
#    
#    
#    def get_output_shape_for(self, input_shape):    
#        return input_shape
