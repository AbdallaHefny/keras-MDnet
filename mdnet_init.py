import os
from keras.models import Model
from keras.layers import Input
from mdnet_keras import create_model

   
def initialize_model():
    net = create_model()  #untrained
    net.load_weights(os.path.join(os.getcwd(), 'model_data', 'MD_tarined_weights.h5'))  # weights loaded till fc5 layer     
    conv_net = Model(net.input, net.get_layer('flatten').output)
    
    #3*3*512= 4608
    fc_in = Input(shape =(4608,))
    next_input = fc_in
    for layer in net.layers[-6:]:
        next_input = layer(next_input)
    fc_net = Model(fc_in, next_input)
        
    net_body = Model(net.input, net.layers[-2].get_output_at(0))
    return conv_net, fc_net, net_body

    