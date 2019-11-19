#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:38:03 2018

@author: Jose M. Saavedra
In this file, different architectures are defined
"""

import tensorflow as tf
from . import layers
#%%
# A net for sketch classification, this is similar to AlexNet
# features: containing feature vectors to be trained
# input_shape: [height, width]
# n_classes int
# is_training: True for training and False for testing

def sketch_fn(features, input_shape, n_classes, n_channels, is_training = True):
    with tf.variable_scope("net_scope"):        
        #reshape input to fit a  4D tensor            
        x_tensor = tf.reshape(features, [-1, input_shape[0], input_shape[1],  n_channels ] )
        #conv_1   block#1
        conv_1 = layers.conv_layer(x_tensor, shape = [3,3, n_channels, 64],  stride =1, name='conv_1', is_training = is_training) #256s                        
        conv_1 = layers.max_pool_layer(conv_1, 3, 2) # 64x64         
        print(" conv_1: {} ".format(conv_1.get_shape().as_list()))
        
        #conv_2   block#2 
        conv_2 = layers.conv_layer(conv_1, shape = [3, 3, 64, 64], name = 'conv_2', is_training = is_training)        
        conv_2 = layers.max_pool_layer(conv_2, 3, 2) # 32x32
        print(" conv_2: {} ".format(conv_2.get_shape().as_list()))                
        
        #conv_3   block#3 
        conv_3 = layers.conv_layer(conv_2, shape = [3, 3, 64, 128], name = 'conv_3', is_training = is_training)        
        conv_3 = layers.max_pool_layer(conv_3, 3, 2) # 16x16
        print(" conv_2: {} ".format(conv_3.get_shape().as_list()))
                                
        #conv_4   block#4 
        conv_4 = layers.conv_layer(conv_3, shape = [3, 3, 128, 256], name = 'conv_3', is_training = is_training)        
        conv_4 = layers.max_pool_layer(conv_4, 3, 2) # 16x16
        print(" conv_2: {} ".format(conv_4.get_shape().as_list()))                                
        #This is the end of the convolutional process for extracting a feature map
    with tf.variable_scope("clasification_scope") :
        # fc 1
        fc1 = layers.fc_layer(conv_4, 1024, name = 'fc5')    
        #fc5 = layers.dropout_layer(fc5, 0.8)
        print(" fc1: {} ".format(fc1.get_shape().as_list()))        
        #fully connected
        fc2 = layers.fc_layer(fc1, n_classes, name = 'fc6', use_relu = False)
        print(" fc6: {} ".format(fc2.get_shape().as_list()))    
    return {"output": fc2, "deep_features": fc1}
    
def simple_fn(features, input_shape, n_classes, n_channels, is_training = True) :
    with tf.variable_scope("net_scope"):        
        #reshape input to fit a  4D tensor            
        x_tensor = tf.reshape(features, [-1, input_shape[0], input_shape[1],  n_channels ] )
        #conv_1   block#1
        conv_1 = layers.conv_layer(x_tensor, shape = [3,3, n_channels, 32],  stride =1, name='conv_1', is_training = is_training) #256s                        
        conv_1 = layers.max_pool_layer(conv_1, 3, 2) # 32x32         
        print(" conv_1: {} ".format(conv_1.get_shape().as_list()))
        
        #conv_2   block#2 
        conv_2 = layers.conv_layer(conv_1, shape = [3, 3, 32, 64], name = 'conv_2', is_training = is_training)        
        conv_2 = layers.max_pool_layer(conv_2, 3, 2) # 16x16
        print(" conv_2: {} ".format(conv_2.get_shape().as_list()))                
                                
        #This is the end of the convolutional process for extracting a feature map
    with tf.variable_scope("clasification_scope") :
        # fc 1
        fc1 = layers.fc_layer(conv_2, 256, name = 'fc5')    
        #fc5 = layers.dropout_layer(fc5, 0.8)
        print(" fc1: {} ".format(fc1.get_shape().as_list()))        
        #fully connected
        fc2 = layers.fc_layer(fc1, n_classes, name = 'fc6', use_relu = False)
        print(" fc6: {} ".format(fc2.get_shape().as_list()))    
    return {"output": fc2, "deep_features": fc1}


def digitnet_fn(features, input_shape, n_classes, n_channels, is_training = True) :
    with tf.variable_scope("net_scope"):        
        #reshape input to fit a  4D tensor            
        x_tensor = tf.reshape(features, [-1, input_shape[0], input_shape[1],  n_channels ] )
        #conv_1   block#1
        conv_1 = layers.conv_layer(x_tensor, shape = [3,3, n_channels, 64],  stride =1, name='conv_1', is_training = is_training) #256s                        
        conv_1 = layers.max_pool_layer(conv_1, 3, 2) # 32x32         
        print(" conv_1: {} ".format(conv_1.get_shape().as_list()))
        
        #conv_2   block#2 
        conv_2 = layers.conv_layer(conv_1, shape = [3, 3, 64, 128], name = 'conv_2', is_training = is_training)        
        conv_2 = layers.max_pool_layer(conv_2, 3, 2) # 16x16
        print(" conv_2: {} ".format(conv_2.get_shape().as_list()))
        
        #conv_3   block#3
        conv_3 = layers.conv_layer(conv_2, shape = [3, 3, 128, 256], name = 'conv_3', is_training = is_training)        
        conv_3 = layers.max_pool_layer(conv_3, 3, 2) # 8x8
        print(" conv_3: {} ".format(conv_3.get_shape().as_list()))
                                
        #This is the end of the convolutional process for extracting a feature map
    with tf.variable_scope("clasification_scope") :
        # fc 1
        fc5 = layers.fc_layer(conv_3, 256, name = 'fc5')    
        #fc5 = layers.dropout_layer(fc5, 0.8)
        print(" fc5: {} ".format(fc5.get_shape().as_list()))        
        #fully connected
        fc6 = layers.fc_layer(fc5, n_classes, name = 'fc6', use_relu = False)
        print(" fc6: {} ".format(fc6.get_shape().as_list()))    
    return {"output": fc6, "deep_features": fc5}
    

