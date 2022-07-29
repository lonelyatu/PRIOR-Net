# -*- coding: utf-8 -*-

import tensorflow as tf 

def lossL2(tensor1,tensor2):
    return tf.reduce_mean(tf.square(tensor1 - tensor2))

def lossL1(tensor1,tensor2):
    return tf.reduce_mean(tf.abs(tensor1 - tensor2))
    
    
    
    
    
    
    
    
    
    