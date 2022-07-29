# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import io
import tensorflow as tf
from importlib import reload
from multiprocessing import Process, Queue, Pool
import model_PriorNet as models_dense 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def DenseS2Helper(inputs, prior):
    p = Process(target=Dense2DS2Helper, args=(inputs,prior))
    p.start()
    p.join()
    return io.loadmat('temp.mat')['temp']

def Dense2DS2Helper(inputs,prior):
    tf.reset_default_graph()
    low_holder = tf.placeholder(tf.float32,[None,None,None,1])
    prior_holder = tf.placeholder(tf.float32,[None,None,None,1])

    result1 = models_dense.Dense2D(low_holder,prior_holder,False)
#    result1 = models_dense.Dense2DConcat(tf.concat([low_holder,prior_holder],3), False)
    #result1 = resulthigh
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    
    chkpt_fname = './../PRIOR-Net/PriorNet/checkpoints/PriorNet-44'
    saver.restore(sess, chkpt_fname)
    for i in range(inputs.shape[-1]):
        low = np.reshape(inputs[:,:,i],[1,512,512,1])
        pri = np.reshape(prior[:,:,i],[1,512,512,1])
        out = sess.run(result1, feed_dict={low_holder:low, prior_holder:pri})
        inputs[:,:,i] = out[0,:,:,0]
        
    io.savemat('temp.mat',{'temp':inputs})
