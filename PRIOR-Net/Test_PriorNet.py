# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import model_PriorNet as models_dense
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

row = 512
column = 512
channel = 1

low_holder = tf.placeholder(tf.float32,[None,row,column,channel])
prior_holder = tf.placeholder(tf.float32,[None,row,column,channel])
label_holder =  tf.placeholder(tf.float32,[None,row,column,channel])
result1 = models_dense.Dense2D(low_holder,prior_holder,False)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver = tf.train.Saver(max_to_keep=500)

chkpt_fname = './PriorNet/checkpoints/PriorNet-44'

saver.restore(sess, chkpt_fname)

for pNo in range(10):
    os.makedirs('./TestPrior/phase_%d' % (pNo+1))
    for v in range(106):
        print(v)
        valid_input = np.load('./../test/Sparse/phase_%d/sparse_%03d.npy' % (pNo+1,v))
        valid_input = np.reshape(valid_input,[1,512,512,channel])
        valid_label = np.load('./../test/Label/phase_%d/label_%03d.npy' % (pNo+1,v))
        valid_label = np.reshape(valid_label,[1,512,512,channel])
        valid_prior = np.load('./../test/Prior/prior_%03d.npy' % v)
        valid_prior = np.reshape(valid_prior, [1,512,512,channel]) 
        out = sess.run(result1,
                            feed_dict={low_holder:valid_input,
                                       prior_holder:valid_prior,
                                       label_holder:valid_label})
        np.save('./TestPrior/phase_%d/out_%03d.npy' % (pNo+1,v), out[0,:,:,0])
        