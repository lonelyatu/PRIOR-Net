# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:41:26 2017
主函数
@author: hudada
"""
import os
import glob
import numpy as np
import tensorflow as tf

import model_PriorNet as models_dense
import TFRecordOp as tfrecordOp
import HddDataOperation as hdd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#reload(module)
DataSize = 1200
batchSize = 2
row = 512
column = 512
channel = 1
num_epoch = 50
Iter = DataSize // batchSize  

low_holder = tf.placeholder(tf.float32,[None,row,row,channel])
prior_holder = tf.placeholder(tf.float32,[None,row,row,channel])
label_holder = tf.placeholder(tf.float32,[None,row,row,channel])
train_holder = tf.placeholder(tf.bool)

learn_rate = tf.placeholder(tf.float32)

result = models_dense.Dense2D(low_holder, prior_holder, train_sign=train_holder)
lossl2 = hdd.lossL2(result,label_holder)

loss = lossl2 

trainlossl2 = tf.summary.scalar('train_lossl2',lossl2)

merge_loss = tf.summary.merge([trainlossl2])

lossvalidl2 = tf.placeholder(tf.float32)
valid_lossl2 = tf.summary.scalar('valid_lossl2',lossvalidl2)
merge_validloss  = tf.summary.merge([valid_lossl2])

optimizer = tf.train.AdamOptimizer(learn_rate)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.minimize(loss)
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
Saver = tf.train.Saver(max_to_keep=500)

check_dir = './PriorNet/checkpoints/'    

writer = tf.summary.FileWriter('./PriorNet/logdir/',sess.graph)
  
filenames = tf.train.match_filenames_once('./TFRecordsFile/*.tfrecord')
#    
img_batch, label_batch, prior_batch = tfrecordOp.input_pipeline(filenames, batch_size=batchSize,
        num_epochs=None, num_features_input=[row,column,channel],
        num_features_label=[row,column,channel],num_features_prior=[row,column,channel])

if not os.path.exists(check_dir):
    os.makedirs(check_dir)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

else : 
    chkpt_fname = tf.train.latest_checkpoint(check_dir)
    Saver.restore(sess, chkpt_fname) 
    sess.run(tf.local_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

try:
    for epoch in range(0,num_epoch):  
        if epoch < 0:	
#            curr_lr = 1e-3
            curr_lr = 1e-3 * (1.0 - epoch * 0.1)
        elif epoch < 20:
            curr_lr = 1e-4
        elif epoch < 40:
            curr_lr = 1e-5
        else:
            curr_lr = 1e-6
        for iter in range(Iter):
            samp_low, samp_label, samp_prior = sess.run([img_batch, label_batch, prior_batch])
			
            mlossl2,_,mergeloss =\
                          sess.run([lossl2,train,merge_loss],
                            feed_dict={low_holder:samp_low,label_holder:samp_label,prior_holder:samp_prior, learn_rate:curr_lr, train_holder:True})
            if (iter % 100==0):
                print("epoch: %3d Iter %7d lossl2: %10.5f" 
                      % (epoch,iter,mlossl2))
            if (iter % 100 == 0):
                writer.add_summary(mergeloss,epoch*Iter + iter)
        Saver.save(sess, os.path.join(check_dir,"PriorNet"), global_step = epoch)
        
        validlossl2 = 0
        for pNo in range(10):
            for v in range(106):
                valid_input = np.load('./../test/Sparse/phase_%d/sparse_%03d.npy' % (pNo+1,v))
                valid_input = np.reshape(valid_input,[1,512,512,channel])
                valid_label = np.load('./../test/Label/phase_%d/label_%03d.npy' % (pNo+1,v))
                valid_label = np.reshape(valid_label,[1,512,512,channel])
                valid_prior = np.load('./../test/Prior/prior_%03d.npy' % v)
                valid_prior = np.reshape(valid_prior, [1,512,512,channel])
#                valid_label[valid_label>0] = 
                valid_loss = sess.run(lossl2,feed_dict={low_holder:valid_input,label_holder:valid_label,prior_holder:valid_prior,train_holder:False})
                validlossl2 = validlossl2 + valid_loss
#            print(valid_loss)
        
        validlossl2 = validlossl2 / 1060
        print(('Epoch %d the average validl2: %.5f') % 
              (epoch,validlossl2))
        validLoss = sess.run(merge_validloss,feed_dict=
                             {lossvalidl2:validlossl2})
        writer.add_summary(validLoss,epoch)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()
    coord.join(threads)    
writer.close()












