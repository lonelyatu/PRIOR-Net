# -*- coding: utf-8 -*-

import os
import glob
import time
import numpy as np
from scipy import io
import tensorflow as tf

import HddDataOperation as hdd

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def npy2tfrecord():
    
    if not os.path.exists('./TFRecordsFile'):
        os.mkdir('./TFRecordsFile')
    for pNo in range(10):
        name = './TFRecordsFile/train_%d.tfrecord' % pNo
        writer = tf.python_io.TFRecordWriter(name)
        for ind in range(120):
            SparseImg = np.load('./../train/Sparse/phase_%d/sparse_%03d.npy' % (pNo+1,ind))
            LabelImg  = np.load('./../train/Label/phase_%d/label_%03d.npy' % (pNo+1,ind))
            PriorImg  = np.load('./../train/Prior/prior_%03d.npy' % ind)
                
            img_raw = SparseImg.tostring()
            lab_raw = LabelImg.tostring()
            pri_raw = PriorImg.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(img_raw),        
            'res': _bytes_feature(lab_raw),
            'prior': _bytes_feature(pri_raw)
            }))
            writer.write(example.SerializeToString())
				
        writer.close()

def read_and_decode(filename_queue, shape_input,shape_label,shape_prior):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'res': tf.FixedLenFeature([], tf.string),
                'prior': tf.FixedLenFeature([], tf.string)
                })

    image = tf.decode_raw(features['image'], tf.float32, little_endian=True)
    image = tf.reshape(image,[shape_input[0],shape_input[1],shape_input[2]])
    
    res = tf.decode_raw(features['res'], tf.float32, little_endian=True)
    res = tf.reshape(res,[shape_label[0],shape_label[1],shape_label[2]])
    
    prior = tf.decode_raw(features['prior'], tf.float32, little_endian=True)
    prior = tf.reshape(prior,[shape_prior[0],shape_prior[1],shape_prior[2]])
    return image, res, prior


def input_pipeline(filenames, batch_size, num_epochs=None, 
                   num_features_input=None,num_features_label=None,num_features_prior=None):
    '''num_features := width * height for 2D image'''
    filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
    example, label, prior = read_and_decode(filename_queue, shape_input=num_features_input,
                                     shape_label=num_features_label,shape_prior=num_features_prior)
#    label = read_and_decode(filename_queue, num_features)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 2000 // 10
    capacity = min_after_dequeue + 10 * batch_size 
    img_batch, res_batch, prior_batch = tf.train.shuffle_batch(
            [example, label, prior], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue,num_threads=64,allow_smaller_final_batch=True)
    return img_batch, res_batch, prior_batch

if __name__ == '__main__':
    npy2tfrecord()
    print("convert finishes")
    #read_test()







