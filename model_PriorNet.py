 # -*- coding: utf-8 -*-

"""
@author: dianlin@seu.edu.cn
"""

import tensorflow as tf

def batch_norm(input_,scope='BN',bn_train=True):
    return tf.contrib.layers.batch_norm(input_,scale=True,epsilon=1e-8,
                                        is_training=bn_train,scope=scope)

    
def conv2d(input, filters, kernel_size, name, strides = (1,1), paddings = 'same', dilation_rate=(1,1)):
    return tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, 
	                        strides=strides,padding=paddings, dilation_rate=(1, 1), 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            bias_initializer=tf.constant_initializer(0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                            name=name)

def conv2d_transpose(input,filters,kernel_size,name,paddings='same',strides=[2,2]):
    return tf.layers.conv2d_transpose(input,filters,kernel_size,strides=strides,padding=paddings,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                            bias_initializer=tf.constant_initializer(0.01),name=name)

def lrelu(inputs, slope=0):
    return tf.maximum(inputs, slope * inputs)	
		
def DenseBlock(input, num=64, name='dense', train_sign=True):
    with tf.variable_scope(name):
        if num != input.shape.as_list()[-1]:
            x = conv2d(input,num,[1,1],'b')
        else:
            x = input
        c1 = conv2d(input,num,[3,3],'c1')	
        b1 = batch_norm(c1,'b1',bn_train=train_sign)
        r1 = lrelu(b1)
        c2 = conv2d(r1,num,[3,3],'c2')
        b2 = batch_norm(c2,'b2',bn_train=train_sign)
        r2 = lrelu(b2)
        c3 = conv2d(tf.concat([r2,r1],3),num,[3,3],'c3')
        b3 = batch_norm(c3,'b3',bn_train=train_sign)
        r3 = lrelu(b3 + x)
        return r3				
		
def ResBlock(input, num=64, name='res', train_sign=True):
    with tf.variable_scope(name):
        if num != input.shape.as_list()[-1]:
            x = conv2d(input,num,[1,1],'b')
        else:
            x = input
        c1 = conv2d(input,num,[3,3],'c1')
        b1 = batch_norm(c1,'b1',bn_train=train_sign)
        r1 = lrelu(b1)
        c2 = conv2d(r1,num,[3,3],'c2')
        b2 = batch_norm(c2,'b2',bn_train=train_sign)
        r2 = lrelu(b2 + x)
        return r2

def PriorNet(inputs, prior, train_sign=True, reuse=None):
    Num = 24
    f1, f2, f3, f4, f5 = PriorNetPrior(prior, False)
    with tf.name_scope('PriorNet'):
        with tf.variable_scope('PriorNet',reuse=reuse):
            
            out1_conv = conv2d(inputs,Num,[3,3],'conv')
            out1 = tf.concat([f1, DenseBlock(out1_conv,Num,'dense1',train_sign)], 3)
            
            maxpool_1 = tf.nn.max_pool(out1,[1,2,2,1],[1,2,2,1],'VALID')
            out2 = tf.concat([f2, DenseBlock(maxpool_1,Num*2,'dense2',train_sign)], 3)
            
            maxpool_2 = tf.nn.max_pool(out2,[1,2,2,1],[1,2,2,1],'VALID')
            out3 = tf.concat([f3, DenseBlock(maxpool_2,Num*4,'dense3',train_sign)], 3)
			
            maxpool_3 = tf.nn.max_pool(out3,[1,2,2,1],[1,2,2,1],'VALID')
            out4 = tf.concat([f4, DenseBlock(maxpool_3,Num*4,'dense4',train_sign)], 3)		

            maxpool_4 = tf.nn.max_pool(out4,[1,2,2,1],[1,2,2,1],'VALID')
            out8 = tf.concat([f5, DenseBlock(maxpool_4,Num*8,'dense5',train_sign)], 3)            

            upsamp1 = conv2d_transpose(out8,Num*4,[3,3],'up1')
            concat1 = tf.concat([out4,upsamp1],3)
            
            out9 = ResBlock(concat1,Num*4,'res1',train_sign)			
			
            upsamp2 = conv2d_transpose(out9,Num*4,[3,3],'up2')
            concat2 = tf.concat([out3,upsamp2],3)
            
            out5 = ResBlock(concat2,Num*4,'res2',train_sign)
			
            upsamp3 = conv2d_transpose(out5,Num*2,[3,3],'up3')
            concat3 = tf.concat([out2,upsamp3],3)
            
            out6 = ResBlock(concat3,Num*2,'res3',train_sign)
            
            upsamp4 = conv2d_transpose(out6,Num,[3,3],'up4')
            concat4 = tf.concat([out1,upsamp4],3)
            
            out7 = ResBlock(concat4,Num,'res4',train_sign) 		
            result = conv2d(out7,1,[3,3],'conv20')
                        
    return tf.nn.relu(result+inputs)

def PriorNetPrior(inputs, train_sign=True, reuse=None):
    Num = 24
    with tf.name_scope('PriorNetPrior'):
        with tf.variable_scope('PriorNetPrior',reuse=reuse):
            
            out1_conv = conv2d(inputs,Num,[3,3],'conv')
            out1 = DenseBlock(out1_conv,Num,'dense1',train_sign)
            
            maxpool_1 = tf.nn.max_pool(out1,[1,2,2,1],[1,2,2,1],'VALID')
            out2 = DenseBlock(maxpool_1,Num*2,'dense2',train_sign)
            
            maxpool_2 = tf.nn.max_pool(out2,[1,2,2,1],[1,2,2,1],'VALID')
            out3 = DenseBlock(maxpool_2,Num*4,'dense3',train_sign)
			
            maxpool_3 = tf.nn.max_pool(out3,[1,2,2,1],[1,2,2,1],'VALID')
            out4 = DenseBlock(maxpool_3,Num*4,'dense4',train_sign)		

            maxpool_4 = tf.nn.max_pool(out4,[1,2,2,1],[1,2,2,1],'VALID')
            out8 = DenseBlock(maxpool_4,Num*8,'dense5',train_sign)            
                        
    return out1, out2, out3, out4, out8