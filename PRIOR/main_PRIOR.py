#-*- coding: utf-8 -*-

import os
import glob
import numpy as np
from scipy import io
import tensorflow as tf
import CNNHelper
import tigre
import tigre.algorithms as algs
import warnings
from tigre.utilities import sample_loader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings('ignore')

def ConeBeamGeoParaSetting(sample_num=60, scan_range=360):
    geo = tigre.geometry(mode='cone', default=True)
    
    geo.DSD = 1500 # mm src-to-det
    geo.DSO = 1000 # mm  src-to-obj
 
    geo.nVoxel = np.array([106, 512, 512])
    geo.dVoxel = np.array([0.9, 0.9, 0.9])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    
    geo.nDetector = np.array([400, 900])
    geo.dDetector = np.array([1.5, 1.5])
    geo.sDetector = geo.nDetector * geo.dDetector
    
    geo.angles = np.linspace(0, (1-1/(sample_num*scan_range/360))*2*np.pi*scan_range/360, 
                             sample_num*scan_range/360, 
                             dtype=np.float32)
    geo.mode = 'cone'
    geo.accuracy = 0.5
    geo.offDetector = np.array([0, 0])
    geo.offOrigin = np.array([0, 0, 0])
    
    geo.filter = 'ram_lak'
    
    return geo

def readProj():
    Proj = np.zeros([10,60,400,900], dtype=np.float32)
    for pNo in range(10):
        for v in range(60):
            img = np.load('./../test/Proj/phase_%d/proj_%03d.npy' % (pNo+1,v))
            Proj[pNo,v] = img
    return Proj

def readPriorImg():
    Prior = np.zeros([512,512,106], dtype=np.float32)
    for v in range(106):
        Prior[:,:,v] = np.load('./../test/Prior/prior_%03d.npy' % v)
    return Prior

def readSparseImg():
    Sparse = np.zeros([10,512,512,106],dtype=np.float32)
    for pNo in range(10):
        for v in range(106):
            img = np.load('./../test/Sparse/phase_%d/sparse_%03d.npy' % (pNo+1,v))
            Sparse[pNo,:,:,v] = img
    return Sparse

def readLabelImg():
    Label = np.zeros([10,512,512,106],dtype=np.float32)
    for pNo in range(10):
        for v in range(106):
            img = np.load('./../test/Label/phase_%d/label_%03d.npy' % (pNo+1,v))
            Label[pNo,:,:,v] = img
    return Label    

def sino2Img(Proj, xk, geo):
    image1 = np.transpose(sample_loader.load_head_phantom(np.array([106,512,512])), [1,2,0])
    image1[:,:,:]= xk
    image1 = np.transpose(np.rot90(np.fliplr(image1)), [2,0,1])

    Proj1 = tigre.Ax(image1, geo, geo.angles)
    ProjDiff = Proj - Proj1

    return algs.fdk(ProjDiff, geo, geo.angles).transpose([1,2,0])

def val_fn_s2(validlow, prior):
    out = CNNHelper.DenseS2Helper(validlow, prior)
    return out

geo = ConeBeamGeoParaSetting()

fileName = './../test/'

proj = readProj()
prior = readPriorImg()
input = readSparseImg()
label = readLabelImg()

Iter = 50
lambd = 15.


for phase in range(9,-1,-1):
    for iters in range(Iter):
        if iters == 0:
            ref_P = proj[phase]
            sparse  = input[phase]
            
            maxval = sparse.max()
            maxvalPrior = prior.max()
            PriorNet = val_fn_s2(sparse, prior)
            print('iter %d' % iters)
            print(np.square(PriorNet-label[phase]).mean())
            PRIOR = PriorNet.copy()

        else:
            gk = sino2Img(ref_P, PRIOR, geo)
            maxV = gk.max()
            minV = gk.min()
            
            gk = (gk - minV) / (maxV - minV) * maxval

            mk = PRIOR - prior
            ma = mk.max()
            mi = mk.min()
            mk = (mk - mi) / (ma - mi) * maxvalPrior

            sk = val_fn_s2(gk, mk)

            sk = sk / maxval * (maxV - minV) + minV
        
            PRIOR = PRIOR + 1/lambd * sk
            PRIOR[PRIOR<0] = 0
            print('iter %d' % iters)
            print(np.square(PRIOR-label[phase]).mean())

    io.savemat('./PRIOR_phase-%d.mat' % (phase+1), {'PriorNet':PriorNet,'PRIOR':PRIOR})
