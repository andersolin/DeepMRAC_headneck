#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:08:56 2018

@author: andersolin
"""

import argparse, os, pickle, sys
import numpy as np
import progressbar
import pyminc.volumes.factory as pyminc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# Function for chopping the images into patches
def cutup(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    print('imgdata = %d/%d/%d' % (sh[2],sh[1],sh[0]))
    print('block = %d/%d/%d' % (blck[2],blck[1],blck[0]))
    print('strd = %d/%d/%d' % (strd[2],strd[1],strd[0]))
    print('nbl = %d/%d/%d' % (nbl[2],nbl[1],nbl[0]))     
    dims = np.r_[nbl, blck]
    data6 = np.lib.stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6


def get_patches(img,patch_shape,stride):
    patches_img = cutup(img,patch_shape,stride)
    ijk = patches_img.shape[0]*patches_img.shape[1]*patches_img.shape[2]
    selected_patches = np.empty((ijk,patch_shape[0],patch_shape[1],patch_shape[2],1), dtype='float32')
    selected_patches = np.reshape(patches_img,(ijk,patch_shape[0],patch_shape[1],patch_shape[2],1))
    selected_patches = selected_patches.astype('float32')
    return selected_patches


def load_minc_as_np(mncfile):
    print(mncfile)
    _img = pyminc.volumeFromFile(mncfile)
    img = np.array(_img.data,dtype='double')
    _img.closeVolume()
    return img

def model_predict_from_patches(model,patches,containersize,stride=(2,1,1)):

    (_,x,y,z,d) = model.input.shape.as_list()

    predicted_combined = np.zeros(containersize)
    predicted_counter = np.zeros(containersize)

    sh = np.array(containersize)
    blck = np.asanyarray((x,y,z))
    strd = np.asanyarray(stride)
    nbl = np.round((sh - blck) / strd + 1)
    print('imgdata = %d/%d/%d' % (sh[2],sh[1],sh[0]))
    print('block = %d/%d/%d' % (blck[2],blck[1],blck[0]))
    print('strd = %d/%d/%d' % (strd[2],strd[1],strd[0]))
    print('nbl = %d/%d/%d' % (nbl[2],nbl[1],nbl[0]))                
    count=np.asanyarray((0,0,0))
    bar = progressbar.ProgressBar(maxval=patches.shape[0], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' (', progressbar.ETA(), ') '])
    bar.start()
    for p in range(patches.shape[0]):
        bar.update(p+1)

        from_z = count[2]*stride[2]
        count[2]+=1
        from_y = count[1]*stride[1]
        from_x = count[0]*stride[0]

        if (count[2]) == int(nbl[2]):
            count[2] = 0
            count[1]+=1
        if (count[1]) == int(nbl[1]):
            count[1] = 0
            count[0]+=1
        
        a = np.reshape(patches[p,:,:,:,:d],(1,x,y,z,d))
    
        predicted = model.predict(a)
        predicted_combined[from_x:(from_x+x),from_y:(from_y+y),from_z:(from_z+z)] += np.reshape(predicted,(x,y,z))
        predicted_counter[from_x:(from_x+x),from_y:(from_y+y),from_z:(from_z+z)] += 1

    bar.finish()
    predicted_combined=np.divide(predicted_combined, predicted_counter, out=np.zeros_like(predicted_combined), where=predicted_counter!=0)
    return predicted_combined

def predict(model_name,input_path,input_filenames,output_path,output_filename,stride=(2,1,1)):

    #model = load_model_own(model_name,model_version=model_version)
    model = load_model(model_name)
    (_,x,y,z,d) = model.input.shape.as_list()
    patchsize = (x,y,z)

    l = []
    for n in range(0,d):
        mncfile = os.path.join(input_path,input_filenames[n])
        img = load_minc_as_np(mncfile)
        patches = get_patches(img,patchsize,stride)
        l.append(patches)
    patches = np.concatenate(l,axis=4)

    out_vol = pyminc.volumeLikeFile(os.path.join(input_path,input_filenames[0]),os.path.join(output_path,output_filename))
    containersize= out_vol.data.shape
    predicted_vol =  model_predict_from_patches(model,patches,containersize,stride=stride)
    out_vol.data = predicted_vol
    out_vol.writeFile()
    out_vol.closeVolume()
    print('Done predicting: ' + os.path.join(output_path,output_filename))


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Predict using DeepMRAC_headneck.')
    parser.add_argument("model", help="Path and name of the trained model")
    parser.add_argument("patient", help="Path to patient data.")
    parser.add_argument("in_phase", help="Name of Dixon in-phase image volume file (.mnc) (must be resampled and z-normalized).")
    parser.add_argument("opposed_phase", help="Name of Dixon opposed-phase image volume file (.mnc) (must be resampled and z-normalized).")
    parser.add_argument("outpath", help="Path to the output folder.")
    parser.add_argument("outname", help="Name for output file.")
    
    #parser.add_argument("--version", help="Software version used to train the model (VB20P or VE11P) Default: VE11P. ", type=str, default='VE11P')
    args = parser.parse_args()


    predict(args.model,args.patient,[args.in_phase,args.opposed_phase],args.outpath,args.outname)
