import numpy as np
import os
import skimage
import skimage.io
from random import shuffle
import tensorflow as tf

from tensorflow.keras import backend as K
#from keras import backend as K
from glob import glob
from tensorflow.keras import optimizers
from imgaug import augmenters as iaa
import imgaug as ia
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler
import imageio
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import binary_crossentropy
import sys
 
import cv2
from HRUNet import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from tensorflow.keras.preprocessing.image import  array_to_img,img_to_array

def preprocess_input(x):#BGR
    #x = skimage.color.rgb2gray(x) 
    x = (x - np.mean(x)) / np.std(x)
    return x

def dice_coe(y_true,output,axis=[1,2], smooth=1e-10):
    inse = tf.reduce_sum(output * y_true, axis=axis)
    l = tf.reduce_sum(output * output, axis=axis)
    r = tf.reduce_sum(y_true * y_true, axis=axis)
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)

    return dice  
def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    d_loss = tf.reduce_sum(0.5*(1-dice_coe(y_true[:,:,:,0], y_pred[:,:,:,0]))+\
                           0.5*(1-dice_coe(y_true[:,:,:,1], y_pred[:,:,:,1])))
    sq=tf.reduce_mean((tf.square(y_true-y_pred))) 
                           
    return 1*d_loss
        
def dice_m(target,output,axis=[1,2], smooth=1e-10):
    target = tf.cast(tf.argmax(target,3),dtype=tf.float32)
    output = tf.cast(tf.argmax(output,3),dtype=tf.float32)
    inse = tf.reduce_sum(output * target, axis=axis)
    l = tf.reduce_sum(output * output, axis=axis)
    r = tf.reduce_sum(target * target, axis=axis)
    dice = tf.divide(tf.multiply(2.,inse),tf.add(l,r))
    dice = tf.reduce_mean(dice)
    return dice
def nameb(path):
    return path.split('/')[-1].split('.')[0]

testimage='./testimage/'
listf=glob(testimage+'*.jpg')
testlengh=len(listf)

testmatrix=np.ndarray([testlengh,320,512,2])
testmatrix[matrix!=0]=0

for it in range(0,10):
	print(fold_test)
	print(id_img_test)
	model=HRUNet(2,320,512,3)
    
	## testing  
	model.load_weights('./result/'+ r'testpyc'+str(fold_test)+r'.h5', by_name=True)

	path_save = r'./result/test/'
	isExists=os.path.exists(path_save)	
	if not isExists:
		os.makedirs(path_save)
		
	for i in range(testlengh):

		x =skimage.io.imread(listf[i])
		x = preprocess_input(x)
		x = np.expand_dims(x,0)
		preds = model.predict(x)
		testmatrix[i]+=preds[0]

for i in range(testlengh):
		tlb=np.argmax(matrix5[i]/10,-1)*255
        
		cv2.imwrite(path_save+nameb(i)+'.png',tlb)            



     
