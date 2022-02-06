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

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from HRUNet import *
from tensorflow.keras.preprocessing.image import  array_to_img,img_to_array

def get_seq():#
    sometimes = lambda aug: iaa.Sometimes(0.3, aug)
    seq = iaa.Sequential([iaa.Fliplr(0.4),iaa.Flipud(0.3),sometimes(iaa.Crop()),sometimes(iaa.ElasticTransformation(alpha=90, sigma=9)),
                          sometimes(iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                rotate=(-45, 45),shear=(-15, 15),cval=0))])
    return seq    
    
def preprocess_input(x):#BGR
    x = (x - np.mean(x)) / np.std(x)
    return x
def scheduler(epoch):
	if epoch %50==0 and epoch!=0:
		lr=K.get_value(model.optimizer.lr)
		K.set_value(model.optimizer.lr,lr*0.1)
		print('lr changed {}'.format(lr*0.1))
		return (K.get_value(model.optimizer.lr)).astype('float32').item()
	return (K.get_value(model.optimizer.lr)).astype('float32').item()
      
def Filer(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))   

def lab_path(file_path):
    return  file_path.split('.')[0]+'.png'
        
def data_gen(list_files, batch_size, p_train='', p_test = '',augment=False):
    seq = get_seq()
    while True:
        shuffle(list_files)
        for batch in Filer(list_files, batch_size):
            X = [(skimage.io.imread(x)) for x in batch]
            Y = [np.expand_dims(cv2.imread(lab_path(x),2)//255,-1) for x in batch] 
            if augment:
                seq_det = seq.to_deterministic()
                X_aug = [seq_det.augment_image(x) for x in X]                
                Y_map = [ia.SegmentationMapOnImage(np.squeeze(y).astype(np.uint8), shape=(320,512,1), nb_classes=2) for y in Y]
                Y_aug = [seq_det.augment_segmentation_maps([y])[0].get_arr_int().astype(np.uint8) for y in Y_map]
                X = X_aug
                Y = Y_aug                        
            Y_hot =[to_categorical(y,2) for y in Y]#
             
            X = [preprocess_input(x) for x in X]
            X = np.array(X)
            Y_hot= np.array(Y_hot)#                  
            yield X,Y_hot


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
    return 1*d_loss
        
                                   
def dice_m(target,output,axis=[1,2], smooth=1e-10):
    #target = tf.cast(target, tf.float32)
    #output = tf.cast(output, tf.float32) 
    target = tf.cast(tf.argmax(target,3),dtype=tf.float32)
    output = tf.cast(tf.argmax(output,3),dtype=tf.float32)
    inse = tf.reduce_sum(output * target, axis=axis)
    l = tf.reduce_sum(output * output, axis=axis)
    r = tf.reduce_sum(target * target, axis=axis)
    dice = tf.divide(tf.multiply(2.,inse),tf.add(l,r))
    dice = tf.reduce_mean(dice)
    return dice

trainimage='./train/'
trainlab=''

allimg='240new'#
crpmg='80newcp'#

for it in range(0,10):
	id_img_train=[]
	id_img_test=[]
	fold_test=it+1
	print(fold_test)
	for i in range(0,24*it):
		id_img_train.append(trainimage+allimg+'/'+str(i+1)+r'.jpg')
	for i in range(24*(it+1),240):
		id_img_train.append(trainimage+allimg+'/'+str(i+1)+r'.jpg')
		
	for i in range(0,8*it):
		id_img_train.append(trainimage+crpmg+'/'+str(i+1)+r'.jpg')
	for i in range(8*(it+1),80):
		id_img_train.append(trainimage+crpmg+'/'+str(i+1)+r'.jpg')
		
	id_img_test.append(trainimage+allimg+'/'+str(24*it+1)+r'.jpg')
	id_img_test.append(trainimage+allimg+'/'+str(24*it+7)+r'.jpg')
	id_img_test.append(trainimage+allimg+'/'+str(24*it+13)+r'.jpg')
	id_img_test.append(trainimage+allimg+'/'+str(24*it+19)+r'.jpg')

	print(len(id_img_train))
	print(len(id_img_test))
	print(id_img_test)
	model=HRUNet(2,320,512,3)
	ada = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(loss=dice_loss, optimizer=ada, metrics=[dice_m])#[focal_loss(alpha=.25, gamma=2)]

	## training   
	batch_size =4#
	h5_path ='./result/'+ r'testpyc'+str(fold_test)+r'.h5'
	checkpoint = ModelCheckpoint(h5_path, monitor='val_dice_m', verbose=1, save_best_only=True, save_weights_only=True,mode='max')
	reduce_lr=LearningRateScheduler(scheduler)

	history = model.fit_generator(
		data_gen(id_img_train, batch_size,p_train=trainimage,p_test =trainlab,augment=False),
		validation_data=data_gen2(id_img_test, batch_size,p_train=trainimage,p_test =trainlab,augment=False),
		epochs=100, verbose=2,callbacks=[checkpoint],steps_per_epoch=len(id_img_train) // batch_size,validation_steps=1)
            
	## testing  
	model.load_weights('./result/'+ r'testpyc'+str(fold_test)+r'.h5', by_name=True)
	p_train=trainimage+allimg+'/'
	path_save = r'./result/'
	isExists=os.path.exists(path_save)	
	if not isExists:
		os.makedirs(path_save)

	id_img_test=[]
	id_img_test.append(str(24*it+1)+r'.jpg')
	id_img_test.append(str(24*it+7)+r'.jpg')
	id_img_test.append(str(24*it+13)+r'.jpg')
	id_img_test.append(str(24*it+19)+r'.jpg')
	id_img_test.sort(key=lambda x: int(x.split('.')[0]))
	
	 
	for i in range(4):
		#x = np.expand_dims(skimage.color.rgb2gray(skimage.io.imread(p_train+id_img_test[i])),-1)#[100:644,50:946]
		img=cv2.imread(p_train+id_img_test[i])
		x =skimage.io.imread(p_train+id_img_test[i])
		
		x = preprocess_input(x)
		x = np.expand_dims(x,0)
		preds = model.predict(x)
		lab = np.squeeze(np.argmax(preds,-1))
		cv2.imwrite(path_save+str(4*(fold_test-1)+i+1)+'.png',lab*255)
        
        
        
        
        

