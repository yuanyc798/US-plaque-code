"""
@author: yyc
@time: 2020/12/21
import"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape, MaxPooling2D,MaxPool2D, concatenate, UpSampling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Dense,Reshape,multiply,Add,Lambda,GlobalMaxPooling2D,ZeroPadding2D,Conv2DTranspose
from tensorflow.keras.layers import SimpleRNN, Activation, Dense,LSTM,GRU,Dropout
from tensorflow.keras import backend as K
#from tensorflow.keras.layers import PReLU,LeakyReLU
#from tensorflow.keras.layers.advanced_activations import PReLU,LeakyReLU
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import regularizers
import tensorflow as tf


def cc(x, dims):  
        pconv1_c = Conv2D(dims,(3,3),strides=(1,1),padding='same',data_format='channels_last')(x)
        #pconv1_c = Conv2D(dims,(3,3),strides=(1,1),padding='same',data_format='channels_last')(x)
        pconv1_b = BatchNormalization()(pconv1_c)#BatchNormalization
        pconv1_a = Activation('relu')(pconv1_b)#Activation('Leaky relu')(pconv1_b)
        conv1_c = Conv2D(dims,(3,3),strides=(1,1),padding='same',data_format='channels_last')(pconv1_a)#           
        #conv1_c = Conv2D(dims,(3,3),strides=(1,1),padding='same',data_format='channels_last')(pconv1_a)
        conv1_b = BatchNormalization()(conv1_c)
        conv1_a = Activation('relu')(conv1_b)
        return conv1_a

def globalpool2(x,num):
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(num, (1, 1), padding='same',use_bias=False)(b4)
    b4 = BatchNormalization(epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = Lambda(lambda x: tf.image.resize_images(x, shape_before[1:3],method=0, align_corners=True))(b4)
    return b4 
   
def senet(xy):
    num=int(xy.shape[3])
    pconv1_c = Conv2D(num,(1,1),strides=(1,1),padding='same',data_format='channels_last')(xy)#,kernel_regularizer=regularizers.l2(0.0001)	
    pcon = BatchNormalization()(pconv1_c)
    
    squeeze = GlobalAveragePooling2D()(xy)
    squeeze2 =GlobalMaxPooling2D()(xy)
    CELL_SIZE =num  # 
    ll = Dense(num)
    lx =ll(squeeze) 
    ly =ll(squeeze2) 
    add = Add()([lx,ly])
    
    excitation = Dense(units=num //2)(squeeze)    
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=num)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, num))(excitation)
    print(xy.shape)
    print(excitation.shape)
    scale = multiply([xy, excitation])
    
    mm=Lambda(lambda x: K.mean(x, axis=-1))(scale) 
    mn=Lambda(lambda x: K.max(x, axis=-1))(scale) 
    mk=Lambda(lambda x: K.expand_dims(x,-1))(mm)
    nk=Lambda(lambda x: K.expand_dims(x,-1))(mn)
    pcn= Conv2D(num//2,(1,1),strides=(1,1),padding='same',data_format='channels_last')(scale)   
    pcn= Conv2D(1,(1,1),strides=(1,1),padding='same',data_format='channels_last')(pcn)  

    og= concatenate([mk,nk], axis=-1)
    #pc= Conv2D(1,(7,7),strides=(1,1),padding='same',data_format='channels_last',kernel_regularizer=regularizers.l2(0.0001))(og)
    pc= Conv2D(1,(5,5),strides=(1,1),padding='same',data_format='channels_last')(og)    
    pc= Activation("sigmoid")(pc)#tanh

    scale2= multiply([scale,pc])        
    return scale2
     
    
def attention(gate,xcon):
		num=int(xcon.shape[3])
		gate= Lambda(lambda x: tf.image.resize_images(x,(xcon.shape[1],xcon.shape[2]),method=0,align_corners=True))(gate)	
		xcon2=Conv2D(num,(1,1),strides=(1,1),padding='same',data_format='channels_last')(xcon)#tf.layers.conv2d(xcon2,num,1,strides=1,padding='same')   
		gateup=Conv2D(num,(1,1),strides=(1,1),padding='same',data_format='channels_last')(gate)#tf.layers.conv2d(gate,num,1,strides=1,padding='same')       

		plusc =Activation('relu')(Add()([gateup,xcon2]))
		plus2=Conv2D(1,(1,1),strides=(1,1),padding='same', data_format='channels_last')(plusc)
		softm=Activation('sigmoid')(plus2) 
		softm=Activation('softmax')(softm) 
		out=multiply([softm,xcon]) #gateup2*xcon      
		return xcon 
    
      
def ccdia3(x, dims):
        pconv1_c = Conv2D(dims,(3,3),strides=(1,1),padding='same', data_format='channels_last',dilation_rate=(1,1))(x)
        pconv1_b = BatchNormalization()(pconv1_c)
        pconv1_a = Activation('relu')(pconv1_b)
		
        conv1_c = Conv2D(dims,(3,3),strides=(1,1),padding='same', data_format='channels_last',dilation_rate=(2,2))(pconv1_a)
        conv1_b = BatchNormalization()(conv1_c)
        conv1_a = Activation('relu')(conv1_b)
		
        conv2_c = Conv2D(dims,(3,3),strides=(1,1),padding='same', data_format='channels_last',dilation_rate=(3,3))(conv1_a)
        conv2_b = BatchNormalization()(conv2_c)
        conv2_a = Activation('relu')(conv2_b)        
        return conv2_a

            
def HAC(x,numb):
	
        numb=int(numb/2)
        cona= Conv2D(numb,(3,3),strides=(1,1),padding='same',data_format='channels_last')(x)
        cona = BatchNormalization()(cona)
        cona = Activation('relu')(cona)  
       
        conb= Conv2D(numb,(3,3),strides=(1,1),padding='same',data_format='channels_last',dilation_rate=(5,5))(x)
        conb = BatchNormalization()(conb)
        conb = Activation('relu')(conb)
		
        conc= Conv2D(numb,(3,3),strides=(1,1),padding='same',data_format='channels_last',dilation_rate=(2,2))(x)
        conc = BatchNormalization()(conc)
        conc = Activation('relu')(conc)
		
        cond= Conv2D(numb,(3,3),strides=(1,1),padding='same',data_format='channels_last',dilation_rate=(3,3))(x)
        cond = BatchNormalization()(cond)
        cond = Activation('relu')(cond)  
             
        cone=ccdia3(x,numb)                    
        con= concatenate([cona,conb,conc,cond,cone],axis=-1)#12
        #con=cc(con,int(numb*2))                      
        return con        

def HRUNet(nClasses, input_height, input_width,chanele):
    #K.set_learning_phase(0)
    img_input = Input(shape=(input_height, input_width,chanele))
    path='./resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    conv_base = ResNet50(weights=path, include_top=False,input_shape=(input_height,input_width,chanele))#'imagenet'img_input (input_height,input_width,chanele)
    #conv_base = ResNet50(weights=None,include_top=False,input_shape=(input_height,input_width,chanele))
    i=0
    conv1 = cc(conv_base.input,16)

    conv2 =conv_base.layers[3].output# cc(pool1, 32)
    print(conv_base.layers[3].name,conv_base.layers[3].output.shape)
    conv2= Conv2D(32,(1,1),padding="same")(conv2)

    conv3 = conv_base.layers[36].output #cc(pool2, 32)
    conv3=ZeroPadding2D(((1, 0),(1, 0)))(conv3)
    print(conv_base.layers[36].name,conv_base.layers[36].output.shape)
    conv3= Conv2D(64,(1,1),padding="same")(conv3)

    conv4 =conv_base.layers[78].output  #cc(pool3, 64)
    print(conv_base.layers[78].name,conv_base.layers[78].output.shape) 
    conv4= Conv2D(96,(1,1),padding="same")(conv4)     
    conv4 =HAC(conv4,96) 

    conv5 = conv_base.layers[140].output #cc(pool4, 128)
    print(conv_base.layers[140].name,conv_base.layers[140].output.shape)
    conv5= Conv2D(128,(1,1),padding="same")(conv5)	       
    conv5=HAC(conv5,128)
    
           
    conv6 =conv_base.layers[172].output #cc(pool5, 128)
    print(conv_base.layers[172].name,conv_base.layers[172].output.shape)    
    conv6= Conv2D(128,(1,1),padding="same")(conv6)
    conv6=HAC(conv6,128)

    o = UpSampling2D((2, 2))(conv6)#Conv2DTranspose(256,(3,3),strides=(2,2),padding='same', data_format='channels_last')(conv6)#
    #o = concatenate([attention(conv6,conv5),o], axis=-1) 
    o = concatenate([conv5,o], axis=-1) 
    #o=spatten(o)       
    o1 =cc(o,128)#1024
   

    o2 = UpSampling2D((2, 2))(o1)#Conv2DTranspose(128,(3,3),strides=(2,2),padding='same', data_format='channels_last')(o)#
    #o2 = concatenate([attention(o1,conv4),o2], axis=-1) 
    o2 = concatenate([conv4,o2], axis=-1) 
    #o2 =spatten(o2)#spatten(o2)     
    o2 =cc(o2,96)#512

    o3 =UpSampling2D((2, 2))(o2)#Conv2DTranspose(96,(3,3),strides=(2,2),padding='same', data_format='channels_last')(o) #
    o3 = concatenate([(conv3),o3], axis=-1)
    #o3 = concatenate([attionsent(o3,conv3),o3], axis=-1)  
    o3 =cc(o3,64)#256

    o4 = UpSampling2D((2, 2))(o3)#Conv2DTranspose(64,(3,3),strides=(2,2),padding='same', data_format='channels_last')(o)#
    o4 = concatenate([(conv2),o4], axis=-1)
    #o4= concatenate([attionsent(o4,conv2),o4], axis=-1) 
    o4 =cc(o4,48)  

    o5 = UpSampling2D((2, 2))(o4)#Conv2DTranspose(48,(3,3),strides=(2,2),padding='same', data_format='channels_last')(o)#
    o5 = concatenate([(conv1),o5], axis=-1)
    #o5 = concatenate([attention(o4,conv1),o5], axis=-1)  
    om =cc(o5,32)

    o = Conv2D(nClasses, (1, 1), padding="same")(om)#,kernel_regularizer=regularizers.l2(0.0001)
    o = Activation("softmax")(o)
    print('xx:',o.shape)
    model = Model(inputs=conv_base.input, outputs=o)#img_input  conv_base.input
    return model

if __name__ == '__main__':
    m= HRUNet(2,320,512,3)
    print(m.summary())
