import tensorflow as tf
from keras.layers import (Conv3D, BatchNormalization, Activation, Input, add, GlobalAvgPool3D, 
                                                                            Dense,MaxPooling3D, concatenate)
from keras.regularizers import l2 as l2_penality
from keras.models import Model

from utils.losses import fmeasure, precision, recall, iou

def identity_resnet_A(x, filters, weights_decay=0., kernel_initializer='he_uniform', use_bias=False, bottleneck=2, activation=lambda: Activation('relu')):
    #channel1
    conv_channel1 = Conv3D(filters//bottleneck, kernel_size=(1,1,1), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(x)
    conv_channel1 = BatchNormalization(scale=False, axis=-1)(conv_channel1)
    conv_channel1 = activation()(conv_channel1)
    #channel2
    conv_channel2 = Conv3D(filters//bottleneck, kernel_size=(1,1,1), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(x)
    conv_channel2 = BatchNormalization(scale=False, axis=-1)(conv_channel2)
    conv_channel2 = activation()(conv_channel2)
    
    conv_channel2 = Conv3D(filters//bottleneck, kernel_size=(3,3,3), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(conv_channel2)
    conv_channel2 = BatchNormalization(scale=False, axis=-1)(conv_channel2)
    conv_channel2 = activation()(conv_channel2)
    #channel3
    conv_channel3 = Conv3D(filters//bottleneck, kernel_size=(1,1,1), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(x)
    conv_channel3 = BatchNormalization(scale=False, axis=-1)(conv_channel3)
    conv_channel3 = activation()(conv_channel3)
    
    conv_channel3 = Conv3D(filters//bottleneck, kernel_size=(3,3,3), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(conv_channel3)
    conv_channel3 = BatchNormalization(scale=False, axis=-1)(conv_channel3)
    conv_channel3 = activation()(conv_channel3)
    
    conv_channel3 = Conv3D(filters//bottleneck, kernel_size=(3,3,3), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(conv_channel3)
    conv_channel3 = BatchNormalization(scale=False, axis=-1)(conv_channel3)
    conv_channel3 = activation()(conv_channel3)
    
    concat1 = concatenate([conv_channel1,conv_channel2,conv_channel3],axis=-1)
    
    conv_channel123 = Conv3D(filters, kernel_size=(1,1,1), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(concat1)
    conv_channel123 = BatchNormalization(scale=False, axis=-1)(conv_channel123)
    
    residual = add([x, conv_channel123])
    relu = activation()(residual)
    return relu

def Reduction_A(x, filters, weights_decay=0., kernel_initializer='he_uniform', use_bias=False, bottleneck=2, activation=lambda: Activation('relu')):
    #down1
    down1 = Conv3D(filters//bottleneck, kernel_size=(1,1,1), strides=(2,2,2), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(x)
    down1 = BatchNormalization(scale=False, axis=-1)(down1)
    down1 = activation()(down1)
    #down2
    down2 = MaxPooling3D((3, 3, 3), strides=(2, 2, 2),padding='same')(x)
    #down3
    down3 = Conv3D(filters//bottleneck, kernel_size=(1,1,1), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(x)
    down3 = BatchNormalization(scale=False, axis=-1)(down3)
    down3 = activation()(down3)
    
    down3 = Conv3D(filters//bottleneck, kernel_size=(3,3,3), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(down3)
    down3 = BatchNormalization(scale=False, axis=-1)(down3)
    down3 = activation()(down3)
    
    down3 = Conv3D(filters, kernel_size=(3,3,3), strides=(2,2,2), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(down3)
    down3 = BatchNormalization(scale=False, axis=-1)(down3)
    down3 = activation()(down3)
    
    down_mixed = concatenate([down1,down2,down3],axis=-1)
    
    down_mixed = Conv3D(filters, kernel_size=(1,1,1), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(down_mixed)
    down_mixed = BatchNormalization(scale=False, axis=-1)(down_mixed)
    down_mixed = activation()(down_mixed)
    return down_mixed

def get_model(dhw=[48,48,48],weights_decay = 0.,kernel_initializer='he_uniform',weights=None, deeper=False, activation=lambda: Activation('relu')):
    shape = dhw+[1]
    
    inputs = Input(shape = shape)
    conv1 = Conv3D(32, kernel_size=(3, 3, 3), padding='same', use_bias=False, kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(inputs)
    
    id1 = identity_resnet_A(conv1, filters=32, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                    use_bias=False, bottleneck=2,activation=activation)

    down1 = Reduction_A(id1, filters=64, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
               use_bias=False, bottleneck=2,activation=activation)
    
    id2 = identity_resnet_A(down1, filters=64, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                    use_bias=False, bottleneck=2,activation=activation)

    down2 = Reduction_A(id2, filters=128, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
               use_bias=False, bottleneck=2,activation=activation)
    
    id3 = identity_resnet_A(down2, filters=128, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                    use_bias=False, bottleneck=2,activation=activation)

    down3 = Reduction_A(id3, filters=256, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
               use_bias=False, bottleneck=2,activation=activation)
    
       
    pool = GlobalAvgPool3D()(down3)
    outputs = Dense(1, kernel_regularizer=l2_penality(weights_decay), kernel_initializer=kernel_initializer, activation='sigmoid')(pool)
    model = Model(inputs, outputs)
    if weights is not None:
        model.load_weights(weights, by_name=True)
    return model

def get_compiled(dhw, loss, optimizer,weights_decay = 0.,
                 kernel_initializer='he_uniform',weights=None, 
                 activation=lambda: Activation('relu'), deeper=False):
    model = get_model(dhw=dhw,weights_decay = weights_decay,
                  kernel_initializer=kernel_initializer,
                  weights=weights, deeper=deeper, activation=activation)
    model.compile(loss=loss, optimizer=optimizer,
              metrics=[loss, 'accuracy', fmeasure, precision, recall])
    return model


if __name__=='__main__':
    model = get_model()
    model.summary()
