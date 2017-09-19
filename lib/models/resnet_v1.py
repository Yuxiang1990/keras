import tensorflow as tf
from keras.layers import (Conv3D, BatchNormalization, Activation, Input, add, GlobalAvgPool3D, Dense)
from keras.regularizers import l2 as l2_penality
from keras.models import Model

from utils.losses import fmeasure, precision, recall, iou

def identity(x, filters, weights_decay=0., kernel_initializer='he_uniform', use_bias=False, bottleneck=2, activation=lambda: Activation('relu')):
    conv = Conv3D(filters//bottleneck, kernel_size=(1,1,1), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(x)
    bn = BatchNormalization(scale=False, axis=-1)(conv)
    conv = activation()(bn)
    conv = Conv3D(filters//bottleneck, kernel_size=(3,3,3), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(conv)
    bn = BatchNormalization(scale=False, axis=-1)(conv)
    conv = activation()(bn)
    conv = Conv3D(filters, kernel_size=(1,1,1), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(conv)
    bn = BatchNormalization(scale=False, axis=-1)(conv)
    residual = add([x, bn])
    relu = activation()(residual)
    return relu

def td(x, filters, weights_decay=0., kernel_initializer='he_uniform', use_bias=False, bottleneck=2, activation=lambda: Activation('relu')):
    conv = Conv3D(filters//bottleneck, kernel_size=(1,1,1), strides=(2,2,2), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(x)
    bn = BatchNormalization(scale=False, axis=-1)(conv)
    conv = activation()(bn)
    conv = Conv3D(filters//bottleneck, kernel_size=(3,3,3), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(conv)
    bn = BatchNormalization(scale=False, axis=-1)(conv)
    conv = activation()(bn)
    conv = Conv3D(filters, kernel_size=(1,1,1), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(conv)
    bn = BatchNormalization(scale=False, axis=-1)(conv)
    
    shortcut = Conv3D(filters, kernel_size=(1,1,1), strides=(2,2,2), padding='same', use_bias=use_bias,
                  kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(x)
    shortcut = BatchNormalization(scale=False, axis=-1)(shortcut)
    residual = add([shortcut, bn])
    relu = activation()(residual)
    return relu

def get_model(dhw=[48,48,48],weights_decay = 0.,kernel_initializer='he_uniform',weights=None, deeper=False, activation=lambda: Activation('relu')):
    shape = dhw+[1]
    
    inputs = Input(shape = shape)
    conv1 = Conv3D(32, kernel_size=(3, 3, 3), padding='same', use_bias=False, kernel_initializer=kernel_initializer,kernel_regularizer=l2_penality(weights_decay))(inputs)
    
    id1 = identity(conv1, filters=32, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                   use_bias=False, bottleneck=2,activation=activation)
    id1 = identity(id1, filters=32, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                   use_bias=False, bottleneck=2,activation=activation)
    id1 = identity(id1, filters=32, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                   use_bias=False, bottleneck=2,activation=activation)
    down1 = td(id1, filters=64, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
               use_bias=False, bottleneck=2,activation=activation)
    
    id2 = identity(down1, filters=64, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                   use_bias=False, bottleneck=2,activation=activation)
    id2 = identity(id2, filters=64, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                   use_bias=False, bottleneck=2,activation=activation)
    id2 = identity(id2, filters=64, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                   use_bias=False, bottleneck=2,activation=activation)
    
    down2 = td(id2, filters=128, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
               use_bias=False, bottleneck=2,activation=activation)
    id3 = identity(down2, filters=128, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                   use_bias=False, bottleneck=2,activation=activation)
    id3 = identity(id3, filters=128, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                   use_bias=False, bottleneck=2,activation=activation)
    id3 = identity(id3, filters=128, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                   use_bias=False, bottleneck=2,activation=activation)
    down3 = td(id3, filters=256, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
               use_bias=True, bottleneck=2,activation=activation)
    if deeper:
        id4 = identity(down3, filters=256, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                       use_bias=False, bottleneck=2, activation=activation)
        id4 = identity(id4, filters=256, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                       use_bias=False, bottleneck=2, activation=activation)
        id4 = identity(id4, filters=256, weights_decay=weights_decay, kernel_initializer=kernel_initializer,
                       use_bias=False, bottleneck=2, activation=activation)        
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
