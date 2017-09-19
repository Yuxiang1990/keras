import tensorflow as tf
from keras.layers import (Conv3D, BatchNormalization, AveragePooling3D, concatenate,
                          Activation, Input, GlobalAvgPool3D, Dense)
from keras.regularizers import l2 as l2_penalty
from keras.models import Model
from keras.utils.vis_utils import plot_model
from utils.losses import fmeasure, precision, recall


def conv_block(x, activation, filters, bottleneck, kernel_initializer, weights_decay, bn_scale):
    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters, kernel_size=(1, 1, 1), padding='same', use_bias=False,
               kernel_initializer=kernel_initializer, kernel_regularizer=l2_penalty(weights_decay))(x)
    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters // bottleneck, kernel_size=(3, 3, 3), padding='same', use_bias=True,
               kernel_initializer=kernel_initializer, kernel_regularizer=l2_penalty(weights_decay))(x)
    return x


def dense_block(x, k, n, bottleneck,
                activation, kernel_initializer, weights_decay, bn_scale):
    for _ in range(n):
        conv = conv_block(x, activation, k, bottleneck,
                          kernel_initializer, weights_decay, bn_scale)
        x = concatenate([conv, x], axis=-1)
    return x


def transmit_block(x, compression, activation, bn_scale, kernel_initializer, weights_decay):
    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    if (compression is not None) and (compression > 1):
        *_, f = x.get_shape().as_list()
        x = Conv3D(f // compression, kernel_size=(1, 1, 1), padding='same', use_bias=True,
                   kernel_initializer=kernel_initializer, kernel_regularizer=l2_penalty(weights_decay))(x)
        x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    else:
        x = GlobalAvgPool3D()(x)
    return x


def get_model(dhw=[48, 48, 48], k=64, n=3, bottleneck=4, compression=2, first_layer=32,
              activation=lambda: Activation('relu'), bn_scale=True,
              weights_decay=0., kernel_initializer='he_uniform', weights=None):
    shape = dhw + [1]

    inputs = Input(shape=shape)
    conv = Conv3D(first_layer, kernel_size=(3, 3, 3), padding='same', use_bias=True,
                  kernel_initializer=kernel_initializer, kernel_regularizer=l2_penalty(weights_decay))(inputs)

    transmit_down_count = 4
    for l in range(transmit_down_count):
        db = dense_block(conv, k, n, bottleneck,
                         activation, kernel_initializer, weights_decay, bn_scale)
        if l == transmit_down_count - 1:
            conv = transmit_block(db, None, activation,
                                  bn_scale, kernel_initializer, weights_decay)
        else:
            conv = transmit_block(db, compression, activation,
                                  bn_scale, kernel_initializer, weights_decay)

    outputs = Dense(1, kernel_regularizer=l2_penalty(weights_decay),
                    kernel_initializer=kernel_initializer, activation='sigmoid')(conv)

    model = Model(inputs, outputs)
    model.summary()

    if weights is not None:
        model.load_weights(weights, by_name=True)
    return model


def get_compiled(dhw=[48, 48, 48], k=64, n=3, bottleneck=4, compression=2, first_layer=32,
                 loss='binary_crossentropy', optimizer='adam', weights_decay=0.,
                 kernel_initializer='he_uniform', weights=None,
                 activation=lambda: Activation('relu'), bn_scale=True):
    model = get_model(dhw, k, n, bottleneck, compression, first_layer,
                      activation, bn_scale, weights_decay, kernel_initializer, weights)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[loss, 'accuracy', fmeasure, precision, recall])
    return model


if __name__ == '__main__':
    # model = get_model()
    # model.summary()
    model = get_compiled()
    plot_model(model,to_file='desnetbc_v1.png',show_shapes=True,show_layer_names=True)
