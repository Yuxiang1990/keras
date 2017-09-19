from keras.layers import Conv2D, BatchNormalization, SpatialDropout2D, Activation, Input, \
                                                                Lambda, MaxPool2D, UpSampling2D, Cropping2D, concatenate
from keras.regularizers import l2
from keras.models import Model

from .losses import dice_loss, precision, recall, fmeasure

CHANNEL_MEAN = [228.42726121, 197.81321541, 231.0754979]
CHANNEL_STD = [11.6068092, 16.37567894, 7.56506719]

def conv_block(x, filters, activation, use_bias, dropout_rate, weight_decay, kernel_initializer):
    conv = BatchNormalization(scale=False, axis=-1)(x)
    conv = activation()(conv)
    conv = Conv2D(filters, kernel_size=(3, 3), padding='valid', use_bias=use_bias,
                  kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(conv)
    if dropout_rate is not None:
        conv = SpatialDropout2D(rate=dropout_rate)(conv)
    return conv


def td(x, filters, activation, use_bias, dropout_rate, weight_decay, kernel_initializer):
    conv = BatchNormalization(scale=False, axis=-1)(x)
    conv = activation()(conv)
    conv = Conv2D(filters, kernel_size=(1, 1), padding='valid', use_bias=use_bias,
                  kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(conv)
    if dropout_rate is not None:
        conv = SpatialDropout2D(rate=dropout_rate)(conv)
    pool = MaxPool2D(pool_size=(2, 2))(conv)
    return pool


def tu(x, filters, use_bias, weight_decay, kernel_initializer):
    up = UpSampling2D(size=(2, 2))(x)
    conv = Conv2D(filters, kernel_size=(1, 1), padding='valid', use_bias=use_bias,
                  kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(up)
    return conv


def crop_and_concat(src, targ):
    _, src_h, src_w, _ = src.get_shape().as_list()
    _, targ_h, targ_w, _ = targ.get_shape().as_list()
    crop_h = (src_h - targ_h) // 2
    crop_w = (src_w - targ_w) // 2
    cropped = Cropping2D([(crop_h, crop_h), (crop_w, crop_w)])(src)
    concat = concatenate([cropped, targ], axis=-1)
    return concat


def get_model(hw=[572, 572], c=3, activation=lambda: Activation('relu'), dropout_rate=0.2, 
              weight_decay=0., kernel_initializer='he_uniform', weights=None, load_by_name=False,
              verbose=True, scale=True):
    inputs = Input(hw + [c])

    if scale:
        scaled = Lambda(lambda x: (x - CHANNEL_MEAN) / CHANNEL_STD)(inputs)
    else:
        scaled = inputs

    filter1 = 48
    conv1 = Conv2D(filter1, kernel_size=(3, 3), padding='valid', use_bias=False,
                   kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(scaled)
    conv1 = conv_block(conv1, filter1, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    td1 = td(conv1, filter1, activation=activation, use_bias=False, dropout_rate=dropout_rate,
             weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    filter2 = 48*2
    conv2 = conv_block(td1, filter2, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    conv2 = conv_block(conv2, filter2, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    td2 = td(conv2, filter2, activation=activation, use_bias=False, dropout_rate=dropout_rate,
             weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    filter3 = 48*4
    conv3 = conv_block(td2, filter3, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    conv3 = conv_block(conv3, filter3, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    td3 = td(conv3, filter3, activation=activation, use_bias=False, dropout_rate=dropout_rate,
             weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    filter4 = 48*8
    conv4 = conv_block(td3, filter4, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    conv4 = conv_block(conv4, filter4, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    td4 = td(conv4, filter4, activation=activation, use_bias=False, dropout_rate=dropout_rate,
             weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    filter5 = 48*16
    conv5 = conv_block(td4, filter5, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    conv5 = conv_block(conv5, filter5, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    filter6 = 48*8
    tu4 = tu(conv5, filter6, use_bias=False, weight_decay=weight_decay,
             kernel_initializer=kernel_initializer)
    concat4 = crop_and_concat(conv4, tu4)
    conv6 = conv_block(concat4, filter6, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    conv6 = conv_block(conv6, filter6, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    filter7 = 48*4
    tu3 = tu(conv6, filter7, use_bias=False, weight_decay=weight_decay,
             kernel_initializer=kernel_initializer)
    concat3 = crop_and_concat(conv3, tu3)
    conv7 = conv_block(concat3, filter7, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    conv7 = conv_block(conv7, filter7, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    filter8 = 48*2
    tu2 = tu(conv7, filter8, use_bias=False, weight_decay=weight_decay,
             kernel_initializer=kernel_initializer)
    concat2 = crop_and_concat(conv2, tu2)
    conv8 = conv_block(concat2, filter8, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    conv8 = conv_block(conv8, filter8, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    filter9 = 48
    tu1 = tu(conv8, filter9, use_bias=False, weight_decay=weight_decay,
             kernel_initializer=kernel_initializer)
    concat2 = crop_and_concat(conv1, tu1)
    conv9 = conv_block(concat2, filter9, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    conv9 = conv_block(conv9, filter9, activation=activation, use_bias=False, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    latest = BatchNormalization(scale=False, axis=-1)(conv9)

    outputs = Conv2D(1, kernel_size=(1, 1), padding='valid', use_bias=True, activation='sigmoid',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(latest)

    model = Model(inputs, outputs)

    if verbose:
        model.summary()

    if weights is not None:
        model.load_weights(weights, by_name=load_by_name)
        if verbose:
            print('weights@{0} loaded.'.format(weights))

    return model


def get_compiled(hw=[572, 572], c=3, activation=lambda: Activation('relu'), dropout_rate=0.2,
                 weight_decay=0., kernel_initializer='he_uniform', weights=None, verbose=True,
                 loss='dice', optimizer='adam'):
    if loss == 'dice':
        loss = dice_loss

    model = get_model(hw=hw, c=c, activation=activation, dropout_rate=dropout_rate,
                      weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                      weights=weights, verbose=verbose)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[loss, fmeasure, precision, recall])
    return model


if __name__ == '__main__':
    # get_model()
    get_compiled(hw=[476, 476])
