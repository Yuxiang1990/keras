import numpy as np
from keras.optimizers import Adam, SGD
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard, EarlyStopping
import math
import shutil
import os
from datetime import datetime
from utils.unet_v1 import get_compiled

def prepare_for_unet3D(img):
    img = img.astype(np.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    return img


def image_generator(data_prefix, batch_size, shuffle=True):
    which_sample_select = 0
    batch_index = 0
    unet_input = []
    unet_output = []
    while True:
        print('-' * 30)
        print('Loading and preprocessing train data...', which_sample_select)
	#tips: we can use data path, shuffle path, then load the data.

        which_sample_select += 1
        if(which_sample_select == 30):
            which_sample_select = 0

        print(imgs_train.shape, imgs_pos_class.shape)
        print(imgs_neg_train.shape, imgs_neg_class.shape)

        imgs_train = prepare_for_unet3D(imgs_train)
        imgs_neg_train = prepare_for_unet3D(imgs_neg_train)

        pos_data = imgs_train
        pos_data_mask = imgs_pos_class
        neg_data = imgs_neg_train
        neg_data_mask = imgs_neg_class
        pos_len = len(pos_data)  # input len = pos_len *2
        neg_len = len(neg_data)

        if shuffle:
            rand_pos = np.random.choice(range(pos_len), pos_len, replace=False)
            rand_neg = np.random.choice(range(neg_len), neg_len, replace=False)
            pos_data = pos_data[rand_pos]
            pos_data_mask = pos_data_mask[rand_pos]

            neg_data = neg_data[rand_neg]
            neg_data_mask = neg_data_mask[rand_neg]

        # method 1:
        for i in range(min(pos_len, neg_len) * 2):
            if(i % 2):
                unet_input.append(pos_data[i // 2])
                unet_output.append(pos_data_mask[i // 2])
                batch_index += 1
            else:
                unet_input.append(neg_data[i // 2])
                unet_output.append(neg_data_mask[i // 2])
                batch_index += 1
            if batch_index >= batch_size:
                x = np.array(unet_input)
                y = np.array(unet_output)
                yield x, y
                unet_input = []
                unet_output = []
                batch_index = 0


def train_and_predict(use_existing):

    model = get_compiled(dhw=[cube_size, cube_size, cube_size], loss=dice_coef_loss,
                         optimizer=Adam(lr=3.e-5, decay=3.e-5),
                         weights_decay=1.e-4,  # smaller, e.g. 3.34e-5
                         kernel_initializer='he_uniform', weights=weight,
                         dropout_rate=0.2, skip_connect=True)  # Noting me!!!!!

    train_gen = image_generator('trainImages', batch_size, True)
    holdout_gen = image_generator('valImages', batch_size, True)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_fmeasure', factor=0.334, patience=10, mode='max')  # , min_lr=1.0e-6)
    csv_logger = CSVLogger(workspace_path + '/training.csv')
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min')
    tensorboard = TensorBoard(workspace_path + '/logs')
    model_checkpoint1 = ModelCheckpoint(workspace_path + '/unet' + '{epoch:02d}-{val_recall:.4f}.hdf5',
                                        verbose=1, save_best_only=False, period=5)

    model_checkpoint2 = ModelCheckpoint(workspace_path + '/best_f1.hdf5', monitor='val_fmeasure',
                                        verbose=1, mode='max', save_best_only=True, period=1)

    model_checkpoint3 = ModelCheckpoint(workspace_path + '/best_recall.hdf5', monitor='val_recall',
                                        verbose=1, mode='max', save_best_only=True, period=1)

    model.fit_generator(train_gen, steps_per_epoch=250, epochs=500, validation_data=holdout_gen, validation_steps=250,
                        max_q_size=500, verbose=1,
                        # epochs=20, steps_per_epoch=250,
                        callbacks=[model_checkpoint1, model_checkpoint2, model_checkpoint3,
                                   reduce_lr, early_stop, tensorboard, csv_logger])

    shutil.copy(workspace_path + '/best.hdf5',
                BEST_MODEL_DIR + 'unet_best.hdf5')


if __name__ == '__main__':
    train_and_predict(False)

