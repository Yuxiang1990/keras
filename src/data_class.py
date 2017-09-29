from keras.preprocessing.image import ImageDataGenerator
from .setting import INFO,NPZ
import random
import itertools
import os
import numpy as np

class stomachlab_data(object):
    
    def __init__(self, val_id=6):
        train_pos = INFO[(INFO['fold_id'].map(lambda x: x!=val_id)) & 
                                     (INFO['name'].map(lambda y: y.startswith("pos")))]['name'].values
        train_neg = INFO[(INFO['fold_id'].map(lambda x: x!=val_id)) & 
                                     (INFO['name'].map(lambda y: y.startswith("neg")))]['name'].values
        val_pos = INFO[(INFO['fold_id'].map(lambda x: x==val_id)) & 
                                       (INFO['name'].map(lambda y: y.startswith("pos")))]['name'].values
        val_neg = INFO[(INFO['fold_id'].map(lambda x: x==val_id)) & 
                                       (INFO['name'].map(lambda y: y.startswith("neg")))]['name'].values
        self.shape = (2048,2048)
        self.train_pos_p = self._random_cycle(train_pos) #480
        self.train_neg_p = self._random_cycle(train_neg) #120
        self.val_pos_p = self._random_cycle(val_pos) #80
        self.val_neg_p = self._random_cycle(val_neg) #20
        
        self.datagen = ImageDataGenerator(rotation_range=0.2,
                                            width_shift_range=0.2,         
                                            height_shift_range=0.2,
                                            shear_range=0.1,
                                            zoom_range=0.1,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='reflect')
        self.data_agument_generate = self._data_augmentation
        
        
    def _random_cycle(self, lst):
        random.shuffle(lst)
        return itertools.cycle(lst)
    
    def _data_augmentation(self, x, batch_size=32):
        seed = np.random.randint(0,100)
        data_agument_x = self.datagen.flow(x, batch_size=batch_size, shuffle=False,seed=seed)
        #data_agument_y = self.datagen.flow(y, batch_size=batch_size, shuffle=False,seed=seed)
        return next(data_agument_x).astype(np.uint8)#, next(data_agument_y).astype(np.uint8)
    
    def _load_npz(self, path):
        with np.load(os.path.join(NPZ,path + '.npz')) as npz:
            img = npz['raw']
            if 'mask' not in npz.files:
                #mask = np.zeros(self.shape).astype(np.bool)
                mask = 1
            else:
                #mask = npz['mask']
                mask = 0
        return img, mask
            
    def next_train_batch(self, batch_size, n_pos=None):
        if n_pos is None:
            n_pos = batch_size // 2
        x_arr, y_arr = self._next_batch(batch_size, n_pos,
                                        self.train_pos_p,
                                        self.train_neg_p,
                                        aug=True)
        return x_arr, y_arr
        
    def next_val_batch(self, batch_size, n_pos=None):
        if n_pos is None:
            n_pos = batch_size // 2
        x_arr, y_arr = self._next_batch(batch_size, n_pos,
                                        self.val_pos_p,
                                        self.val_neg_p,
                                        aug=True)
        return x_arr, y_arr
    
    def train_generator(self, batch_size, n_pos=None):
        while True:
            yield self.next_train_batch(batch_size, n_pos)

    def val_generator(self, batch_size, n_pos=None):
        while True:
            yield self.next_val_batch(batch_size, n_pos)
            
    def _next_batch(self, batch_size, n_pos, pos_pointer, neg_pointer, aug=False):
        xs = []
        ys = []
        assert n_pos <= batch_size
        n_neg = batch_size - n_pos
        for _ in range(n_pos):
            img, mask = self._load_npz(next(pos_pointer))
            xs.append(img)
            ys.append(mask)
        for _ in range(n_neg):
            img, mask = self._load_npz(next(neg_pointer))
            xs.append(img)
            ys.append(mask)
        xs = np.array(xs)
        ys = np.array(ys)

        if aug:
            xs= self.data_agument_generate(xs, batch_size=batch_size)
        print(xs.shape,ys.shape)
        return xs, ys
    
    @classmethod
    def get_train_val(cls, val_id=6, train_batch_size=32, val_batch_size=None, train_n_pos=None, val_n_pos=None):
        loader = cls(val_id = val_id)
        
        if val_batch_size is None:
            val_batch_size = train_batch_size
        if val_n_pos is None:
            val_n_pos = train_n_pos
        return (loader.train_generator(train_batch_size, train_n_pos),
                loader.val_generator(val_batch_size, val_n_pos))


if __name__ == "__main__":
    train_generate, val_generate = stomachlab_data.get_train_val(train_batch_size=32, val_id=6)
    train = next(train_generate)
    val = next(val_generate)
    pass