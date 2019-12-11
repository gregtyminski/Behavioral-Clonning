import pandas as pd
import numpy as np
import math
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tqdm
from PIL import Image
from sklearn.utils import shuffle

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, folder_name: str = './udacity_data/', epochs: int = 100, batch_size: int = 32, balance: bool = False, debug: bool = False, empty: bool = False):
        ## some constants
        # side cam correction
        self.side_cam_correction = 0.2
        self.folder_name = folder_name.strip()
        self.images_folder_name = self.folder_name
        self.data_location = 'driving_log.csv'
        # columns names
        self.center = 'center'
        self.left = 'left'
        self.right = 'right'
        self.steering = 'steering'
        self.throttle = 'throttle'
        self.brake = 'brake'
        self.debug = debug
        self.balance = balance
        self.out_image_folder = './image_folder/'

        if empty:
            pass

        # load dataset
        self.df = self.__load_db__(self.folder_name)

        # number of epochs
        self.epochs = epochs
        # size of batch
        self.batch_size = batch_size

        # balance dataset
        if balance:
            self.balance_dataset()

        self.__recalc_sizes__()

        self.__debug_data__()
        pass

    def copy_empty(self):
        return DataGenerator(self.folder_name, self.epochs, self.batch_size, self.balance, self.debug, empty = True)

    def __len__(self):
        return self.batches

    def __debug_data__(self):
        if self.debug:
            print(self.df.head())
            print(self.df.tail())
            print('type(steering)=', type(self.df[self.steering][0]))
            print('type(throttle)=', type(self.df[self.throttle][0]))
            print('type(brake)=', type(self.df[self.brake][0]))
            print('type(center)=', type(self.df[self.center][0]))
            print('lenght=', self.lenght)
            print('epochs=', self.epochs)
            print('batch_size=', self.batch_size)
            print('batches=', self.batches)
            print('batch_shape=', self.batch_shape)
            print('1st row=', self.df[self.center][0])
            print('896nd row=', self.df[self.center][896])
        pass

    def __recalc_sizes__(self):
        # size of dataset
        self.lenght = len(self.df)
        # Number of batches
        self.batches = math.ceil(self.lenght / self.batch_size)
        # shape of the batch
        self.batch_shape = (self.batch_size, 160, 320, 3)
        pass

    def __limit__(self, val):
        '''
        Method limits value to range <-1, 1>

        :param val: Initial value.
        :return: Value cropped to range <-1, 1>
        '''
        if val < 0:
            if val < -1:
                return -1
            else:
                return val
            pass
        if val > 0:
            if val > 1:
                return 1
            else:
                return val
            pass
        return val

    def __getitem__(self, index: int):
        # index = index of bach
        range_start = index * self.batch_size
        range_end = range_start + self.batch_size
        if range_end > self.lenght:
            range_end = self.lenght - 1
        
        if self.debug:
            print('range_start=', range_start)
            print('range_end=', range_end)
        
        images = np.zeros(self.batch_shape)
        results = np.zeros((images.shape[0]))
        batch_index = 0
        
        for x in range(range_start, range_end):
            if self.debug:
                print('x=', x)
            
            # image
            c_img = self.df[self.center][x]
            images[batch_index][:,:,:] = cv2.imread(c_img)
            steering = self.df[self.steering][x]
            results[batch_index] = steering
            batch_index += 1
            
        x = np.reshape(images, self.batch_shape)
        return x, results

    def __load_db__(self, path):
        # load dataset
        df = pd.read_csv(path + self.data_location, header=0)

        df[self.center] = df[self.center].apply(lambda x: x.split('/')[-1])
        df[self.center] = df[self.center].apply(lambda x: path + 'IMG/' + x.strip())

        df[self.left] = df[self.left].apply(lambda x: x.split('/')[-1])
        df[self.left] = df[self.left].apply(lambda x: path + 'IMG/' + x.strip())

        df[self.right] = df[self.right].apply(lambda x: x.split('/')[-1])
        df[self.right] = df[self.right].apply(lambda x: path + 'IMG/' + x.strip())

        # create 'center' copy and drop 'left' and 'right' images
        df_center = df.copy()
        df_center.drop(self.left, axis=1, inplace=True)
        df_center.drop(self.right, axis=1, inplace=True)

        # create 'left' copy and drop 'center' and 'right' images and add some steering cooeficient
        df_left = df.copy()
        df_left[self.steering] = df_left[self.steering].apply(
            lambda x: self.__limit__(float(x) + self.side_cam_correction))
        df_left.drop(self.center, axis=1, inplace=True)
        df_left.drop(self.right, axis=1, inplace=True)
        df_left.rename(columns={self.left: self.center}, inplace=True)

        # create 'right' copy and drop 'left' and 'center' images and subtract some steering cooeficient
        df_right = df.copy()
        df_right[self.steering] = df_right[self.steering].apply(
            lambda x: self.__limit__(float(x) - self.side_cam_correction))
        df_right.drop(self.left, axis=1, inplace=True)
        df_right.drop(self.center, axis=1, inplace=True)
        df_right.rename(columns={self.right: self.center}, inplace=True)

        # merge all 3 copies (left, right, center)
        df = pd.concat([df_right, df_center, df_left], axis=0)

        # shuffle dataset
        df = df.sample(frac=1).reset_index(drop=True)
        return df

    def merge(self, data):
        assert data is not None
        assert type(data) is DataGenerator
        merge = self.copy_empty()
        df = pd.concat([self.df, data.df])
        merge.df = df
        merge.df.drop_duplicates()
        merge.epochs = self.epochs
        merge.batch_size = self.batch_size
        if self.balance:
            merge.balance_dataset()
        merge.__recalc_sizes__()
        merge.__debug_data__()
        return merge

    def split(self, factor: float = 0.2, bins: int = 10):
        train = []
        copy_df = self.df.copy()

        for i in range(bins): # 10 bins
            min = -1. + i * 0.2
            max = -1. + (i+1) * 0.2
            rng = (min, max)
            train.append(self.df[((self.df[self.steering] >= rng[0]) & (self.df[self.steering] <= rng[1]))].sample(frac=factor))

        valid_df = pd.concat(train).reset_index(drop=True)
        train_df = copy_df.drop(valid_df.index).reset_index(drop=True)

        train_g = self.copy_empty()
        train_g.df = train_df
        train_g.epochs = self.epochs
        train_g.batch_size = self.batch_size
        train_g.__recalc_sizes__()
        train_g.__debug_data__()

        valid_g = self.copy_empty()
        valid_g.df = valid_df
        valid_g.epochs = self.epochs
        valid_g.batch_size = self.batch_size
        valid_g.__recalc_sizes__()
        valid_g.__debug_data__()

        return train_g, valid_g

    def show_stats(self, logarithmic: bool = False, bins: int = 50, range: tuple=None):
        plt.figure(figsize=(12, 3))
        return plt.hist(self.df[self.steering], bins=bins, log=logarithmic, range=range)

    def preview_data(self, entries: int = 10, random: bool = True):
        pass

    def mean_of_all(self):
        mean = np.zeros((160, 320, 3), dtype='long')
        images = list(self.df[self.center].values)
        for image in images:
            f = cv2.imread(image)
            if f is not None:
                f = f[:,:,::-1]
                img = np.array(f)
                mean += img
            else:
                print(image + ' is None')
        mean = (mean // len(images)).astype('uint8')
        return Image.fromarray(mean)

    def drop(self, fraction: float, range: tuple):
        assert range is not None
        assert fraction is not None
        indexes = self.df[((self.df[self.steering] >= range[0]) & (self.df[self.steering] <= range[1]))].sample(frac=fraction).index
        self.df = self.df.drop(indexes).reset_index(drop=True)
        self.__recalc_sizes__()
        return self.df

    def balance_dataset(self):
        self.drop(fraction=0.9, range=(-0.22, -0.18))
        self.drop(fraction=0.9, range=(0.18, 0.22))
        self.drop(fraction=0.9, range=(-0.01, 0.01))
        pass