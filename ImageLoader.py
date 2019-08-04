import cv2
import os
import random
import numpy as np
import json

from parameter import *
from utils import *

class TextImageLoader:
    def __init__(self, img_path, label_path, img_w, img_h,
                 batch_size, downsample_factor, max_text_len=params['max_text_len'], do_shuffle = False):
        self.img_dirpath = img_path
        self.label_dirpath = label_path
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.do_shuffle = do_shuffle
        self.total_img_list = os.listdir(self.img_dirpath)
        self.total_label_list = os.listdir(self.label_dirpath)
        self.total_batch = len(os.listdir(self.img_dirpath))

        print('Find ', len(self.total_img_list),' images')
        print('Find ', len(self.total_label_list),' labels')

    def img_transform(self, img):
        img = img.T
        img = np.expand_dims(img, -1)
        return img


    def get_image(self, img_file_path, w , h):
        img = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32)
        img = (img / 255.0)

        return img

    def get_label(self, label_file_path, img_filename):
        jstring = open(label_file_path + os.path.splitext(img_filename)[0] + '.json', "r").read()
        jstring = json.loads(jstring)

        chars = jstring['image_label']

        return chars

    def generator(self):
        while True:
            if self.do_shuffle == True:
                idx_arr = np.random.permutation(self.total_batch)
            else:
                idx_arr = np.arange(self.total_batch)

            for batch in range(0, len(idx_arr), self.batch_size):

                l_bound = batch
                r_bound = batch + self.batch_size

                if r_bound > len(idx_arr):
                    r_bound = len(idx_arr)
                    l_bound = r_bound - self.batch_size

                current_batch = idx_arr[l_bound:r_bound]

                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
                Y_data = np.ones([self.batch_size, self.max_text_len])*-1

                input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
                label_length = np.zeros((self.batch_size, 1))

                for i, v in enumerate(current_batch):

                    img = self.get_image(self.img_dirpath  + self.total_img_list[v], self.img_w, self.img_h)
                    text = self.get_label(self.label_dirpath, self.total_img_list[v])

                    X_data[i] = self.img_transform(img)
                    Y_data[i,0:len(text)] = text_to_labels(text)
                    label_length[i] = len(text)

                inputs = {
                    'the_input': X_data,
                    'the_labels': Y_data,
                    'input_length': input_length,
                    'label_length': label_length
                }

                outputs = {'ctc': np.zeros([self.batch_size])}

                yield (inputs, outputs)
