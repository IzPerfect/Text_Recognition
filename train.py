from keras import backend as K
from ImageLoader import TextImageLoader
from CRNN import *
import os
import matplotlib.pyplot as plt
import numpy as np

from parameter import *
from utils import *

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

# Because there is more data in the test image file, the test file iamges was used for training.
train_img_file_path = './IIIT5K/test/'
train_label_file_path = './IIIT5K/test_label/'

train_img_generator_params = {
    'img_path' : train_img_file_path,
    'label_path' : train_label_file_path,
    'img_w' : params['img_w'],
    'img_h' : params['img_h'],
    'batch_size' : params['batch_size'],
    'downsample_factor' : params['downsample_factor'],
    'max_text_len' : params['max_text_len'],
    'do_shuffle' : params['do_shuffle']
}

# TextImageLoader for training
train_dataloader = TextImageLoader(**train_img_generator_params)

val_img_file_path = './IIIT5K/train/'
val_label_file_path = './IIIT5K/train_label/'

val_img_generator_params = {
    'img_path' : val_img_file_path,
    'label_path' : val_label_file_path,
    'img_w' : params['img_w'],
    'img_h' : params['img_h'],
    'batch_size' : params['val_batch_size'],
    'downsample_factor' : params['downsample_factor'],
    'max_text_len' : params['max_text_len'],
    'do_shuffle' : params['do_shuffle']
}

# TextImageLoader for testing
val_dataloader = TextImageLoader(**val_img_generator_params)

crnn_params = {
    'img_w' : train_dataloader.img_w,
    'img_h' : train_dataloader.img_h,
    'num_classes' : params['num_classes'],
    'max_text_len' : train_dataloader.max_text_len,
    'drop_rate' : params['drop_rate'],
    'weight_decay' : params['weight_decay'],
    'learning_rate' : params['learning_rate'],
    'training' : True
}

# CRNN class for text recognition
crnn_model = CRNN(**crnn_params)

# train
crnn_model_history = crnn_model.train(train_dataloader = train_dataloader,
                                     val_dataloader = val_dataloader,
                                     epoch = 150)
