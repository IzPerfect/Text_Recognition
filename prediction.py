from keras import backend as K
from ImageLoader import TextImageLoader
from CRNN import *
import json
import argparse

import os
import matplotlib.pyplot as plt
import numpy as np

from parameter import *
from utils import *

# get arguments
def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', dest = 'model_path',default = './save_model/crnn/crnn_weights.hdf5', help ='load model path')
    parser.add_argument('--image_path', dest = 'image_path', help ='image path', default = './IIIT5K/train/')
    parser.add_argument('--label_path', dest = 'label_path', help ='label path', default = './IIIT5K/train_label/')

    return parser.parse_args()

# main function
def main(args):
    # data load
    crnn_params = {
    'img_w' : params['img_w'],
    'img_h' : params['img_h'],
    'num_classes' : params['num_classes'],
    'max_text_len' : params['max_text_len'],
    'drop_rate' : params['drop_rate'],
    'weight_decay' : params['weight_decay'],
    'learning_rate' : 1e-3,
    'training' : True
    }

    crnn_model = CRNN(**crnn_params)

    try:
        crnn_model.saved_model_use(args.model_path)
    except:
        raise Exception("No weight!")

    test_dir = args.image_path
    test_imgs = os.listdir(args.image_path)
    test_labels = os.listdir(args.label_path)

    for k, test_img in enumerate(test_imgs):
        net_out_value = crnn_model.predict(test_dir + test_img)
        pred_texts= decode_label(net_out_value)

        jstring = open(args.label_path + os.path.splitext(test_img)[0] + '.json', "r").read()
        jstring = json.loads(jstring)

        chars = jstring['image_label']

        print('===')
        print(k, '/', len(test_imgs))
        print('Predicted: %s  /  True: %s' % (pred_texts.lower(), chars.lower()))



if __name__ == '__main__':
    args = arg_parser()
    print("Args : ", args)
    main(args)
