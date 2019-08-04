import cv2
import itertools, os, time
import numpy as np
from parameter import *

def change_char(x):
    result = []
    for c in x:
        if c not in char_elements and c.lower() in char_elements:
            result.append(c.lower())
        else:
            result.append(c)
    return "".join(result)

def labels_to_text(labels):
    result = []
    for c in labels:
        if c == len(params['letters']):
            result.append("")
        else:
            result.append(params['letters'][c])
    return "".join(result)

def text_to_labels(text):
    result = []
    if len(text) > params['max_text_len']:
        text = text[:params['max_text_len']]
    for c in text:
        if c in params['letters']:
            result.append(params['letters'].index(c))
        else:
            result.append(params['letters'].index('_'))
    return result

def isnumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], axis=1))
    out_best = [k for k, g in itertools.groupby(out_best)]

    result = ''
    for i in out_best:
        if i < len(params['letters']):
            result += params['letters'][i]
    return result
