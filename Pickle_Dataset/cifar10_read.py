#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar10_read.py
# Author: Yahui Liu <yahui.cvrs@gmail.com>

import pickle

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo)
  return dict

if __name__ == '__main__':
  print (unpickle('./bin/data_batch_1'))
