#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np


def downsample(im):
    im = cv2.resize(im, (320, 240))
    im = np.rot90(im, 3)
    im = im.copy()
    return im
