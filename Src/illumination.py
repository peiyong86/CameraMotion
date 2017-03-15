#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np


class illuminationAdjust(object):

    """Class for illumination adjust. First tranfer input image to HSV color space,
    then adjust illumination by adjust value of V."""

    def __init__(self, C=1, low=-70, high=30, step=5):
        """TODO: to be defined.

        :C: TODO
        :low: TODO
        :high: TODO
        :step: TODO

        """
        self._C = C
        self._low = low
        self._high = high
        self._step = step
        self._pos = 0
        self._direct = -1

    def randomadjust(self, im, C=10):
        r = random.random()
        r = (r - 0.5) * 2 * C
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im = im.astype('float64')
        im[:, :, 2] += r
        im = np.maximum(im, 0)
        im = np.minimum(im, 255)
        im = im.astype('uint8')
        im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        return im

    def adjust(self, im):
        r = self.getr()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im = im.astype('float64')
        im[:, :, 2] += r
        im = np.maximum(im, 0)
        im = np.minimum(im, 255)
        im = im.astype('uint8')
        im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        return im

    def getr(self):
        r = self._pos * self._C
        self.movepos()
        return r

    def movepos(self):
        nextpos = self._pos + self._step * self._direct
        if nextpos < self._low or nextpos > self._high:
            self._direct = self._direct * -1
            self.movepos()
        else:
            self._pos = nextpos
