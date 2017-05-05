#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage:
    CameraMotion.py  <videoname> [--iladjust] [--drawmatch] [--showdis] [--nodisplay]
    CameraMotion.py  (-h | --help)
    CameraMotion.py  --version

Options:
    -h --help     Show this screen.
    --version     Show version
    --iladjust    Add illumination adjust simulation.
    --drawmatch   drawing match points.
    --showids     print out distance between current frame and base frame.
    --nodisplay   do not display, only print detection results

Description:
    Press Esc to exit.
    Press r to reset base frame.
"""
from docopt import docopt
import time
import cv2
import numpy as np
from Src.illumination import illuminationAdjust
from Src.downsample import downsample


def labelIm(im):
    h, w, c = im.shape
    cv2.rectangle(im, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)),
                  (0, 0, 255), 2)

    
def flowmove(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    inds = np.nonzero(np.abs(flow) < 0.1)
    total = img1.shape[0] * img1.shape[1] * 2
    if len(inds[0])/float(total) > 0.1:
        return True
    else:
        return False


def matchim(img1, img2, ifshowdis):
    flag = flowmove(img1, img2)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    dises = [m.distance for m in matches]
    avedis = np.mean(dises[:10])
    if ifshowdis:
        print("ave dis {}".format(avedis))
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches,  None, flags=2)
    # return img3
    if flag:
        return False, img3
    else:
        return avedis > 10, img3


def main():
        args = docopt(__doc__, version='1.0')
        videoname = args['<videoname>']
        ifshowdis = args['--showdis']
        ifdisplay = not args['--nodisplay']
        print('Processing video {}'.format(videoname))
        cap = cv2.VideoCapture(videoname)
        frame_pre = None
        il = illuminationAdjust()
        framei = 1
        st = time.time()
        while True:
            flag, frame = cap.read()
            if flag:
                # The frame is ready and already captured
                frame = downsample(frame)
                if args['--iladjust']:
                    frame = il.adjust(frame)
                    # frame = il.randomadjust(frame)
                if frame_pre is None:
                    frame_pre = np.copy(frame)
                mflag, mim = matchim(frame_pre, frame, ifshowdis)
                if ifdisplay:
                    if mflag:
                        labelIm(frame)
                    cv2.imshow('video', frame)
                    if args['--drawmatch']:
                        cv2.imshow('m video', mim)
                if mflag:
                    sec = framei/25.
                    print("{} min {} seconds detected".format(int(sec/60.), sec%60 ) )
                framei += 1
            else:
                break
            if ifdisplay:
                k = cv2.waitKey(10)
                if k == 27:
                    break
                elif k == ord('r'):
                    frame_pre = np.copy(frame)
        et = time.time()
        print("total {} frames processed in {} seconds".format(framei-1, et-st))


if __name__ == "__main__":
    main()
