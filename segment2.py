#!/usr/bin/env python

'''
Watershed segmentation
=========

This program demonstrates the watershed segmentation algorithm
in OpenCV: watershed().

Usage
-----
watershed.py [image filename]

Keys
----
  1-7   - switch marker color
  SPACE - update segmentation
  r     - reset
  a     - toggle autoupdate
  ESC   - exit

'''


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import sys
from common import Sketcher


inputname=str(sys.argv[1])
outputname_cn=inputname.replace(".png", "seg.png")

outputname_cn=outputname_cn.replace("INPUT", "OUTPUT_BN")
#plt.savefig(outputname_cn)
#print(outputname_bn)

#cv2.imwrite(outputname_bn,img_bin)


class App:
    def __init__(self, fn):
        self.img = cv2.imread(fn)
        if self.img is None:
            raise Exception('Failed to load image file: %s' % fn)

        h, w = self.img.shape[:2]
        self.markers = np.zeros((h, w), np.int32)
        self.markers_vis = self.img.copy()
        self.cur_marker = 1
        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255

        self.auto_update = True
        self.sketch = Sketcher('img', [self.markers_vis, self.markers], self.get_colors)

    def get_colors(self):
        return list(map(int, self.colors[self.cur_marker])), self.cur_marker

    def watershed(self):
        m = self.markers.copy()
        cv2.watershed(self.img, m)
        overlay = self.colors[np.maximum(m, 0)]
        self.vis = cv2.addWeighted(self.img, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)
        cv2.imshow('watershed', self.vis)

    def run(self):
        while cv2.getWindowProperty('img', 0) != -1 or cv2.getWindowProperty('watershed', 0) != -1:
            ch = cv2.waitKey(50)
            if ch == 27:
                cv2.imwrite(outputname_cn,self.vis)
                break
            if ch >= ord('1') and ch <= ord('7'):
                self.cur_marker = ch - ord('0')
                print('marker: ', self.cur_marker)
            if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
                self.watershed()
                self.sketch.dirty = False
            if ch in [ord('a'), ord('A')]:
                self.auto_update = not self.auto_update
                print('auto_update if', ['off', 'on'][self.auto_update])
            if ch in [ord('r'), ord('R')]:
                self.markers[:] = 0
                self.markers_vis[:] = self.img
                self.sketch.show()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = '../data/fruits.jpg'
    print(__doc__)
    App(fn).run()

