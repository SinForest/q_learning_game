import pygame as pg
import numpy as np
import random
from chrono import Timer
from mechanics import Game

import matplotlib.pyplot as plt

MAP_STEPS = 4

xx, yy = np.meshgrid(np.linspace(0,2*np.pi,50), np.linspace(0,2*np.pi,50))

candidates = []

for map_i in range(5):
    a_sum = np.zeros_like(xx)
    for i in range(1000):
        sign = bool(random.getrandbits(1))
        p    = np.random.uniform(0, 2*np.pi)
        f    = np.random.uniform(0, 3) ** 2
        a    = np.random.uniform(1, 2)# / np.sqrt(f)
        r    = np.random.uniform(0, 1) ** 2
        q    = int(random.getrandbits(1))
        ty   = -yy if sign else yy
        if q:
            arr = np.sin((r*xx+ty)*f+p) * a
        else:
            arr = np.sin((xx+r*ty)*f+p) * a
        
        a_sum += arr
        mx = a_sum.max()
        mn = a_sum.min()
        ll = (mx - mn) / 4
    print(mx, mn, np.median(a_sum))
    for cut_i in range(1,3):
        plt.subplot(5,2,map_i*2+cut_i)
        # a_sum[a_sum < (mn + cut_i * ll)] = 0
        im = (a_sum >= (mn + cut_i * ll)).astype(float)
        im2 = (a_sum <= (mn + (cut_i + 1) * ll)).astype(float)
        im_arr = im * im2
        candidates.append(im_arr) 
        plt.imshow(im_arr, cmap='gray', interpolation='nearest')
        
    
plt.show()