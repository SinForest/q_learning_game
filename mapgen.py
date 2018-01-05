import pygame as pg
import numpy as np
import random
from chrono import Timer
from mechanics import Game
from queue import Queue

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
        a    = np.random.uniform(1, 2)
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
        # plt.subplot(5,2,map_i*2+cut_i)
        # a_sum[a_sum < (mn + cut_i * ll)] = 0
        im = (a_sum >= (mn + cut_i * ll)).astype(int)
        im2 = (a_sum <= (mn + (cut_i + 1) * ll)).astype(int)
        im_arr = im * im2
        candidates.append(im_arr) 
        #plt.imshow(im_arr, cmap='gray', interpolation='nearest')

for a_map in candidates:
    print(len(a_map))
    free    = np.stack(np.where(a_map == 1)).T  # 0: wall, 1:air
    start   = np.random.randint(len(free))
    print(start)
    coord   = free[start]
    print(coord)
    cluster = np.zeros_like(a_map).astype(bool)
    kyu   = Queue()
    kyu.put(coord)
    while not kyu.empty():
        #print(kyu.qsize())
        x = kyu.get()
        if cluster[tuple(x)]:
            continue
        cluster[tuple(x)] = True
        for y in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            xy = x + y
            if (xy < 0).any(): continue
            if (xy >= len(a_map)).any(): continue
            xy = tuple(xy)
            if a_map[xy] == 0: continue
            if cluster[xy] == True: continue
            #TODO:  test if already in other cluster
            kyu.put(np.array(xy))
            print(xy)
    plt.imshow(a_map + cluster / 2)
    plt.show()


    
    
plt.show()