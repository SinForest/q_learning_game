import pygame as pg
import numpy as np
import random
from chrono import Timer
from mechanics import Game
from queue import Queue

import matplotlib.pyplot as plt

MAP_STEPS = 4
def generate_world(size=50):
    xx, yy = np.meshgrid(np.linspace(0,2*np.pi,size), np.linspace(0,2*np.pi,size))

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
            im = (a_sum >= (mn + cut_i * ll)).astype(int)
            im2 = (a_sum <= (mn + (cut_i + 1) * ll)).astype(int)
            im_arr = im * im2
            candidates.append(im_arr)
            candidates.append(1 - im_arr) 

    new_candidates = []
    for a_map in candidates:
        clusters = []
        mask = ((1 - np.sum(clusters, axis=0)) * a_map).astype(bool)
        while(mask.any()):
            free    = np.stack(np.where(mask)).T  # 0: wall, 1:air
            start   = np.random.randint(len(free))
            coord   = free[start]
            cluster = np.zeros_like(a_map).astype(bool)
            kyu   = Queue()
            kyu.put(coord)
            while not kyu.empty():
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
            clusters.append(cluster)
            mask = ((1 - np.sum(clusters, axis=0)) * a_map).astype(bool)
            
        clusters.sort(key=lambda x:x.sum(), reverse=True)
        game_map = (1 - clusters[0]).astype(bool)
        new_candidates.append(game_map)

    new_candidates.sort(key=lambda x:x.sum())
    return new_candidates[0]
