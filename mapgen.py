import pygame as pg
import numpy as np
import random
from chrono import Timer
from queue import Queue
import tqdm
from tqdm import trange
import itertools
import math

import matplotlib.pyplot as plt

def l2(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

def generate_world(size=50, n_traps=None, n_nests=None):
    xx, yy = np.meshgrid(np.linspace(0,2*np.pi,size), np.linspace(0,2*np.pi,size))

    if n_traps is None:
        n_traps = max(math.ceil((size**2) / (50**2) * 10 + 1), 4)
    
    if n_nests is None:
        n_nests = max(math.ceil((size**2) / (50**2) * 4 + 1), 3)

    print(n_traps, n_nests)

    candidates = []

    for map_i in range(5):
        a_sum = np.zeros_like(xx)
        for i in range(1000):
            sign = bool(random.getrandbits(1))
            p    = np.random.uniform(0, 2*np.pi)
            f    = np.random.uniform(0, 3) ** 2
            a    = np.random.uniform(1, 2)
            r    = np.random.uniform(0, 1) ** 2
            q    = bool(random.getrandbits(1))
            ty   = -yy if sign else yy
            if q:
                arr = np.sin((r*xx+ty)*f+p) * a
            else:
                arr = np.sin((xx+r*ty)*f+p) * a
            
            a_sum += arr
            mx = a_sum.max()
            mn = a_sum.min()
            ll = (mx - mn) / 4
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
                    kyu.put(np.array(xy))
            clusters.append(cluster)
            mask = ((1 - np.sum(clusters, axis=0)) * a_map).astype(bool)
            
        clusters.sort(key=lambda x:x.sum(), reverse=True)
        game_map = (1 - clusters[0]).astype(bool)
        new_candidates.append(game_map)

    new_candidates.sort(key=lambda x:x.sum())
    #TODO: check, if suff. space
    blocked = new_candidates[0]
    
    # add traps:
    traps = np.zeros_like(blocked).astype(bool)
    for i in range(n_traps):
        """
        while True:
            x = np.random.randint(0, size, 2)
            lines = []
            if blocked[tuple(x)] == False:
                continue
            near = [l2(x, p) for p in np.stack(np.where(traps)).T]
            if near and (min(near) - 3) < np.random.uniform(30):
                continue
                
            for y in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                xy = x + y
                if (xy < 0).any(): continue
                if (xy >= size).any(): continue
                line = []
                while blocked[tuple(xy)] == False:
                    line.append(tuple(xy))
                    xy += y
                    if (xy < 0).any() or (xy >= size).any():
                        line = []
                        break
                
                lines.append(line)
            lines = [x for x in lines if 2 <= len(x) <= 20]
            if len(lines) > 0:
                random.shuffle(lines)
                for t in lines[0]:
                    traps[t] = True
                break
        """
        
        while True:
            x = np.random.randint(0, size, 2)
            lines = []
            if blocked[tuple(x)] == True:
                continue
            near = [l2(x, p) for p in np.stack(np.where(traps)).T]
            if near and (min(near) - 3) < np.random.uniform(30):
                continue
            xy = [x + y for y in [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1], [1, -1], [-1, 1]]]
            xy = [tuple(t) for t in xy if not ((t >= size).any() or (t < 0).any())]
            xy = [t for t in xy if blocked[t] == False]
            if len(xy) >= 4:
                chance = (len(xy) - 6) / 8
                for t in xy:
                    rnd = np.random.uniform(0, 1)
                    if rnd > chance:
                        traps[t] = True

                break

    free = np.stack(np.where(((1 - blocked) - traps).astype(bool))).T
    mx, ms = None, 0
    for i in range(10000):
        x = np.random.randint(0, len(free), n_nests - 1)
        x = [free[e] for e in x]
        s = sum([np.sqrt(l2(*t)) for t in itertools.product(x,x)])
        if s > ms:
            mx, ms = x, s
    nests = [tuple(t) for t in mx]

    ind = np.argmin(((free - np.mean(mx, axis=0)) ** 2).sum(axis=1))
    nests.append(tuple(free[ind]))

    mx, ms = None, 0
    for i in range(10000):
        x = np.random.randint(0, len(free))
        x = free[x]
        s = sum([l2(x, ne) for ne in nests])
        if s > ms:
            mx, ms = x, s
    player = tuple(mx)

    return blocked, traps, nests, player
