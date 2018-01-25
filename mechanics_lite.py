import numpy as np
import random
from queue import PriorityQueue
import string
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from mapgen import generate_world
import itertools
import os
import pickle
from time import sleep
from multiprocessing import Process

GAME_DEBUG = True

def process_pregen(args, interval=0, max_lvls=300):
    while(True):
        lvl_dir = "lite_levels/{}/".format(args[0])
        if len([name for name in os.listdir(lvl_dir) if os.path.isfile(lvl_dir + name)]) < max_lvls:
            pregen_level(*args)
        
        sleep(interval)

def pregen_level(size, n_traps=None, n_nests=None):
    lvl_dir = "lite_levels/{}/".format(size)

    world = generate_world(size, n_traps, n_nests)

    name = hex(hash(world[0].tostring()))[3:11] + ".p"
    pickle.dump(world, open(lvl_dir + name, 'wb'))

def char_to_pixels(text, path='DejaVuSans.ttf', fontsize=14):
    """
    Based on https://stackoverflow.com/a/27753869/190597 (jsheperd)
    """
    font = ImageFont.truetype(path, fontsize) 
    w, h = font.getsize(text)  
    h *= 2
    image = Image.new('L', (w, h), 1)  
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font) 
    arr = np.asarray(image)
    arr = np.where(arr, 0, 1)
    arr = arr[(arr != 0).any(axis=1)]
    return arr.astype(bool)

def l2(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

class Game:

    CHASE_TIME = 10
    STAND_TIME = 6

    C_BG     = [ 50,  50,  50] #rgb(50,50,50)
    C_TRAP   = [  0, 255,   0] #rgb(0,255,0)
    C_BLOCK  = [191, 191, 191] #rgb(191,191,191)
    C_PLAYER = [  0,   0, 255] #rgb(0,0,255)
    C_ENEMY  = [255,   0,   0] #rgb(255,0,0)
    C_COIN   = [255, 255,   0] #rgb(255,255,0)
    C_ECOIN  = [255, 127,   0] #rgb(255,127,0)
    C_PTRAP  = [ 60, 150,  60] #rgb(60,150,60)
    C_NEST   = [110, 120, 180] #rgb(110,20,180)

    DIRS = {'u': ( 0,-1),
              0: ( 0,-1),
            'd': ( 0, 1),
              1: ( 0, 1),
            'l': (-1, 0),
              2: (-1, 0),
            'r': ( 1, 0),
              3: ( 1, 0),}

    def __init__(self, size=50, stretch=8, pregen=2):

        self.size      = size
        self.stretch   = stretch
        self.pregen    = pregen
        n_traps, n_nests = 0, 0

        if self.pregen:
            lvl_dir = "lite_levels/{}/".format(size)
            if not os.path.exists(lvl_dir):
                os.makedirs(lvl_dir)
            self.pregen = [Process(target=process_pregen, args=((size, n_traps, n_nests),)) for __ in range(pregen)]
            for p in self.pregen:
                p.start()

        self.init_game()

    def init_game(self):
        self.you_lost = False
        self.score    = 0
        self.chase    = False
        self.timer    = self.STAND_TIME

        if self.pregen:
            try:
                lvl_dir  = "lite_levels/{}/".format(self.size)
                lvl_name = random.choice([name for name in os.listdir(lvl_dir) if os.path.isfile(lvl_dir + name)])
                self.blocked, self.traps, self.nests, self.player = pickle.load(open(lvl_dir + lvl_name, 'rb'))
                os.remove(lvl_dir + lvl_name)
            except:
                print("Couldn't load level, generating new one")
                self.blocked, self.traps, self.nests, self.player = generate_world(self.size, 0, 0)
        else:
            self.blocked, self.traps, self.nests, self.player = generate_world(self.size, 0, 0)
        
        self.coins = []        
        self.spawn_enemy()

        for _ in range(10):
            self.new_coin()
    
    def kill_pregen(self):
        if self.pregen:
            [x.terminate() for x in self.pregen]

    def get_visual(self, hud=True):
        """
        creates a visual representation of the game as a numpy array
        """
        vis = np.zeros(self.traps.shape + (3,)).astype(int) + self.C_BG

        if not self.you_lost:
            vis[self.blocked] = self.C_BLOCK
            
            if self.chase:
                vis[self.enemy] = self.C_ENEMY
            else:
                vis[self.enemy] = self.C_TRAP
            
            for co in self.coins:
                if co == self.enemy:
                    vis[co] = self.C_ECOIN
                else:
                    vis[co] = self.C_COIN

            vis[self.player] = self.C_PLAYER

        if hud:
            vis = np.pad(vis, ((1, 1), (1, 1), (0, 0)), 'constant')
            vis = np.pad(vis, ((2, 2), (5, 1), (0, 0)), 'constant', constant_values=25)
            vis = np.pad(vis, ((1, 1), (1, 1), (0, 0)), 'constant')
            vis = vis.repeat(self.stretch, 0).repeat(self.stretch, 1)
            
            fs = self.size // 2 + 3 

            text = char_to_pixels("Score: {}".format(self.score), fontsize=fs).T
            x, y = text.shape
            pad = 3*self.stretch
            vis[-x-pad:-pad, pad:pad+y][text] = [222, 222, 222]

        else:
            vis = np.pad(vis, ((2, 2), (2, 2), (0, 0)), 'constant')

        return vis
    
    def n_actions(self):
            return 4

    def move_player(self, dir):
        # restart game if neccesary
        if self.you_lost:
            self.init_game()
            return None

        # move player
        dir = self.DIRS[dir]
        new_player = (self.player[0] + dir[0], self.player[1] + dir[1])
        if not (0 <= new_player[0] < self.size): return False
        if not (0 <= new_player[1] < self.size): return False
        if self.blocked[new_player]: return False
        old_player  = self.player 
        self.player = new_player

        # move enemies
        if self.enemy != self.player:
            self.move_enemy()

        # test positions
        if self.player in self.coins:
            self.scored(1)
            self.coins.remove(self.player)
            self.new_coin()
        if self.player == self.enemy:
            self.damage()
        
        return True


    def spawn_enemy(self):
        self.enemy = tuple(np.random.randint(self.size, size=2))
        while self.blocked[self.enemy] or self.enemy in self.coins or self.enemy == self.player:
            self.enemy = tuple(np.random.randint(self.size, size=2))

    def move_enemy(self):
        if self.chase:
            self.enemy = self.next_step(self.enemy)
        self.timer -= 1
        if self.timer == 0:
            if self.chase:
                self.chase = False
                self.timer = self.STAND_TIME
            else:
                self.chase = True
                self.timer = self.CHASE_TIME
        
    def game_over(self):
        self.you_lost = True

    def damage(self):
        self.scored(-10)
        self.spawn_enemy()
    
    def new_coin(self):
        coin = tuple(np.random.randint(self.size, size=2))
        while self.blocked[coin] or coin in self.coins or coin == self.enemy or coin == self.player:
            coin = tuple(np.random.randint(self.size, size=2))
        self.coins.append(tuple(coin))

    def scored(self, sc=1):
        self.score  += sc
    
    def get_score(self):
        return self.score
    
    def valid_neighbors(self, x, rnd=True):
        x = np.array(x)
        res = []
        for y in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            xy = x + y
            if (xy < 0).any(): continue
            if (xy >= self.size).any(): continue
            xy = tuple(xy)
            if self.blocked[xy]: continue
            res.append(xy)
        if rnd:
            random.shuffle(res)
        return res

    def next_step(self, goal, traps=False):
        """
        Calculates path from player to monster
        using the A* algorithm.
        Returns next monster step.
        """
        kyu = PriorityQueue()
        kyu.put((0, self.player))
        came_from = {self.player: None}
        costs_agg = {self.player: 0}

        while not kyu.empty():
            curr = kyu.get()[1]
            if curr == goal: break

            for next in self.valid_neighbors(curr):
                new_cost = costs_agg[curr] + (5 if traps and self.traps[next] else 1)
                if next not in costs_agg.keys() or new_cost < costs_agg[next]:
                    costs_agg[next] = new_cost
                    kyu.put((new_cost + l2(next, goal), next))
                    came_from[next] = curr
        
        if goal in came_from.keys():
            return came_from[goal]
        else:
            raise RuntimeWarning("no path between monster and player")
            return goal
