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
        lvl_dir = "levels/{}_{}_{}/".format(*args)
        if len([name for name in os.listdir(lvl_dir) if os.path.isfile(lvl_dir + name)]) < max_lvls:
            pregen_level(*args)
        
        sleep(interval)

def pregen_level(size, n_traps=None, n_nests=None):
    lvl_dir = "levels/{}_{}_{}/".format(size, n_traps, n_nests)

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

    START_CHANCE  = 25
    START_MAXDOWN = 12

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

    def __init__(self, size=50, stretch=8, n_traps=None, n_nests=None, easy=False, pregen=2):

        self.size      = size
        self.stretch   = stretch
        self.easy      = easy
        self.n_traps   = n_traps
        self.n_nests   = n_nests
        self.pregen    = pregen
        if easy:
            self.pregen = False

        if self.pregen:
            lvl_dir = "levels/{}_{}_{}/".format(size, n_traps, n_nests)
            if not os.path.exists(lvl_dir):
                os.makedirs(lvl_dir)
            self.pregen = [Process(target=process_pregen, args=((size, n_traps, n_nests),)) for __ in range(pregen)]
            for p in self.pregen:
                p.start()

        self.init_game()

    def init_game(self):
        self.you_lost = False
        self.enemies   = []
        self.score     = 0
        self.level     = 0
        self.lives     = 3

        self.maxdown   = self.START_MAXDOWN
        self.cooldown  = self.maxdown
        self.chance    = self.START_CHANCE

        if self.easy:
            self.blocked = np.zeros((self.size, self.size)).astype(bool)
            self.traps   = np.zeros((self.size, self.size)).astype(bool)
            self.nests   = []
            self.player  = (self.size // 2, self.size // 2)
            self.pregen  = []

        elif self.pregen:
            try:
                lvl_dir  = "levels/{}_{}_{}/".format(self.size, self.n_traps, self.n_nests)
                lvl_name = random.choice([name for name in os.listdir(lvl_dir) if os.path.isfile(lvl_dir + name)])
                self.blocked, self.traps, self.nests, self.player = pickle.load(open(lvl_dir + lvl_name, 'rb'))
                os.remove(lvl_dir + lvl_name)
            except:
                print("Couldn't load level, generating new one")
                self.blocked, self.traps, self.nests, self.player = generate_world(self.size, self.n_traps, self.n_nests)
        else:
            self.blocked, self.traps, self.nests, self.player = generate_world(self.size, self.n_traps, self.n_nests)

        self.v_nests = np.zeros_like(self.blocked).astype(bool)
        for ne in self.nests:
            self.visualize_nest(ne)
        
        self.coins = []
        for _ in range(10):
            self.new_coin()
    
    def visualize_nest(self, ne):
        for dx in [-1, 0, 1]:
            xx = ne[0] + dx
            if xx < 0 or xx >= self.size: continue
            for dy in [-1, 0, 1]:
                yy = ne[1] + dy
                if yy < 0 or yy >= self.size: continue
                if dx == 0 == dy: continue
                self.v_nests[xx,yy] = True
    
    def kill_pregen(self):
        if self.pregen:
            [x.terminate() for x in self.pregen]

    def get_visual(self, hud=True):
        """
        creates a visual representation of the game as a numpy array
        """
        vis = np.zeros(self.traps.shape + (3,)).astype(int) + self.C_BG

        if not self.you_lost:
            vis[self.v_nests] = self.C_NEST
            vis[self.blocked] = self.C_BLOCK
            vis[self.traps]   = self.C_TRAP
            
            for en in self.enemies:
                vis[en] = self.C_ENEMY
            
            for co in self.coins:
                if co in self.enemies:
                    vis[co] = self.C_ECOIN
                else:
                    vis[co] = self.C_COIN

            if self.traps[self.player]:
                vis[self.player] = self.C_PTRAP
            else:
                vis[self.player] = self.C_PLAYER
        
        

        if hud:
            vis = np.pad(vis, ((1, 1), (1, 1), (0, 0)), 'constant')
            vis = np.pad(vis, ((2, 2), (10, 5), (0, 0)), 'constant', constant_values=25)
            vis = np.pad(vis, ((1, 1), (1, 1), (0, 0)), 'constant')
            vis = vis.repeat(self.stretch, 0).repeat(self.stretch, 1)
            
            fs = self.size // 2 + 3 

            # right side of HUD
            text = char_to_pixels("Score: {}".format(self.score), fontsize=fs).T
            x, y = text.shape
            pad = 3*self.stretch
            vis[-x-pad:-pad, pad:pad+y][text] = [222, 222, 222]

            text = char_to_pixels("Level: {}".format(self.level), fontsize=fs).T
            x2, y2 = text.shape
            pad = 3*self.stretch
            vis[-x2-pad:-pad, pad+8+y:pad+8+y+y2][text] = [222, 222, 222]

            # left side of HUD
            text = char_to_pixels("Lives: {}".format("♥"*self.lives), fontsize=fs).T
            x, y = text.shape
            pad = 3*self.stretch
            vis[pad:x+pad, pad:pad+y][text] = [222, 222, 222]

            text = char_to_pixels("Chance: {}%".format(self.chance), fontsize=fs).T
            x2, y2 = text.shape
            pad = 3*self.stretch
            vis[pad:x2+pad, pad+8+y:pad+8+y+y2][text] = [222, 222, 222]
            
            # bottom side of HUD
            if self.cooldown > 0:
                text  = char_to_pixels("|" * self.cooldown, fontsize=fs).T
            x, y = text.shape
            pad = 2*self.stretch
            if self.START_MAXDOWN > self.maxdown:
                text2 = char_to_pixels("|" * (self.START_MAXDOWN - self.maxdown), fontsize=fs).T
                x2, y2 = text2.shape
                vis[pad:x2+pad, -pad-y2:-pad][text2] = [222, 22, 22]
            else:
                x2, y2 = 0, 0
            if self.cooldown > 0:
                vis[pad+x2:x+x2+pad, -pad-y:-pad][text] = [222, 222, 222]

        else:
            vis = np.pad(vis, ((2, 2), (2, 2), (0, 0)), 'constant')
            vis[-self.level - 1:, 0] = [255, 255, 255]
            vis[:self.lives*2, 0] = [255, 32, 32]
            vis[:self.cooldown*2, -1] = [32, 32, 255]



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
        self.move_enemies()

        self.tick_spawns()

        # test positions
        if self.player in self.coins:
            self.scored(5)
            self.coins.remove(self.player)
            self.new_coin()
        if self.traps[self.player] and self.traps[old_player]:
            self.scored(-25)
        if self.player in self.enemies:
            self.damage()
        
        return True

    def tick_spawns(self):
        if self.easy: return
        if self.chance >= np.random.randint(0,100):
            self.cooldown -= 1
            if self.cooldown < 0:
                self.cooldown = self.maxdown
                for nest in self.nests:
                    if self.chance >= np.random.randint(0,100):
                        self.enemies.append(nest)

    def move_enemies(self):
        new_enemies = []
        sort_enemies = sorted(self.enemies, key=lambda x:l2(x, self.player))
        for i, enemy in enumerate(sort_enemies):
            rand = np.random.randint(10)
            if rand < 3: # 30% chance
                new_pos = self.next_step(enemy, traps=True)
            elif rand < 4: # 10% chance
                new_pos = self.valid_neighbors(enemy)[0]
            else: # 50% chance
                new_pos = self.next_step(enemy)
            if new_pos is None:
                new_pos = self.player
            if self.traps[new_pos]:
                if np.sqrt(l2(new_pos, self.player)) < self.size * 0.4:
                    self.scored()
            else:
                if new_pos not in new_enemies and new_pos not in sort_enemies[i:]:
                    new_enemies.append(new_pos)
                else:
                    new_enemies.append(enemy)
        self.enemies = new_enemies

    def game_over(self):
        self.you_lost = True

    def damage(self):
        self.scored(-100)
        self.lives -= 1
        if self.lives < 0:
            self.game_over()
        self.enemies = [x for x in self.enemies if l2(self.player, x) > 20]
    
    def new_coin(self):
        coin = tuple(np.random.randint(self.size, size=2))
        while self.blocked[coin] or self.traps[coin] or self.v_nests[coin] or coin in self.nests or coin in self.coins:
            coin = tuple(np.random.randint(self.size, size=2))
        self.coins.append(tuple(coin))
    
    def level_up(self):
        if self.level % 3 == 0:
            if self.maxdown > 4:
                self.maxdown -= 1
                self.cooldown = self.maxdown
        elif self.level % 3 == 1:
            self.chance += 5
            if self.chance > 99:
                self.chance = 99
        else:
            self.spawn_nest()
            self.cooldown = self.maxdown
        self.level += 1
    
    def spawn_nest(self):
        if self.easy: return
        free = np.stack(np.where(((1 - self.blocked) - self.traps).astype(bool))).T
        mx, ms = None, 0
        for x in free:
            dists = [np.sqrt(l2(ne, x)) for ne in self.nests]
            if min(dists) < 2:
                continue
            s = sum(dists) - (len(self.nests) // 2 * np.sqrt(l2(self.player, x)))
            if s > ms:
                mx, ms = x, s
        self.nests.append(tuple(mx))
        self.visualize_nest(tuple(mx))


    def nextlevel(self):
        return (self.level + 1) * 50

    def scored(self, sc=1):
        self.score  += sc
        if sc > 0 and self.score >= self.nextlevel():
            self.level_up()
    
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

    def next_step(self, goal, traps=False):  #TODO: test (maybe change to l1 dist?)
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
