import numpy as np
import random
from queue import PriorityQueue
import string
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from mapgen import generate_world
import itertools

GAME_DEBUG = True

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
    C_PTRAP  = [ 60, 150,  60] #rgb(60,150,60)
    C_NEST   = [110, 120, 180] #rgb(110,20,180)

    DIRS = {'u': ( 0,-1),
            'd': ( 0, 1),
            'l': (-1, 0),
            'r': ( 1, 0)}

    def __init__(self, size=50, stretch=8, n_traps=11, n_nests=5):
        self.size      = size
        self.stretch   = stretch
        self.enemies   = []
        self.score     = 0
        self.level     = 0
        self.lives     = 3

        self.maxdown   = self.START_MAXDOWN
        self.cooldown  = self.maxdown
        self.chance    = self.START_CHANCE

        self.blocked, self.traps, self.nests, self.player = generate_world(size)
        self.v_nests = np.zeros_like(self.blocked).astype(bool)
        for ne in self.nests:
            self.visualize_nest(ne)
        
        self.coins = []
        for i in range(10):
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

    def get_visual(self, hud=True):
        """
        creates a visual representation of the game as a numpy array
        """
        vis = np.zeros(self.traps.shape + (3,)).astype(int) + self.C_BG

        vis[self.v_nests] = self.C_NEST
        vis[self.blocked] = self.C_BLOCK
        vis[self.traps]   = self.C_TRAP
        
        for en in self.enemies:
            vis[en] = self.C_ENEMY
        
        for co in self.coins:
            vis[co]   = self.C_COIN

        if self.traps[self.player]:
            vis[self.player] = self.C_PTRAP
        else:
            vis[self.player] = self.C_PLAYER
        
        

        if hud:
            vis = np.pad(vis, ((1, 1), (1, 1), (0, 0)), 'constant')
            vis = np.pad(vis, ((2, 2), (10, 5), (0, 0)), 'constant', constant_values=25)
            vis = np.pad(vis, ((1, 1), (1, 1), (0, 0)), 'constant')
            vis = vis.repeat(self.stretch, 0).repeat(self.stretch, 1)
            
            # right side of HUD
            text = char_to_pixels("Score: {}".format(self.score), fontsize=28).T
            x, y = text.shape
            pad = 3*self.stretch
            vis[-x-pad:-pad, pad:pad+y][text] = [222, 222, 222]

            text = char_to_pixels("Level: {}".format(self.level), fontsize=28).T
            x2, y2 = text.shape
            pad = 3*self.stretch
            vis[-x2-pad:-pad, pad+8+y:pad+8+y+y2][text] = [222, 222, 222]

            # left side of HUD
            text = char_to_pixels("Lives: {}".format("♥"*self.lives), fontsize=28).T
            x, y = text.shape
            pad = 3*self.stretch
            vis[pad:x+pad, pad:pad+y][text] = [222, 222, 222]

            text = char_to_pixels("Chance: {}%".format(self.chance), fontsize=28).T
            x2, y2 = text.shape
            pad = 3*self.stretch
            vis[pad:x2+pad, pad+8+y:pad+8+y+y2][text] = [222, 222, 222]
            
            # bottom side of HUD
            if self.cooldown > 0:
                text  = char_to_pixels("|" * self.cooldown, fontsize=28).T
            x, y = text.shape
            pad = 2*self.stretch
            if self.START_MAXDOWN > self.maxdown:
                text2 = char_to_pixels("|" * (self.START_MAXDOWN - self.maxdown), fontsize=28).T
                x2, y2 = text2.shape
                vis[pad:x2+pad, -pad-y2:-pad][text2] = [222, 22, 22]
            else:
                x2, y2 = 0, 0
            if self.cooldown > 0:
                vis[pad+x2:x+x2+pad, -pad-y:-pad][text] = [222, 222, 222]

        else:
            vis = vis.repeat(self.stretch,0).repeat(self.stretch,1)


        return vis
    
    def move_player(self, dir):
        # move player
        dir = self.DIRS[dir]
        new_player = (self.player[0] + dir[0], self.player[1] + dir[1])
        if not (0 <= new_player[0] < self.size): return
        if not (0 <= new_player[1] < self.size): return
        if self.blocked[new_player]: return
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
            self.scored(-1)
        if self.player in self.enemies:
            self.damage()
        
        return

    def tick_spawns(self):
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
            new_pos = self.next_step(enemy)
            if new_pos is None:
                new_pos = self.player
            if self.traps[new_pos]:
                self.scored()
            else:
                if new_pos not in new_enemies and new_pos not in sort_enemies[i:]:
                    new_enemies.append(new_pos)
                else:
                    new_enemies.append(enemy)
        self.enemies = new_enemies

    def game_over(self):
        pass
        exit(123)

    def damage(self):
        self.scored(-20)
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
        free = np.stack(np.where(((1 - self.blocked) - self.traps).astype(bool))).T
        mx, ms = None, 0
        for i in range(10000):
            x = np.random.randint(0, len(free))
            x = free[x]
            s = sum([np.sqrt(l2(ne, x)) for ne in self.nests])
            if s > ms:
                mx, ms = x, s
        self.nests.append(tuple(mx))
        self.visualize_nest(tuple(mx))


    def nextlevel(self):
        return (self.level + 1) * 50

    def scored(self, sc=1):
        self.score  += sc
        if sc > 0 and self.score > self.nextlevel():
            self.level_up()
    
    def valid_neighbors(self, x, rnd=True):  #TODO: test
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

    def next_step(self, goal):  #TODO: test
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
                new_cost = costs_agg[curr] + 1
                if next not in costs_agg.keys() or new_cost < costs_agg[next]:
                    costs_agg[next] = new_cost
                    kyu.put((new_cost + l2(next, goal), next))
                    came_from[next] = curr
        
        if goal in came_from.keys():
            return came_from[goal]
        else:
            raise RuntimeWarning("no path between monster and player")
            return goal
