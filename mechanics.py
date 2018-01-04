import numpy as np
import random

GAME_DEBUG = True


class Game:

    C_BG     = [ 50,  50,  50]
    C_TRAP   = [  0, 255,   0]
    C_BLOCK  = [191, 191, 191]
    C_PLAYER = [  0,   0, 255]
    C_ENEMY  = [255,   0,   0]
    C_COIN   = [255, 255,   0]

    DIRS = {'u': ( 0,-1),
            'd': ( 0, 1),
            'l': (-1, 0),
            'r': ( 1, 0)}

    def __init__(self, size=50):
        self.player  = (size // 2, size // 2)
        self.blocked = np.zeros((size, size)).astype(bool)
        self.traps   = np.zeros((size, size)).astype(bool)
        self.size    = size
        self.enemies = []
        self.score   = 0

        if GAME_DEBUG:  # new function for world generation here
            for i in range(30):
                x = np.random.randint(size, size=2)
                self.blocked[x[0]-1:x[0]+1, x[1]-1:x[1]+1] = 1
            for i in range(30):
                x = np.random.randint(size, size=2)
                if bool(random.getrandbits(1)):
                    self.traps[x[0], x[1]-2:x[1]+2] = 1
                else:
                    self.traps[x[0]-2:x[0]+2, x[1]] = 1
            for i in range(20):
                x = tuple(np.random.randint(size, size=2))
                while self.blocked[x] or self.traps[x]:
                    x = tuple(np.random.randint(size, size=2))
                self.enemies.append(x)
        self.new_coin()
    
    def get_visual(self):
        """
        creates a visual representation of the game as a numpy array
        """
        vis = np.zeros(self.traps.shape + (3,)).astype(int) + self.C_BG

        vis[self.traps]   = self.C_TRAP
        vis[self.blocked] = self.C_BLOCK

        for en in self.enemies:
            vis[en] = self.C_ENEMY
        
        vis[self.coin]   = self.C_COIN
        vis[self.player] = self.C_PLAYER

        return vis
    
    def move_player(self, dir):
        # move player
        dir = self.DIRS[dir]
        new_player = (self.player[0] + dir[0], self.player[1] + dir[1])
        if not (0 <= new_player[0] < self.size): return
        if not (0 <= new_player[1] < self.size): return
        if self.blocked[new_player]: return
        self.player = new_player

        # move enemies
        self.move_enemies()

        # test positions
        if self.coin == self.player:
            self.scored()
            self.new_coin()
        if self.player in self.enemies:
            self.game_over()
        
        return

    def move_enemies(self):
        pass
    
    def game_over(self):
        pass
        exit(123)
    
    def new_coin(self):
        self.coin = tuple(np.random.randint(self.size, size=2))
        while self.blocked[self.coin] or self.traps[self.coin]:
            self.coin = tuple(np.random.randint(self.size, size=2))
    
    def scored(self):
        self.score += 1
