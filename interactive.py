import pygame as pg
import numpy as np
from chrono import Timer
from mechanics import Game
import argparse
import os

parser = argparse.ArgumentParser(description='Start the game')
parser.add_argument("--size", "-s", help="hight/width of the game field", action="store", type=int, default=50)
parser.add_argument("--easy", "-e", help="trigger easy mode", action="store_true")
args = parser.parse_args()

# game = Game(30, 16, 5, 3)
game = Game(size=args.size, easy=args.easy)
resolution = game.get_visual().shape[:2]
screen = pg.display.set_mode(resolution)

DIRS = {pg.K_DOWN: 'd',
        pg.K_UP: 'u',
        pg.K_LEFT: 'l',
        pg.K_RIGHT: 'r'}

while True:
    field = game.get_visual()

    pg.surfarray.blit_array(screen, field)
    pg.display.flip()

    for event in pg.event.get():
        if event.type == pg.KEYDOWN:

            if event.key == pg.K_ESCAPE:
                game.kill_pregen()
                exit()

            if event.key in DIRS.keys():
                game.move_player(DIRS[event.key])

            if event.key == pg.K_c:
                game.scored(20)
            
            if event.key == pg.K_p:
                fn = "./debug_screens.npy"
                shot = game.get_visual(hud=False)
                shot = shot[None,:]
                if os.path.isfile(fn):
                    f = np.load(fn)
                    shot = np.concatenate([f, shot])
                np.save(fn, shot)