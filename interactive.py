import pygame as pg
import numpy as np
from chrono import Timer
from mechanics import Game

screen = pg.display.set_mode((400, 400))
game = Game()

DIRS = {pg.K_DOWN: 'd',
        pg.K_UP: 'u',
        pg.K_LEFT: 'l',
        pg.K_RIGHT: 'r'}

while True:
    field = game.get_visual().repeat(8,0).repeat(8,1)
    pg.surfarray.blit_array(screen, field)
    pg.display.flip()

    for event in pg.event.get():
        if event.type == pg.KEYDOWN:

            if event.key == pg.K_ESCAPE:
                exit()

            if event.key in DIRS.keys():
                game.move_player(DIRS[event.key])