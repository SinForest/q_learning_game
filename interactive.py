import pygame as pg
import numpy as np
from chrono import Timer
import argparse

parser = argparse.ArgumentParser(description='Start the game')
parser.add_argument("--size",  "-s", help="hight/width of the game field", action="store", type=int, default=50)
parser.add_argument("--easy",  "-e", help="trigger easy mode", action="store_true")
parser.add_argument("--plain", "-p", help="draw plain HUD", action="store_true")
parser.add_argument("--lite",  "-l", help="run alternative lite version", action="store_true")
args = parser.parse_args()
draw_hud = not args.plain

if args.lite:
    from mechanics_lite import Game
    game = Game(size=args.size)
else:
    from mechanics import Game
    game = Game(size=args.size, easy=args.easy)

resolution = game.get_visual(hud=draw_hud).shape[:2]
screen = pg.display.set_mode(resolution)

DIRS = {pg.K_DOWN: 'd',
        pg.K_UP: 'u',
        pg.K_LEFT: 'l',
        pg.K_RIGHT: 'r'}

while True:
    field = game.get_visual(hud=draw_hud)

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