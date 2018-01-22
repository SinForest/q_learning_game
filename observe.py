from model import NetworkSmallDuell
from mechanics import Game
from agent import Agent

import pygame as pg
import os
import sys
import torch

DIR = ["↑", "↓", "←", "→"]

def resume(fp):
    if os.path.isfile(fp):
        cp = torch.load(fp, map_location={'cuda:0': 'cpu'})
        model = NetworkSmallDuell(32, 4)
        model.load_state_dict(cp['state_dict'])
        return model
    else:
        raise FileNotFoundError("File {} not found.".format(fp))

def print_q_values(a, aa):
    s = lambda i: "\33[31m{:.2f}\33[m".format(a[i]) if aa == i else "{:.2f}".format(a[i])
    print("  " + " ".join(["{}:{}".format(DIR[x], s(x)) for x in range(len(a))]), end="\r")

if __name__ == "__main__":
    game  = Game(easy=True, size=28)
    model = resume(sys.argv[1])
    model.eval()
    ag = Agent(model, cuda=False) # for to_var
    resolution = game.get_visual().shape[:2]
    screen = pg.display.set_mode(resolution)

    while(True):
        state = game.get_visual(hud=False)
        field = game.get_visual(hud=True)
        pg.surfarray.blit_array(screen, field)
        a = model(ag.to_var(state)).data[0] # Tensor dim=(4)
        m = False
        while not m:
            aa = a.max(0)[1][0] # argmax as scalar
            print_q_values(a, aa)
            pg.display.flip()
            if input() == "x":
                game.game_over()
                game.move_player(None)
                print("-"*35)
                break
            if a.max() == float("-inf"):
                #this should never happen!
                print("No valid moves. Exiting.")
                exit(123)
            m = game.move_player(aa)
            a[aa] = float("-inf")

