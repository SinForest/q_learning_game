from model import NetworkSmallDuell
from mechanics import Game
from agent import Agent
import numpy as np
import argparse
from time import sleep

import pygame as pg
import os
import sys
import torch

DIR = ["↑", "↓", "←", "→"]

def resume(fp):
    if os.path.isfile(fp):
        cp = torch.load(fp, map_location={'cuda:0': 'cpu'})
        model = NetworkSmallDuell(32, 4)
        model2 = NetworkSmallDuell(32, 4)
        model.load_state_dict(cp['state_dict'])
        model2.load_state_dict(cp['state_dict2'])
        return model, model2
    else:
        raise FileNotFoundError("File {} not found.".format(fp))

def print_q_values(a, aa):
    s = lambda i: "\33[31m{: <6.2f}\33[m".format(a[i]) if aa == i else "{: <6.2f}".format(a[i])
    print("  " + " ".join(["{}:{}".format(DIR[x], s(x)) for x in range(len(a))]), end="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Observe the agent')
    parser.add_argument("--trace", "-t", help="show trace of player", action="store", type=float, default=0)
    parser.add_argument("checkpoint", help="checkpoint to load")
    args = parser.parse_args()

    game  = Game(easy=True, size=28)
    model, model2 = resume(args.checkpoint)
    model.eval()
    model2.eval()
    ag = Agent(model, cuda=False) # for to_var
    resolution = game.get_visual().shape[:2]
    screen = pg.display.set_mode(resolution)
    field  = None
    auto   = False
    score = 0
    stuck_count = 0

    while(True):
        if np.random.randint(0,2):
            model, model2 = model2, model
        state = game.get_visual(hud=False)
        if field is not None:
            field = field * args.trace + (1-args.trace) * game.get_visual(hud=True)
        else:
            field = game.get_visual(hud=True)

        last_score = score
        score = game.get_score()
        if score == last_score:
            stuck_count += 1
            if auto and stuck_count > 150:
                game.game_over()
                game.move_player(None)
        else:
            stuck_count = 0

        pg.surfarray.blit_array(screen, field)
        a = model(ag.to_var(state)).data[0] # Tensor dim=(4)
        m = False
        while not m:
            aa = a.max(0)[1][0] # argmax as scalar
            print_q_values(a, aa)
            pg.display.flip()
            if not auto:
                print("\r",end="")
                inp = input()
            else:
                print()
                sleep(0.05)
            if inp == "x":
                game.game_over()
                game.move_player(None)
                print("-"*35)
                break
            elif inp == "a":
                game.game_over()
                game.move_player(None)
                print("-"*35)
                auto = True
                inp = None
                break
            if a.max() == float("-inf"):
                #this should never happen!
                print("No valid moves. Exiting.")
                exit(123)
            m = game.move_player(aa)
            a[aa] = float("-inf")

