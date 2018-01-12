import torch
from torch import Tensor
from torch.autograd import Variable
import numpy as np

from model import Network

def while_range(n):
    """
    like range(n), if n scalar
    like while(True), in f None
    """
    if n is None:
        n = float("inf")
    i = 0
    while i < n:
        yield i
        i += 1

def to_var(x):
    """
    converts one sample (3dim) to a variable (4dim)
    """
    return Variable(Tensor(x.transpose(2,0,1)[np.newaxis]))

class Agent:

    def __init__(self, model):
        self.model = model

    def train(self, game, n_epochs=None):
        # TODO: setup game
        
        for epoch in while_range(n_epochs):
            last_score = game.get_score()
            
            while(True):
                #TODO: epsilon greedy

                x = game.get_visual(hud=False)
                action = self.model(to_var(x)).data.numpy.argmax()

                game.move_player(action)

                score = game.get_score()
                r = score - last_score
                last_score = score





