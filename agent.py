import torch
from torch import Tensor
from torch.autograd import Variable
import numpy as np

import pygame as pg

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

def action_loss(prediction, actions, target):
    n_actions = prediction.size(1)
    actions = Variable(Tensor(np.eye(n_actions)[actions])) #one-hot
    losses = ((prediction - target[:, None]) * actions) ** 2
    return (losses / losses.size(0)).sum()


class Agent:

    def __init__(self, model, cuda=True, view=None):
        self.cuda = cuda
        if cuda:
            self.model = model
        else:
            self.model = model.cuda()
        self.opti = torch.optim.SGD(model.parameters(), lr=0.0000001)
        self.memory = []
        self.view = bool(view)
        if view:
            self.screen = pg.display.set_mode(view)


    def train(self, game, n_epochs=100, batch_size=256, gamma=0.9, epsilons=[1.0, 0.1]):

        # TODO: setup game
        n_actions = game.n_actions()

        self.model.eval()

        eps_delta = (epsilons[0] - epsilons[1]) / (n_epochs - 1)

        for epoch in while_range(n_epochs):

            if type(epsilons) == list:
                epsilon = epsilons[0] - eps_delta * epoch
            else:
                epsilon = epsilons
            
            print("### Starting Epoch {} \w eps={} ###".format(epoch, epsilon))


            last_score = game.get_score()
            S = game.get_visual(hud=False)
            
            while not game.you_lost:

                # choose action via epsilon greedy:
                if np.random.rand() < epsilon:
                    a = np.random.randint(n_actions)
                else:
                    a = self.model(self.to_var(S)).data.numpy().argmax()
                
                game.move_player(a)

                score = game.get_score()
                r = score - last_score
                last_score = score

                Sp = game.get_visual(hud=False)
                self.memory.append((S, a, r, Sp))
                S  = Sp
                if self.view:
                    pg.surfarray.blit_array(self.screen, game.get_visual())
                    pg.display.flip()

                if len(self.memory) >= batch_size:
                    print("starting training, current score: {}".format(game.get_score()))
                    loss = self.train_on_memory(gamma)
                    self.memory = []
                    print("finished training, loss: {}".format(loss))
            
            game.move_player(None) #restart game
    
    def train_on_memory(self, gamma):

        (S, a, r, Sp) = zip(*self.memory)
        Sp = self.to_var(np.stack(Sp))

        Q_max = self.model(Sp).data.max(1)[0] # Tensor containing maximum Q-value per S'
        r = Tensor(np.array(r))
        target = Variable(r + Q_max * gamma)

        S = self.to_var(np.stack(S))
        self.model.train()
        self.opti.zero_grad()
        pred = self.model(S)
        loss = action_loss(pred, list(a), target)
        loss.backward()
        self.opti.step()
        self.model.eval()
        return loss.data[0]
    
    def to_var(self, x):
        """
        converts one sample (3dim) to a variable (4dim)
        or a batch of samples (3dim) to variable (4dim)
        """
        if x.ndim == 3:
            x = Tensor(x.transpose(2,0,1)[np.newaxis])
        elif x.ndim == 4:
            x = Tensor(x.transpose(0,3,1,2))
        else:
            raise RuntimeError("wrong input dimensions")
        if self.cuda:
            return Variable(x / 128 - 1).cuda()
        else:
            return Variable(x / 128 - 1)



if __name__ == "__main__":
    from mechanics import Game
    from model import Network
    game  = Game()
    inp   = game.get_visual(hud=False).shape[0]
    net   = Network(inp, 4)
    # agent = Agent(net, view=game.get_visual().shape[:2])
    agent = Agent(net)

    agent.train(game, batch_size=32)
