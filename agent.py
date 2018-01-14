import torch
from torch import Tensor
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

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
    if prediction.is_cuda:
        actions = actions.cuda()
    # print(type(prediction.data), type(target.data), type(actions.data))
    losses = ((prediction - target[:, None]) * actions) ** 2
    return (losses / losses.size(0)).sum()


class Agent:

    def __init__(self, model, cuda=True, view=None):
        self.cuda = cuda
        if cuda:
            self.model = model.cuda()
        else:
            self.model = model
        self.opti = torch.optim.SGD(model.parameters(), lr=0.0000001)
        self.memory = []
        self.view = bool(view)
        if view:
            self.screen = pg.display.set_mode(view)


    def train(self, game, n_epochs=1000, batch_size=256, gamma=0.9, epsilons=[1.0, 0.1], max_steps=None, save_interval=10):

        if n_epochs is None and type(epsilons) == list:
            raise RuntimeError("n_epochs or epsilons must be fixed scalar")

        # TODO: setup game
        n_actions = game.n_actions()

        self.model.eval()

        if type(epsilons) == list:
            eps_delta = (epsilons[0] - epsilons[1]) / (n_epochs - 1)

        for epoch in while_range(n_epochs):

            try:

                if type(epsilons) == list:
                    epsilon = epsilons[0] - eps_delta * epoch
                else:
                    epsilon = epsilons

                print("### Starting Epoch {} \w eps={:.2f} ###\n".format(epoch, epsilon))
                steps = 0

                last_score = game.get_score()
                S = game.get_visual(hud=False)

            

                while not game.you_lost:

                    # choose action via epsilon greedy:
                    if np.random.rand() < epsilon:
                        a = np.random.randint(n_actions)
                    else:
                        a = self.model(self.to_var(S)).max(1)[1].data[0]

                    moved = game.move_player(a)

                    score = game.get_score()
                    r = score - last_score
                    last_score = score

                    # penalize invalid movements
                    if moved == False:
                        r -= 200

                    Sp = game.get_visual(hud=False)
                    self.memory.append((S, a, r, Sp))
                    S  = Sp
                    if self.view:
                        pg.surfarray.blit_array(self.screen, game.get_visual())
                        pg.display.flip()

                    if len(self.memory) >= batch_size or game.you_lost:
                        print("  --> starting training, {}score: {}{}  [{}]".format("\33[33m", game.get_score(), "\33[37m", epoch))
                        print("     -->                 {}lives: {}{}".format("\33[31m", "♥" * game.lives, "\33[37m"))
                        loss = self.train_on_memory(gamma)
                        self.memory = []
                        print("     -->                  {}loss: {:.4f}{}\n".format("\33[32m", loss, "\33[37m"))
                    
                    steps += 1
                    if max_steps and steps > max_steps:
                        game.game_over()

                #[end] while not game.you_lost

                if (epoch + 1) % save_interval == 0 or epoch == n_epochs - 1:
                    print("  --> writing model to file...\n")
                    self.save(epoch)
            
                game.move_player(None) #restart game

            except KeyboardInterrupt:

                plt.imshow(S.astype(np.uint8))
                plt.show()
                exit(123)
        
        #[end] for epoch in while_range(n_epochs)

            

    def train_on_memory(self, gamma):

        (S, a, r, Sp) = zip(*self.memory)
        Sp = self.to_var(np.stack(Sp))

        Q_max = self.model(Sp).data.max(1)[0] # Tensor containing maximum Q-value per S'
        r = Tensor(np.array(r))
        if self.cuda:
            r = r.cuda()
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
        x = Variable(x / 128 - 1)
        if self.cuda:
            return x.cuda()
        else:
            return x
    
    def save(self, epoch):
        d = {'epoch'     : epoch,
             'state_dict': self.model.state_dict(),
             'optimizer' : self.opti.state_dict()}
        torch.save(d, "snapshot_{}.nn".format(epoch))



if __name__ == "__main__":
    from mechanics import Game
    from model import Network
    game  = Game()
    inp   = game.get_visual(hud=False).shape[0]
    net   = Network(inp, 4)
    # agent = Agent(net, view=game.get_visual().shape[:2])
    agent = Agent(net)

    agent.train(game, batch_size=512, max_steps=5120, save_interval=20, n_epochs=3000)
