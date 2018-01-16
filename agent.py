import torch
from torch import Tensor
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import trange

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
    losses = torch.abs((prediction - target[:, None]) * actions)
    try:
        return losses.sum()
    except RuntimeError:
        return None #Quickfix


class Agent:

    def __init__(self, model, cuda=True, view=None):
        self.cuda = cuda
        if cuda:
            self.model = model.cuda()
        else:
            self.model = model
        self.opti = torch.optim.SGD(model.parameters(), lr=0.0001)
        self.memory = []
        self.view = bool(view)
        if view:
            self.screen = pg.display.set_mode(view)


    def train(self, game, n_epochs=None, batch_size=256, gamma=0.9, epsilons=(1.0, 0.1, 0.001), max_steps=None, save_interval=10, memory_size=25600):


        # TODO: setup game
        n_actions = game.n_actions()

        self.model.eval()

        epsilon = epsilons[0]
        train_epoch = 0
        highscore = float("-inf")

        for epoch in while_range(n_epochs):

            try:

                print("### Starting Game-Epoch {} \w eps={:.2f} ###".format(epoch, epsilon))
                steps = 0

                last_score = game.get_score()
                S = game.get_visual(hud=False)

                while not game.you_lost:

                    # choose action via epsilon greedy:
                    if np.random.rand() < epsilon:
                        a = np.random.randint(n_actions)
                    else:
                        a = self.model(self.to_var(S)).max(1)[1].data[0]

                    # move player, calculate reward
                    moved = game.move_player(a)
                    score = game.get_score()
                    r = score - last_score
                    last_score = score

                    # penalize invalid movements
                    if moved == False:
                        r -= 200

                    # get next state
                    Sp = game.get_visual(hud=False)
                    self.memory.append((S, a, r, Sp))
                    S  = Sp

                    # render view for spectating
                    if self.view:
                        pg.surfarray.blit_array(self.screen, game.get_visual())
                        pg.display.flip()

                    # train if memory is full
                    if len(self.memory) >= memory_size:
                        print("\33[32m" + "\n" + "-"*15 + "TRAINING PHASE" + "-"*15)
                        print(" --> starting training #{}...".format(train_epoch))
                        loss = self.train_on_memory(gamma, batch_size)
                        self.memory = []
                        print(" --> loss: {:.4f}".format(loss))
                        epsilon -= epsilons[2]
                        epsilon = max(epsilon, epsilons[1])
                        train_epoch += 1

                        if train_epoch % save_interval == 0 or train_epoch + 1 == n_epochs:
                            print(" --> starting testing...")
                            sc = [self.play(game, max_steps) for __ in trange(20, ncols=44)]
                            sc = [x for x in sc if x is not None]
                            print(" --> best: {}, avg: {:.1f}".format(max(sc), sum(sc)/len(sc)))

                            print("   --> writing model to file...")
                            self.save(train_epoch, highscore)
                            highscore = 0
                        print("-"*(30 + len("TRAINING PHASE")) + "\33[m")

                    steps += 1
                    if max_steps and steps > max_steps:
                        game.game_over()

                #[end] while not game.you_lost

                print("  --> end of round, {}score: {}{}\n".format("\33[33m", game.get_score(), "\33[m"))
                if game.get_score() > highscore:
                    highscore = game.get_score()
                game.move_player(None) #restart game

            except KeyboardInterrupt:

                exit(123)
        
        #[end] for epoch in while_range(n_epochs)

    def play(self, game, max_steps):
        game.game_over()
        game.move_player(None)
        steps = 0
        self.model.eval()
        while steps < max_steps and not game.you_lost:
            S = game.get_visual(hud=False)
            a = self.model(self.to_var(S))#.max(1)[1].data[0]
            if self.cuda:
                a = a.cpu()
            a = a.data.numpy()[0]
            m = False
            while not m:
                aa = np.argmax(a)
                if a.max() == -np.inf:
                    #this shold never happen!
                    print("     no valid moves...")
                    return None
                m = game.move_player(aa)
                a[aa] = -np.inf
            steps += 1
        
        return game.get_score()
            

    def train_on_memory(self, gamma, batch_size):

        random.shuffle(self.memory)
        n_iter = int(np.ceil(len(self.memory) / batch_size))
        losses = 0
        omit = 0

        for i in trange(n_iter, ncols=44):
            (S, a, r, Sp) = zip(*(self.memory[i*batch_size:(i+1)*batch_size]))

            Sp = self.to_var(np.stack(Sp))

            self.model.eval()
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
            if loss is None:
                omit += len(self.memory[i*batch_size:(i+1)*batch_size])
                continue
            loss.backward()
            self.opti.step()
            losses += loss.data[0]
        self.model.eval()
        return losses / (len(self.memory) - omit)

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
    
    def save(self, epoch, highscore):
        d = {'epoch'     : epoch,
             'state_dict': self.model.state_dict(),
             'optimizer' : self.opti.state_dict()}
        torch.save(d, "snapshot_{}.nn".format(epoch))

if __name__ == "__main__":
    from mechanics import Game
    from model import *
    import argparse

    parser = argparse.ArgumentParser(description='Train the agent')
    parser.add_argument("--cuda", "-c", help="use CUDA", action="store_true")
    parser.add_argument("--resume", "-r", help="resume from snapshot", action="store", type=str, default="")
    args = parser.parse_args()

    game  = Game(easy=True, size=28)
    inp   = game.get_visual(hud=False).shape[0]
    net   = NetworkSmall(inp, 4)

    if args.resume:
        pass #TODO

    agent = Agent(net, cuda=args.cuda)

    agent.train(game, batch_size=512, max_steps=1500, save_interval=10, memory_size=51200)
