import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import trange, tqdm
from copy import deepcopy
import os

TERM = {'y'  : "\33[33m",
        'g'  : "\33[32m",
        'c'  : "\33[36m",
        'clr': "\33[m"}

def while_range(n, start=0):
    """
    like range(n), if n scalar
    like while(True), in f None
    """
    if n is None:
        n = float("inf")
    i = start
    while i < n:
        yield i
        i += 1

class Memory:

    def __init__(self, size, eps=1e-5, alpha=1):

        self.size  = size
        self.mem   = []
        self.pri   = []
        self.pos   = 0
        self.eps   = eps
        self.alpha = alpha
    
    def store(self, S, a, r, Sp, err=0):
        if len(self.mem) < self.size:
            self.mem.append((S, a, r, Sp))
            self.pri.append((err + self.eps) ** self.alpha)
        else:
            self.mem[self.pos] = (S, a, r, Sp)
            self.pos = (self.pos + 1) % self.size
    
    def sample(self, batch_size):
        p   = np.array(self.pri)
        p  /= p.sum()
        idx =  np.random.choice(len(self.mem), size=batch_size, p=p, replace=False)
        return np.array(self.mem)[idx]

    def pickle(self):
        import pickle
        pickle.dump(self.mem, open("./memory.p", "wb"))

    def __len__(self):
        return len(self.mem)

class Agent:

    def __init__(self, model, cuda=True, view=None, memory_size=1000, opti_state=None, lr=None, model2=None):
        # generate or load model for DDQN
        if model2 is None:
            self.target_model = deepcopy(model)
        else:
            self.target_model = model2
        
        # move models to CUDA device
        self.cuda = cuda
        if cuda:
            self.model = model.cuda()
            self.target_model = self.target_model.cuda()
        else:
            self.model = model
        
        # init and load optimizer
        self.opti   = torch.optim.RMSprop(model.parameters(),  lr=(lr if lr else 0.001))
        self.t_opti = torch.optim.RMSprop(model2.parameters(), lr=(lr if lr else 0.001))
        if opti_state:
            self.opti.load_state_dict(opti_state['optimizer'])
            self.t_opti.load_state_dict(opti_state['optimizer2'])
        
        self.memory = Memory(memory_size)

        self.view = bool(view)
        if view:
            import pygame as pg
            self.screen = pg.display.set_mode(view)


    def train(self, game, n_epochs=None, batch_size=256, gamma=0.85, epsilons=None, max_steps=None, save_interval=10, move_pen=1, observe=0, start_epoch=0):

        n_actions = game.n_actions()
        self.model.eval()
        self.target_model.eval()

        if epsilons is None:
            epsilons = (0.9, 0.05, 900)
        eps = lambda s:epsilons[1] + (epsilons[0] - epsilons[1]) * np.exp(-s / epsilons[2])

        for epoch in while_range(n_epochs, start=start_epoch):

            epsilon = eps(epoch)
            print("### Starting Game-Epoch {} \w eps={:.2f} ###".format(epoch, epsilon))

            last_score = game.get_score()
            S = game.get_visual(hud=False)
            loss = 0
            n_lo = 0

            #while not game.you_lost:
            for steps in trange(max_steps, ncols=50):

                # stop epoch if game is lost
                if game.you_lost:
                    break

                # choose action via epsilon greedy:
                self.switch_models()
                Q_val = self.model(self.to_var(S))
                if np.random.rand() < epsilon:
                    a = np.random.randint(n_actions)
                else:
                    a = Q_val.max(1)[1].data[0]

                # move player, calculate reward
                moved = game.move_player(a)
                score = game.get_score()
                r = score - last_score - move_pen
                last_score = score

                # penalize invalid movements
                if moved == False:
                    r -= 10
                
                # get next state
                Sp = game.get_visual(hud=False)

                # calc error for priority
                Q_val = Q_val.max(1)[0].data[0]
                Q_max = self.target_model(self.to_var(Sp)).max(1)[0].data[0]
                err   = np.abs(Q_val - (r + gamma * Q_max))

                # save transition
                self.memory.store(S, a, r, Sp, err)
                S  = Sp

                # render view for spectating
                if self.view:
                    pg.surfarray.blit_array(self.screen, game.get_visual())
                    pg.display.flip()

                # train if memory is sufficiently full
                if len(self.memory) >= max(batch_size, observe):
                    loss += self.train_on_memory(gamma, batch_size)
                    n_lo += 1

            #[end] for steps in trange(max_steps, ncols=50)
            game.game_over()
            loss = (loss / n_lo if n_lo > 0 else -1)
            
            print("  --> end of round, {}score: {}{}, {}loss:{:.4f}{}\n".format(TERM['y'], game.get_score(), TERM['clr'],
                                                                                TERM['g'], loss, TERM['clr'],))

            if (epoch % save_interval == 0 and len(self.memory) >= observe) or epoch + 1 == n_epochs:
                print(TERM['c'] + " --> starting testing...")
                sc = [self.play(game, max_steps) for __ in trange(20, ncols=44)]
                sc = [x for x in sc if x is not None]
                print(" --> best: {}, avg: {:.2f}".format(max(sc), sum(sc)/len(sc)))

                print("   --> writing model to file...\n" + TERM['clr'])
                self.save(epoch)
                # self.memory.pickle()

            game.move_player(None) #restart game

        #[end] for epoch in while_range(n_epochs)
    
    def switch_models(self, force=False):
        if force or np.random.randint(0,2):
            self.model, self.target_model = self.target_model, self.model
            self.opti, self.t_opti = self.t_opti, self.opti

    def play(self, game, max_steps):
        game.game_over()
        game.move_player(None)
        steps = 0
        self.model.eval()
        self.target_model.eval()
        while steps < max_steps and not game.you_lost:
            self.switch_models()
            S = game.get_visual(hud=False)
            a = self.model(self.to_var(S)).data[0] # Tensor dim=(4)
            m = False
            while not m:
                aa = a.max(0)[1][0] # argmax as scalar
                if a.max() == -np.inf:
                    #this should never happen!
                    print("     no valid moves...")
                    return None
                m = game.move_player(aa)
                a[aa] = -np.inf
            steps += 1
        game.game_over()
        return game.get_score()
            

    def train_on_memory(self, gamma, batch_size):

        (S, a, r, Sp) = zip(*(self.memory.sample(batch_size)))

        S  = self.to_var(np.stack(S))
        a  = Variable(LongTensor(a).cuda() if self.cuda else LongTensor(a)).view(-1, 1)
        r  = Tensor(r).cuda() if self.cuda else Tensor(r)
        Sp = self.to_var(np.stack(Sp))

        self.switch_models()

        self.target_model.eval()
        Q_max = self.target_model(Sp).data.max(1)[0] # Variable containing maximum Q-value per S'
        target = Variable(r + Q_max * gamma)

        self.model.train()
        self.opti.zero_grad()
        pred = self.model(S).gather(1, a)

        loss = nn.functional.l1_loss(pred, target)
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
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
        x = Variable(x / 127.5 - 1)
        if self.cuda:
            return x.cuda()
        else:
            return x
    
    def save(self, epoch):
        #TODO: save meta data to recreate model (inp_size, n_actions, network type)
        #      maybe also store Game information (which would include some model meta data)
        d = {'epoch'      : epoch,
             'state_dict' : self.model.state_dict(),
             'state_dict2': self.model2.state_dict(),
             'optimizer'  : self.opti.state_dict()
             'optimizer2' : self.opti2.state_dict()}
        torch.save(d, "snapshot_{}.nn".format(epoch))

if __name__ == "__main__":
    from mechanics import Game
    from model import NetworkSmallDuell
    import argparse

    parser = argparse.ArgumentParser(description='Train the agent')
    parser.add_argument("--cuda", "-c", help="use CUDA", action="store_true")
    parser.add_argument("--resume", "-r", help="resume from snapshot", action="store", type=str, default="")
    parser.add_argument("--epsilon", "-e", help="fixed epsilon", action="store", type=float, default=None)
    parser.add_argument("--epoch", help="starting epoch", action="store", type=int, default=0)
    parser.add_argument("--lr", help="learning rate for RMSProp", action="store", type=float, default=None)
    parser.add_argument("--size", "-s", help="size of game", action="store", type=int, default=28)
    args = parser.parse_args()

    game   = Game(easy=True, size=args.size)
    inp    = game.get_visual(hud=False).shape[0]
    net    = NetworkSmallDuell(inp, 4)
    ostate = None
    net2   = None

    if args.resume:
        if os.path.isfile(args.resume):
            cp = torch.load(args.resume, map_location={'cuda:0': 'cpu'})
            net.load_state_dict(cp['state_dict'])
            net2.load_load_state_dict(cp['state_dict_2'])
            ostate = cp
        else:
            raise FileNotFoundError("File {} not found.".format(args.resume))
    
    if args.epsilon is not None:
        args.epsilon = (args.epsilon, args.epsilon, 1)

    agent = Agent(net, model2=net2, cuda=args.cuda, memory_size=50000, lr=args.lr)

    agent.train(game, batch_size=128, max_steps=2000, save_interval=5, observe=10000, epsilons=args.epsilon, start_epoch=args.epoch)
