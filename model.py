import torch
from torch.autograd import Variable
import torch.nn as nn
import math

from functools import reduce 
import operator

class Network(nn.Module):
    
    def __init__(self, inp_size, n_actions):
        super(Network, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=2),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2),

                                  nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2),

                                  nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))

        tmp = Variable(torch.Tensor(1, 3, inp_size, inp_size))
        out_size = self.conv(tmp).size(-1)

        self.lin = nn.Sequential(nn.Dropout2d(),
                                 nn.Conv2d(256, 1024, kernel_size=out_size),
                                 nn.ReLU(),
                                 nn.Dropout2d(),
                                 nn.Conv2d(1024, 1024, kernel_size=1),
                                 nn.ReLU(),
                                 nn.Conv2d(1024, n_actions, kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.lin(self.conv(x))
        return x.view(x.size(0), -1)

class NetworkSmall(nn.Module):
    
    def __init__(self, inp_size, n_actions):
        super(NetworkSmall, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 16, kernel_size=7, padding=3),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.Conv2d(16, 32, kernel_size=5, padding=2),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU())

        tmp = Variable(torch.Tensor(1, 3, inp_size, inp_size))
        out_size = reduce(operator.mul, self.conv(tmp).size())

        self.lin = nn.Sequential(nn.Linear(out_size, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, n_actions))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return self.lin(x.view(x.size(0), -1))

if __name__ == "__main__":

    net = NetworkSmall(32, 4)
    tmp = Variable(torch.rand(8, 3, 32, 32))
    print("cpu:", net(tmp))

    net = net.cuda()
    tmp = tmp.cuda()
    out = net(tmp)
    print("gpu:", out)

    opti = torch.optim.SGD(net.parameters(), lr=0.00001)
    targ = Variable(torch.rand(out.size())).cuda()
    l_fn = nn.SmoothL1Loss()
    loss = l_fn(out, targ)
    print("loss:", loss)

    opti.zero_grad()
    loss.backward()
    opti.step()
    out = net(tmp)
    print("after train:", out)

    import numpy as np
    n = np.load("./debug_screens.npy")
    n = torch.Tensor(n.transpose(0,3,1,2))
    n = Variable(n / 128 - 1).cuda()
    print("screens:", net(n))
