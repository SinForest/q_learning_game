import torch
from torch.autograd import Variable
import torch.nn as nn
import math

from functools import reduce 
import operator

class NetworkSmallDuell(nn.Module):
    
    def __init__(self, inp_size, n_actions):
        super(NetworkSmallDuell, self).__init__() # designed for input=(30x30)
        self.conv = nn.Sequential(nn.Conv2d( 3, 16, kernel_size=5, stride=3, padding=1),
                                  nn.PReLU(), # output = (10x10)
                                  nn.Conv2d(16, 32, kernel_size=3, padding=1),
                                  nn.PReLU(),
                                  nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                  nn.PReLU())

        tmp = Variable(torch.Tensor(1, 3, inp_size, inp_size))
        out_size = reduce(operator.mul, self.conv(tmp).size())

        self.lin = nn.Sequential(nn.Linear(out_size, 128),
                                 nn.PReLU(),
                                 nn.Linear(128, 256),
                                 nn.PReLU())
        self.adv_val = nn.Linear(256, n_actions + 1) # [0] == val

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x   = self.conv(x)
        x   = self.lin(x.view(x.size(0), -1))
        x   = self.adv_val(x)
        adv = x[:, 1:]
        val = x[:, :1]
        val =  val.expand_as(adv)
        avg =  adv.mean(1).unsqueeze(1).expand_as(adv)
        return val + adv - avg

if __name__ == "__main__":
    net = NetworkSmallDuell(14, 4)
    tmp = Variable(torch.Tensor(64, 3, 14, 14))
    print(net(tmp))