import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
from copy import deepcopy

class BNN(nn.Module):
    def __init__(self, in_dim, hidden_units, out_dim,
                 scale_mixture_prior=None,
                 initial_mu=None, initial_rho=None):
        super(BNN, self).__init__()
        self.units = deepcopy(hidden_units)
        self.units.append(out_dim)
        self.units.insert(0, in_dim)
        self.scale_mixture_prior = scale_mixture_prior
        self.epsilon = lambda t: dist.Normal(torch.zeros_like(t), torch.ones_like(t)).sample()
        self.mu, self.rho = self.set_weights(initial_mu, initial_rho)

    def forward(self, x):
        out = x
        for l in range(len(self.units)-1):
            # sample w^l
            W = self.mu['w'][l] \
                + torch.log(1+torch.exp(self.rho['w'][l])) \
                * self.epsilon(self.mu['w'][l])
            b = self.mu['b'][l] \
                + torch.log(1+torch.exp(self.rho['b'][l])) \
                * self.epsilon(self.mu['b'][l])
            out = out @ W + b
            if l == len(self.units)-2:
                return out
            else:
                out = F.relu(out, inplace=True)

    def set_weights(self, initial_mu, initial_rho):
        if initial_mu is None:
            mu = self.random_weights()
        else:
            mu = initial_mu
        if initial_rho is None:
            rho = self.random_weights()
        else:
            rho = initial_rho
        return mu, rho

    def random_weights(self):
        layers = {'w':[], 'b':[]}
        for l in range(len(self.units)-1):
            current_w = torch.rand(self.units[l], self.units[l+1])
            current_w.requires_grad = True
            current_b = torch.rand(self.units[l+1])
            current_b.requires_grad = True
            layers['w'].append(current_w)
            layers['b'].append(current_b)
        return layers

# test
x = torch.ones(10)
x = x.view(1, 10)
net = BNN(10, [5,3], 2)
y = net(x)
print(y)
