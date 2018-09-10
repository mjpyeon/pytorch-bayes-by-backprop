import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np

def log_gaussian_prob(x, mu, sigma, log_sigma=False):
    if not log_sigma:
        element_wise_log_prob = -0.5*torch.Tensor([np.log(2*np.pi)]).to(mu.device) - torch.log(sigma) - 0.5*(x-mu)**2 / sigma**2
    else:
        element_wise_log_prob = -0.5*torch.Tensor([np.log(2*np.pi)]).to(mu.device) - F.softplus(sigma) - 0.5*(x-mu)**2 / F.softplus(sigma)**2
    return element_wise_log_prob.sum()

class GaussianLinear(nn.Module):
    def __init__(self, in_dim, out_dim, stddev_prior, bias=True):
        super(GaussianLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stddev_prior = stddev_prior
        self.w_mu = nn.Parameter(torch.Tensor(in_dim, out_dim).normal_(0, stddev_prior))
        self.w_rho = nn.Parameter(torch.Tensor(in_dim, out_dim).normal_(0, stddev_prior))
        self.b_mu = nn.Parameter(torch.Tensor(out_dim).normal_(0, stddev_prior)) if bias else None
        self.b_rho = nn.Parameter(torch.Tensor(out_dim).normal_(0, stddev_prior)) if bias else None
        self.bias = bias
        self.q_w = 0.
        self.p_w = 0.

    def forward(self, x, test=False):
        if test:
            w = self.w_mu
            b = self.b_mu if self.bias else None
        else:
            device = self.w_mu.device
            w_stddev = F.softplus(self.w_rho)
            b_stddev = F.softplus(self.b_rho) if self.bias else None
            w = self.w_mu + w_stddev * torch.Tensor(self.in_dim, self.out_dim).to(device).normal_(0,self.stddev_prior)
            b = self.b_mu + b_stddev * torch.Tensor(self.out_dim).to(device).normal_(0,self.stddev_prior) if self.bias else None
            self.q_w = log_gaussian_prob(w, self.w_mu, self.w_rho, log_sigma=True)
            self.p_w = log_gaussian_prob(w, torch.zeros_like(self.w_mu, device=device), self.stddev_prior*torch.ones_like(w_stddev, device=device))
            if self.bias:
                self.q_w += log_gaussian_prob(b, self.b_mu, self.b_rho, log_sigma=True)
                self.p_w += log_gaussian_prob(b, torch.zeros_like(self.b_mu, device=device), self.stddev_prior*torch.ones_like(b_stddev, device=device))
        output = x@w+b
        return output

    def get_pw(self):
        return self.p_w

    def get_qw(self):
        return self.q_w

class BNN_Gaussian(nn.Module):
    def __init__(self, hidden_size, stddev_prior, bias=True):
        super(BNN_Gaussian, self).__init__()
        self.stddev_prior = stddev_prior
        self.fc1 = GaussianLinear(784, hidden_size, stddev_prior, bias=bias)
        self.fc2 = GaussianLinear(hidden_size, hidden_size, stddev_prior, bias=bias)
        self.fc3 = GaussianLinear(hidden_size, 10, stddev_prior, bias=bias)

    def forward(self, x, test=False):
        x = F.relu(self.fc1(x, test))
        x = F.relu(self.fc2(x, test))
        x = self.fc3(x, test)
        return F.softmax(x, dim=1)

    def forward_samples(self, x, y, nb_samples=3):
        total_qw, total_pw, total_log_likelihood = 0., 0., 0.
        for _ in range(nb_samples):
            output = self.forward(x)
            total_qw += self.get_qw()
            total_pw += self.get_pw()
            y = y.view(len(y), -1)
            y_onehot = torch.Tensor(len(y), 10).to(x.device)
            y_onehot.zero_()
            y_onehot.scatter_(1, y, 1)
            total_log_likelihood += log_gaussian_prob(y_onehot, output, self.stddev_prior*torch.ones_like(y_onehot, device=y_onehot.device))
        return total_qw / nb_samples, total_pw / nb_samples, total_log_likelihood / nb_samples

    def get_pw(self):
        return self.fc1.p_w + self.fc2.p_w + self.fc3.p_w

    def get_qw(self):
        return self.fc1.q_w + self.fc2.q_w + self.fc3.q_w








