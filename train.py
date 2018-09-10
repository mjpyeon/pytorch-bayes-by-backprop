import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader

def train(bnn, train_data, test_data, nb_samples,
          nb_epochs=20, train_batch_size=128,
          test_batch_size=1000,
          lr=0.001, beta_1=0.9, beta_2=0.999,
          nb_workers=4, device="cuda"):
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=nb_workers)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True, num_workers=nb_workers)
    loaders = {'train': train_loader, 'test':test_loader}
    optimizer = torch.optim.Adam(bnn.parameters(), lr/10, (beta_1, beta_2))
    for epoch in range(nb_epochs):
        for phase, loader in loaders.items():
            nb_correct = 0
            total_nll_loss = 0.
            total_kl_loss = 0.
            if phase == "train":
                nb_total = len(train_data)
            else:
                nb_total = len(test_data)
            for batch_idx, (x, y_true) in enumerate(loader):
                x = x.to(device)
                y_true = y_true.to(device)
                x = x.view(-1,784)
                if phase =='train':
                    optimizer.zero_grad()
                    x.requires_grad = True
                    y_true.requires_grad = False
                qw, pw, mle = bnn.forward_samples(x, y_true, nb_samples)
                kl_loss = (qw-pw)/len(loader)/len(y_true)
                total_kl_loss += kl_loss.item()
                nll_loss = -mle/len(y_true)
                total_nll_loss += nll_loss.item()
                loss = nll_loss + kl_loss
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                output = bnn.forward(x, test=True)
                y_pred = output.argmax(1)
                nb_correct += (y_pred == y_true).sum().item()
            print('{} Epoch: {}, NLL Loss: {:.3e}, KL loss:{:.3e}, Acc:{:.2f}%'.format(
                    phase, epoch+1, total_nll_loss/len(loader), total_kl_loss/len(loader), 100 * nb_correct / nb_total
                ))

#test
