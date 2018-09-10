import torch
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
#from model import BNN_Gaussian
from bnn import BNN_Gaussian
from train import train

def get_mnist_dataset(mnist_path):
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])
    train_data = tv.datasets.MNIST(mnist_path, train=True, download=True, transform=trans)
    test_data = tv.datasets.MNIST(mnist_path, train=False, download=True, transform=trans)
    return train_data, test_data

def main():
    device = 'cuda'
    nb_samples = 10
    train_batch_size = 128
    test_batch_size = 1024
    lr = 0.01
    beta_1 = 0.5
    beta_2 = 0.999
    nb_workers = 32
    nb_epochs = 600
    train_data, test_data = get_mnist_dataset('./dataset/mnist')
    bnn = BNN_Gaussian(200, 0.001)
    bnn.to(device)
    train(bnn, train_data, test_data, nb_samples, nb_epochs, train_batch_size, test_batch_size, lr, beta_1, beta_2, nb_workers, device)

if __name__ == "__main__":
    main()
