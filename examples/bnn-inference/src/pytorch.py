import math
import sys
import torch.optim as optim
sys.path.append('../')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='PyTorch Bayesian Regression and forward pass')
parser.add_argument('n', type=int, default=1000)
parser.add_argument('t', type=int, default=1000)

args = parser.parse_args()

torch.manual_seed(0)  # for reproducibility
hasGPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if hasGPU else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if hasGPU else {}

GAUSSIAN_SCALER = 1. / np.sqrt(2.0 * np.pi)
def gaussian(x, mu, sigma):
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return torch.clamp(GAUSSIAN_SCALER / sigma * bell, 1e-10, 1.)  # clip to avoid numerical issues


def scale_mixture_prior(input, PI, SIGMA_1, SIGMA_2):
    prob1 = PI * gaussian(input, 0., SIGMA_1)
    prob2 = (1. - PI) * gaussian(input, 0., SIGMA_2)
    return torch.log(prob1 + prob2)


# Single Bayesian fully connected Layer with linear activation function
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, parent):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialise weights and bias
        if parent.GOOGLE_INIT: # These are used in the Tensorflow implementation.
            self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0., .05))  # or .01
            self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-5., .05))  # or -4
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0., .05))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-5., .05))
        else: # These are the ones we've been using so far.
            self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0., .1))
            self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-3., -3.))
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0., .1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-3., -3.))

        # Initialise prior and posterior
        self.lpw = 0.
        self.lqw = 0.

        self.PI = parent.PI
        self.SIGMA_1 = parent.SIGMA_1
        self.SIGMA_2 = parent.SIGMA_2
        self.hasScalarMixturePrior = parent.hasScalarMixturePrior

    # Forward propagation
    def forward(self, input, infer=False):
        if infer:
            return F.linear(input, self.weight_mu, self.bias_mu)

        # Obtain positive sigma from logsigma, as in paper
        weight_sigma = torch.log(1. + torch.exp(self.weight_rho))
        bias_sigma = torch.log(1. + torch.exp(self.bias_rho))

        # Sample weights and bias
        epsilon_weight = Variable(torch.Tensor(self.out_features, self.in_features).normal_(0., 1.)).to(DEVICE)
        epsilon_bias = Variable(torch.Tensor(self.out_features).normal_(0., 1.)).to(DEVICE)
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias

        # Compute posterior and prior probabilities
        if self.hasScalarMixturePrior:  # for Scalar mixture vs Gaussian analysis
            self.lpw = scale_mixture_prior(weight, self.PI, self.SIGMA_1, self.SIGMA_2).sum() + scale_mixture_prior(
                bias, self.PI, self.SIGMA_1, self.SIGMA_2).sum()
        else:
            self.lpw = torch.log(gaussian(weight, 0, self.SIGMA_1).sum() + gaussian(bias, 0, self.SIGMA_1).sum())

        self.lqw = torch.log(gaussian(weight, self.weight_mu, weight_sigma)).sum() + torch.log(
            gaussian(bias, self.bias_mu, bias_sigma)).sum()

        # Pass sampled weights and bias on to linear layer
        return F.linear(input, weight, bias)


class BayesianNetwork(nn.Module):
    def __init__(self, inputSize, CLASSES, layers, activations, SAMPLES, BATCH_SIZE, NUM_BATCHES, hasScalarMixturePrior,
                 PI, SIGMA_1, SIGMA_2, GOOGLE_INIT=False):
        super().__init__()
        self.inputSize = inputSize
        self.activations = activations
        self.CLASSES = CLASSES
        self.SAMPLES = SAMPLES
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_BATCHES = NUM_BATCHES
        self.DEPTH = 0  # captures depth of network
        self.GOOGLE_INIT = GOOGLE_INIT
        # to make sure that number of hidden layers is one less than number of activation function
        assert (activations.size - layers.size) == 1

        self.SIGMA_1 = SIGMA_1
        self.hasScalarMixturePrior = hasScalarMixturePrior
        if hasScalarMixturePrior == True:
            self.SIGMA_2 = SIGMA_2
            self.PI = PI

        self.layers = nn.ModuleList([])  # To combine consecutive layers
        if layers.size == 0:
            self.layers.append(BayesianLinear(inputSize, CLASSES, self))
            self.DEPTH += 1
        else:
            self.layers.append(BayesianLinear(inputSize, layers[0], self))
            self.DEPTH += 1
            for i in range(layers.size - 1):
                self.layers.append(BayesianLinear(layers[i], layers[i + 1], self))
                self.DEPTH += 1
            self.layers.append(BayesianLinear(layers[layers.size - 1], CLASSES, self))  # output layer
            self.DEPTH += 1

    # Forward propagation and assigning activation functions to linear layers
    def forward(self, x, infer=False):
        x = x.view(-1, self.inputSize)
        layerNumber = 0
        for i in range(self.activations.size):
            if self.activations[i] == 'relu':
                x = F.relu(self.layers[layerNumber](x, infer))
            elif self.activations[i] == 'softmax':
                x = F.log_softmax(self.layers[layerNumber](x, infer), dim=1)
            else:
                x = self.layers[layerNumber](x, infer)
            layerNumber += 1
        return x

    def get_lpw_lqw(self):
        lpw = 0.
        lpq = 0.

        for i in range(self.DEPTH):
            lpw += self.layers[i].lpw
            lpq += self.layers[i].lqw
        return lpw, lpq

    def BBB_loss(self, input, target, batch_idx = None):

        s_log_pw, s_log_qw, s_log_likelihood, sample_log_likelihood = 0., 0., 0., 0.
        for _ in range(self.SAMPLES):
            output = self.forward(input)
            sample_log_pw, sample_log_qw = self.get_lpw_lqw()
            if self.CLASSES > 1:
                sample_log_likelihood = -F.nll_loss(output, target, reduction='sum')
            else:
                sample_log_likelihood = -(.5 * (target - output) ** 2).sum()
            s_log_pw += sample_log_pw
            s_log_qw += sample_log_qw
            s_log_likelihood += sample_log_likelihood

        l_pw, l_qw, l_likelihood = s_log_pw / self.SAMPLES, s_log_qw / self.SAMPLES, s_log_likelihood / self.SAMPLES

        # KL weighting
        if batch_idx is None: # standard literature approach - Graves (2011)
            return (1. / (self.NUM_BATCHES)) * (l_qw - l_pw) - l_likelihood
        else: # alternative - Blundell (2015)
            return 2. ** ( self.NUM_BATCHES - batch_idx - 1. ) / ( 2. ** self.NUM_BATCHES - 1 ) * (l_qw - l_pw) - l_likelihood


TRAIN_EPOCHS = args.t
SAMPLES = 5
TEST_SAMPLES = 10
BATCH_SIZE = 200
NUM_BATCHES = 10
TEST_BATCH_SIZE = 50
CLASSES = 1
PI = 0.25
SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(DEVICE)
SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(DEVICE)

# Define training step for regression
def train(net, optimizer, data, target, NUM_BATCHES):
    #net.train()
    for i in range(NUM_BATCHES):
        net.zero_grad()
        x = data[i].reshape((-1, 1))
        y = target[i].reshape((-1,1))
        loss = net.BBB_loss(x, y)
        loss.backward()
        optimizer.step()

#Hyperparameter setting


print('Generating Data set.')

#Data Generation step
if torch.cuda.is_available():
    Var = lambda x, dtype=torch.cuda.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor
else:
    Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor

x = np.random.uniform(-0.1, 0.61, size=(NUM_BATCHES,BATCH_SIZE))
noise = np.random.normal(0, 0.02, size=(NUM_BATCHES,BATCH_SIZE)) #metric as mentioned in the paper
y = x + 0.3*np.sin(2*np.pi*(x+noise)) + 0.3*np.sin(4*np.pi*(x+noise)) + noise


def BBB_Regression(x,y,x_test, N):
    X = Var(x)
    Y = Var(y)
    X_test = Var(x_test)

    #Declare Network
    net = BayesianNetwork(inputSize = 1,\
                        CLASSES = CLASSES, \
                        layers=np.array([16,16,16]), \
                        activations = np.array(['relu','relu','relu','none']), \
                        SAMPLES = SAMPLES, \
                        BATCH_SIZE = BATCH_SIZE,\
                        NUM_BATCHES = NUM_BATCHES,\
                        hasScalarMixturePrior = True,\
                        PI = PI,\
                        SIGMA_1 = SIGMA_1,\
                        SIGMA_2 = SIGMA_2,\
                        GOOGLE_INIT= False).to(DEVICE)

    #Declare the optimizer
    optimizer = optim.SGD(net.parameters(),lr=1e-3,momentum=0.95)

    for epoch in tqdm(range(TRAIN_EPOCHS)):
        train(net, optimizer,data=X,target=Y,NUM_BATCHES=NUM_BATCHES)

    for name, param in net.named_parameters():
        if param.requires_grad:
            if "rho" in name:
               sigma = torch.log(1. + torch.exp(param))
               _name = name.replace("rho", "sigma")
               np.savetxt(f"model_params/{_name}.csv", sigma.data.cpu().numpy().flatten(), delimiter=",")
            else:
                np.savetxt(f"model_params/{name}.csv", param.data.cpu().numpy().flatten(), delimiter=",")

    #Testing
    y_test = []
    for i in tqdm(range(N)):
        y_test.append(net.forward(X_test, infer=False).cpu().data.numpy()[0])

    y_test = np.array(y_test)
    np.savetxt(f"samples/pytorch.csv", y_test.flatten(), delimiter=",")


if __name__ == '__main__':
    N = args.n
    x_test = np.repeat(-0.2, 1)
    BBB_Regression(x,y,x_test, N)