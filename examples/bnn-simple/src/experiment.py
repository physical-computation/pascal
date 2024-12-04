import time
import torch
import argparse
import math

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# torch.set_num_threads(1)
torch.set_printoptions(precision=12, sci_mode = False)
np.set_printoptions(precision=18, linewidth=100000)

parser = argparse.ArgumentParser(description='PyTorch Bayesian Neural Network')
parser.add_argument('learning_rate', type=float)
parser.add_argument('n', type=int)
parser.add_argument("n_test_data_points", type=int)
parser.add_argument("n_repetitions", type=int)
parser.add_argument("n_data_points", type=int)
parser.add_argument("n_nodes", type=int)
parser.add_argument("n_layers", type=int)
parser.add_argument("print_frequency", type=int)

args = parser.parse_args()

N_NODES = args.n_nodes
N_LAYERS = args.n_layers
N = args.n
SAMPLES = 1
N_TEST_DATA_POINTS = args.n_test_data_points
N_DATA_POINTS = args.n_data_points
NUM_BATCHES = 1
CLASSES = 1
PI = 0.25
N_REPETITIONS = args.n_repetitions
PRINT_FREQUENCY = args.print_frequency
LEARNING_RATE = args.learning_rate

activations = []
layers = []
for i in range(N_LAYERS):
    layers.append(N_NODES)
    activations.append("tanh")

activations.append("sin")
activations = np.array(activations)
layers = np.array(layers)


hasGPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if hasGPU else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if hasGPU else {}

SIGMA_1 = torch.DoubleTensor([math.exp(-0)]).to(DEVICE)
NEG_ONE = torch.DoubleTensor([-1.0]).to(DEVICE)
SIGMA_2 = torch.DoubleTensor([math.exp(-6)]).to(DEVICE)

def set_grad(var):
	def hook(grad):
		var.grad = grad
		print(grad.numpy())
	return hook

def load_epsilons(file_name: str):
    n_epsilons = (N + ((N_REPETITIONS*2))) * (N_LAYERS + 1)

    epsilons = []
    with open(file_name, "r+") as f:
        for i in range(n_epsilons):
            epsilons.append(float(f.readline()))

    return np.array(epsilons)

EPSILONS = load_epsilons("data/epsilons.dat")
EPSILONS_INDEX = 0
EPOCH = 0

GAUSSIAN_SCALER = 1. / np.sqrt(2.0 * np.pi)

def gaussian(x, mu, sigma, _print=True):
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    out = torch.clamp(GAUSSIAN_SCALER / sigma * bell, 1e-10, 1.)

    return  out

def log_gaussian(w, mu, sigma):
    constant_part = 0.5 * math.log(2. * np.pi)
    diff = w - mu
    # diff.register_hook(set_grad(diff))

    std_inv = torch.reciprocal(sigma)
    # std_inv.register_hook(set_grad(std_inv))

    mean_part_root = diff * std_inv
    # mean_part_root.register_hook(set_grad(mean_part_root))

    mean_part_sq = torch.square(mean_part_root)
    # mean_part_sq.register_hook(set_grad(mean_part_sq))

    mean_part_square = 0.5 * mean_part_sq
    # mean_part_square.register_hook(set_grad(mean_part_square))

    log_std = torch.log(sigma)
    # log_std.register_hook(set_grad(log_std))

    log_gaussian_unscaled = log_std + (mean_part_square + constant_part)
    # log_gaussian_unscaled.register_hook(set_grad(log_gaussian_unscaled))

    # print(log_gaussian_unscaled)
    log_gaussian_v = NEG_ONE * log_gaussian_unscaled
    # log_gaussian_v.register_hook(set_grad(log_gaussian_v))
    # print(log_gaussian_v)
    out = torch.clamp(log_gaussian_v, -23.025850929940457, 0)
    # out.register_hook(set_grad(out))
    return out

def scale_mixture_prior(input, PI, SIGMA_1, SIGMA_2):
    prob1 = PI * gaussian(input, 0., SIGMA_1)
    prob2 = (1. - PI) * gaussian(input, 0., SIGMA_2)
    return torch.log(prob1 + prob2)


def load_weights(file_name: str, shape: tuple):
    n, m = shape
    shape[0] = m
    shape[1] = n

    weights = []
    size = 1
    for s in shape:
        size *= s

    with open(file_name, "r+") as f:
        for i in range(size):
            weights.append(float(f.readline()))

    return np.array(weights).reshape(shape).T


# Single Bayesian fully connected Layer with linear activation function
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, parent, u_index):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        mu = load_weights(f"data/w_mean_{u_index}.dat", [out_features, in_features])
        rho = load_weights(f"data/w_rho_{u_index}.dat", [out_features, in_features])

        self.u_index = u_index

        self.weight_mu = nn.Parameter(torch.tensor(mu, dtype=torch.double, requires_grad=True))
        self.weight_rho = nn.Parameter(torch.tensor(rho, dtype=torch.double, requires_grad=True))

        # Initialise prior and posterior
        self.lpw = 0.
        self.lqw = 0.

        self.PI = parent.PI
        self.SIGMA_1 = parent.SIGMA_1
        self.SIGMA_2 = parent.SIGMA_2
        self.hasScalarMixturePrior = parent.hasScalarMixturePrior

    # Forward propagation
    def forward(self, input, infer=False):
        global EPSILONS_INDEX
        global EPOCH

        if infer:
            return F.linear(input, self.weight_mu, self.bias_mu)

        # Obtain positive sigma from logsigma, as in paper
        rho = self.weight_rho.clone()
        # rho.register_hook(set_grad(rho))

        weight_sigma = torch.log(1. + torch.exp(rho))

        # Sample weights and bias
        epsilon_weight = Variable(torch.tensor([EPSILONS[EPSILONS_INDEX]], dtype=torch.double).reshape([1, 1])).to(DEVICE)
        EPSILONS_INDEX += 1

        # print(epsilon_weight)
        sig = weight_sigma.clone()
        mu = self.weight_mu.clone()
        # if EPOCH == 2 and self.u_index == 0:
        #     # sig.register_hook(set_grad(sig))
        #     mu.register_hook(set_grad(mu))

        w_mul = sig * epsilon_weight
        # w_mul.register_hook(set_grad(w_mul))

        weight = mu + w_mul
        # if EPOCH == 2 and self.u_index == 0:
        #     weight.register_hook(set_grad(weight))

        # print(weight)
        # weight.register_hook(set_grad(weight))
        # Compute posterior and prior probabilities
        w_lg = weight.clone()
        # if EPOCH == 2 and self.u_index == 0:
        #     w_lg.register_hook(set_grad(w_lg))
        _lg = torch.log(gaussian(w_lg, 0, self.SIGMA_1))
        # _lg.register_hook(set_grad(_lg))
        self.lpw = _lg.sum()

        w_sp = weight.clone()
        mu_sp = self.weight_mu.clone()
        sig_sp = weight_sigma.clone()

        # if EPOCH == 2:
            # w_sp.register_hook(set_grad(w_sp))
            # mu_sp.register_hook(set_grad(mu_sp))
            # sig_sp.register_hook(set_grad(sig_sp))
        # w_sp.register_hook(set_grad(w_sp))
        # mu_sp.register_hook(set_grad(mu_sp))
        # sig_sp.register_hook(set_grad(sig_sp))

        _ll = log_gaussian(w_sp, mu_sp, sig_sp)
        # _ll = log_gaussian(w, self.weight_mu, weight_sigma)
        self.lqw = _ll.sum()
        # _ll.register_hook(set_grad(_ll))

        # Pass sampled weights and bias on to linear layer
        # print(input)
        w = weight.clone()
        # w.register_hook(set_grad(w))
        # if EPOCH == 0 and self.u_index == 0:
        #     w.register_hook(set_grad(w))
        return input @ w.T
        # return F.linear(input, weight, None)


class BayesianNetwork(nn.Module):
    def __init__(self, inputSize, CLASSES, layers, activations, SAMPLES, N_DATA_POINTS, NUM_BATCHES, hasScalarMixturePrior,
                 PI, SIGMA_1, SIGMA_2, GOOGLE_INIT=False):
        super().__init__()
        self.inputSize = inputSize
        self.activations = activations
        self.CLASSES = CLASSES
        self.SAMPLES = SAMPLES
        self.N_DATA_POINTS = N_DATA_POINTS
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
        else:
            self.SIGMA_2 = None
            self.PI = None

        self.layers = nn.ModuleList([])  # To combine consecutive layers
        if layers.size == 0:
            self.layers.append(BayesianLinear(inputSize, CLASSES, self))
            self.DEPTH += 1
        else:
            self.layers.append(BayesianLinear(inputSize, layers[0], self, self.DEPTH))
            self.DEPTH += 1
            for i in range(layers.size - 1):
                self.layers.append(BayesianLinear(layers[i], layers[i + 1], self, self.DEPTH))
                self.DEPTH += 1
            self.layers.append(BayesianLinear(layers[layers.size - 1], CLASSES, self, self.DEPTH))  # output layer
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
            elif self.activations[i] == "tanh":
                # print(x)
                linear = self.layers[layerNumber](x, infer)
                # print(linear)
                out = torch.tanh(linear)
                # print(out)
                # linear.register_hook(set_grad(linear))
                x = out
            elif self.activations[i] == "sin":
                linear = self.layers[layerNumber](x, infer)
                out = torch.sin(linear)
                x = out
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
            # print(output)
            # output.register_hook(set_grad(output))
            sample_log_pw, sample_log_qw = self.get_lpw_lqw()
            if self.CLASSES > 1:
                sample_log_likelihood = -F.nll_loss(output, target, reduction='sum')
            else:
                diff = target - output
                # diff.register_hook(set_grad(diff))
                mean_part_squared = torch.square(diff)
                # mean_part_squared.register_hook(set_grad(mean_part_squared))

                _ll = -(.5 * mean_part_squared)
                # _ll = -(.5 * (target - output) ** 2)
                # print(_ll)
                # _ll.register_hook(set_grad(_ll))
                sample_log_likelihood = _ll.sum()
            s_log_pw += sample_log_pw
            s_log_qw += sample_log_qw
            s_log_likelihood += sample_log_likelihood


        l_pw, l_qw, l_likelihood = s_log_pw / self.SAMPLES, s_log_qw / self.SAMPLES, s_log_likelihood / self.SAMPLES


        # print(l_pw)
        # print(l_qw)
        # l_pw.register_hook(set_grad(l_pw))
        # l_qw.register_hook(set_grad(l_qw))
        # print(l_likelihood)
        # l_likelihood.register_hook(set_grad(l_likelihood))
        # s_log_qw.register_hook(set_grad(s_log_qw))

        # KL weighting
        if batch_idx is None: # standard literature approach - Graves (2011)
            return (1. / (self.NUM_BATCHES)) * (l_qw - l_pw) - l_likelihood
        else: # alternative - Blundell (2015)
            return 2. ** ( self.NUM_BATCHES - batch_idx - 1. ) / ( 2. ** self.NUM_BATCHES - 1 ) * (l_qw - l_pw) - l_likelihood


# Define training step for regression
def train(net, optimizer, data, target, NUM_BATCHES, epoch):
    #net.train()
    global EPOCH
    for i in range(NUM_BATCHES):
        start = time.time()
        net.zero_grad()
        x = data[i].reshape((-1, 1))
        y = target[i].reshape((-1,1))

        loss = net.BBB_loss(x, y)
        # loss.register_hook(set_grad(loss))
        loss.backward()

        # if EPOCH == 2:
        #     print(net.layers[0].weight_mu.grad.data.numpy().T)
        # print(net.layers[0].weight_rho.grad.T.data.numpy())

        optimizer.step()
        if epoch % PRINT_FREQUENCY == 0:
            print(f"iteration={epoch+1}\t\tloss={loss.data.item():15.8f}\t\ttime={(time.time() - start)*1000:.5f}ms")

#Hyperparameter setting

def load_data(file_name: str, num_data_points: int, x_dim: int, y_dim: int):
	x_data = []
	y_data = []

	with open(file_name, "r+") as f:
		for i in range(num_data_points):
			for j in range(x_dim):
				x_data.append(float(f.readline()))

			for j in range(y_dim):
				y_data.append(float(f.readline()))

	x = np.array(x_data).reshape(num_data_points, x_dim)
	y = np.array(y_data).reshape(num_data_points, y_dim)

	return x, y

#Data Generation step
if torch.cuda.is_available():
    Var = lambda x, dtype=torch.cuda.DoubleTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor
else:
    Var = lambda x, dtype=torch.DoubleTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor

def BBB_Regression(x,y,x_test,y_test):
    X = Var(x)
    Y = Var(y)
    X_test = Var(x_test)

    #Declare Network
    net = BayesianNetwork(inputSize = 1,\
                        CLASSES = CLASSES, \
                        layers=layers, \
                        activations = activations, \
                        SAMPLES = SAMPLES, \
                        N_DATA_POINTS = N_DATA_POINTS,\
                        NUM_BATCHES = NUM_BATCHES,\
                        hasScalarMixturePrior = False,\
                        PI = PI,\
                        SIGMA_1 = SIGMA_1,\
                        SIGMA_2 = SIGMA_2,\
                        GOOGLE_INIT= False).to(DEVICE)

    #Declare the optimizer
    optimizer = optim.SGD(net.parameters(),lr=LEARNING_RATE,momentum=0.0)

    global EPOCH
    for epoch in range(N):
        train(net, optimizer,data=X,target=Y,NUM_BATCHES=NUM_BATCHES, epoch=epoch)
        EPOCH += 1

    # Testing
    outputs = torch.zeros(N_REPETITIONS, N_TEST_DATA_POINTS, CLASSES).to(DEVICE)
    for i in range(N_REPETITIONS):
        outputs[i] = net.forward(X_test)

    np.savetxt("results/bnn-eval-pytorch.dat", outputs.detach().numpy().flatten())


if __name__ == '__main__':
    x, y = load_data("data/train_data.dat", N_DATA_POINTS, 1, 1)
    x = x.reshape((1, N_DATA_POINTS))
    y = y.reshape((1, N_DATA_POINTS))

    x_test, y_test = load_data("data/test_data.dat", N_TEST_DATA_POINTS, 1, 1)
    x_test = x_test.squeeze(1)
    y_test = y_test.squeeze(1)

    BBB_Regression(x,y,x_test,y_test)