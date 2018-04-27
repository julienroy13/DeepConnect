import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
import os
import pdb


class agent(object):
    def __init__(self, model, params, env, p=1):
        self.estimator = model
        self.env = env
        self.p = p

        self.epsilon =params["epsilon"]
        self._gamma =params["gamma"]
        self._lambda =params["lambda"]
        self._alpha =params["alpha"]

    def select_action(self):
        raise NotImplementedError

    def update(self, state, reward, next_state):
        raise NotImplementedError

    def save(self, save_dir, name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.estimator.state_dict(), os.path.join(save_dir, name))

    def load(self, path):
        self.estimator.load_state_dict(torch.load(path))


class MLP(nn.Module):
    def __init__(self, inp_size, h_sizes, out_size, act_fn, init_type, verbose=False):

        super(MLP, self).__init__()

        # Hidden layers
        self.hidden = nn.ModuleList([nn.Linear(inp_size, h_sizes[0])])
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        # Activation function
        if act_fn == "relu":
            self.act_fn = F.relu

        elif act_fn == "sigmoid":
            self.act_fn = F.sigmoid

        elif act_fn == "tanh":
            self.act_fn = F.tanh

        else:
            raise ValueError('Specified activation function "{}" is not recognized.'.format(act_fn))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

        # Initializes the parameters
        self.parameters_init(init_type)

        if verbose:
            print('\nModel Info ------------')
            print(self)
            print("Total number of parameters : {:.2f} k".format(self.get_number_of_params() / 1e3))
            print('---------------------- \n')

    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            a = layer(x)
            x = self.act_fn(a)

        # Sigmoid layer to output a probability of winning
        output = F.softmax(self.out(x), dim=1)

        return output

    def parameters_init(self, init_type):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant(module.bias, 0)

                if init_type == "glorot":
                    nn.init.xavier_uniform(module.weight, gain=1)

                elif init_type == "default":
                    stdv = 1. / math.sqrt(module.weight.size(1))
                    nn.init.uniform(module.weight, -stdv, stdv)

        for p in self.parameters():
            p.requires_grad = True

    def get_number_of_params(self):

        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        return num_params

    def name(self):
        return "MLP"

        
class smart(agent):
    def __init__(self, model, params, env, p=1, beta=2.7, eps=1e-8, tcl=False, break_ties='random'):
        super().__init__(model, params, env, p)
        #
        self.optimizer = torch.optim.SGD(self.estimator.parameters(), lr=self._alpha)
        self._TCL   = tcl
        self._beta  = beta
        self._eps   = eps
        self.break_ties = break_ties
        self.reset()
    
    def reset(self):
        self.eligibilities    = dict()
        self.net_changes      = dict()  # the algebraic sum of all weight changes (displacement in the parameter space) up to now.
        self.absolute_changes = dict()  # the absolute sum (sum of absolute values) of all weight changes up to now.
        self.I = 1

        for i, group in enumerate(self.optimizer.param_groups):
            
            # initialise eligibilities ([re]set to zero(0))
            es = dict()
            for j, p in enumerate(group["params"]):
                es[j] = torch.zeros_like(p.data)
            self.eligibilities[i] = es
        
            # initialise net changes ([re]set to zero(0))
            ns = dict()
            for j, p in enumerate(group["params"]):
                ns[j] = torch.zeros_like(p.data)
            self.net_changes[i] = ns

            # initialise absolute changes ([re]set to zero(0))
            zs = dict()
            for j, p in enumerate(group["params"]):
                zs[j] = torch.zeros_like(p.data)
            self.absolute_changes[i] = zs
    
    def _one_ply(self, env):        
        # Queries a list of all legal moves and successors
        tuples = env.get_successors(self.p)
        successors, indices = zip(*tuples)
        indices = np.array(indices)
        successors = Variable(torch.Tensor(np.stack(successors)).view((-1, self.env.d * self.env.game.n_rows * self.env.game.n_columns + 2)))
        
        # Evaluates all legal afterstates (successors)
        values = self.estimator(successors)
        values = values.data[:, self.p].numpy()
        
        # Chooses the action that leads to highest valued afterstate (successor)   
        if self.break_ties == 'argmax':
            best_action = indices[np.argmax(values)]
        elif self.break_ties == 'random':
            idx = np.random.choice(np.where(values == values.max())[0])
            best_action = indices[idx]

        return best_action
    
    def select_action(self):
        action = None
        
        # Takes a random action with probability epsilon
        if np.random.uniform() < self.epsilon:
            moves = self.env.game.get_valid_moves()
            action = np.random.choice(moves)
        
        # Selects the action that leads to maximum valued afterstate
        else:
            action = self._one_ply(self.env)
        
        return action
    
    def update(self, state, next_state, reward=None, terminal=False):
        
        # Transforms states ndarrays into Torch Vectors
        state = Variable(torch.Tensor(state).view((1, self.env.d * self.env.game.n_rows * self.env.game.n_columns + 2)))
        next_state = Variable(torch.Tensor(next_state).view((1, self.env.d * self.env.game.n_rows * self.env.game.n_columns + 2)))

        # Computes the temporal difference (TD error)
        if not terminal:
            error = (self._gamma * self.estimator(next_state)) - self.estimator(state)
        else:
            reward = Variable(torch.Tensor(reward))
            error = reward - self.estimator(state)

        # Forward pass
        values = self.estimator(state)

        for outcome in range(3):
            _delta = error.data[0, outcome]
        
            # Backpropagates the gradient of the estimation that the agent will win
            v = values[0, outcome]
            retain = True if outcome < 2 else False
            v.backward(retain_graph=retain)
            
            # Updates the parameters
            for i, group in enumerate(self.optimizer.param_groups):

                for j, p in enumerate(group["params"]):
                    if p.grad is None:
                        raise Exception("No gradient for these parameters")
                    # retrieve current eligibility
                    z = self.eligibilities[i][j]
                    # retrieve current gradient
                    grad = p.grad.data
                    # update eligibility
                    z.mul_(self._gamma * self._lambda).add_(self.I, grad)
                    # update parameters
                    if self._TCL != False:
                        # retrive net change and absolute change
                        n = self.net_changes[i][j]
                        a = self.absolute_changes[i][j]
                        # compute learning rate decay
                        if self._TCL == 'v1':
                            _x = torch.abs(n)/(a + self._eps)
                        elif self._TCL == 'v2':
                            _x = torch.abs(n)/a if (a>0).all() else torch.ones_like(p.data)
                        _lr_decay = torch.exp(self._beta*(_x - 1.))
                        # apply update
                        u =  _delta * z
                        p.data.add_(self._alpha * _lr_decay * u)
                        # update net change and absolute change
                        a.add_(torch.abs(u))
                        n.add_(u)
                    else:
                        # regular update
                        p.data.add_(self._alpha * _delta * z)
                    # reset gradients
                    p.grad.zero_()

        # Detach the params from the graph
        for i, group in enumerate(self.optimizer.param_groups):
            for j, p in enumerate(group["params"]):
                # reset gradients
                p.grad.detach_()
        
        self.I *= self._gamma

        return error.data.numpy()


class random(agent):
    def select_action(self):
        # Randomly picks an action among the acceptable ones
        moves = self.env.game.get_valid_moves()
        action = np.random.choice(moves)
        return action