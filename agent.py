import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
import os


class agent(object):

    def __init__(self, model, params, env):
        self.estimator = model
        self.env = env

        self.epsilon =params["epsilon"]
        self._gamma =params["gamma"]
        self._lambda =params["lambda"]
        self._alpha =params["alpha"]

    def select_action(self):
        raise NotImplemented

    def update(self, state, reward, next_state):
        raise NotImplemented

    def save(self, save_dir, name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.estimator.state_dict(), os.path.join(save_dir, name))


class RandomAgent(object):

    def select_action(self, game):
        A_s = game.get_valid_moves()
        a_id = np.random.randint(len(A_s))
        return A_s[a_id]


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
            print("Total number of parameters : {:.2f} M".format(self.get_number_of_params() / 1e6))
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

        total_params = 0
        for params in self.parameters():
            total_size = 1
            for size in params.size():
                total_size *= size
            total_params += total_size
        return total_params

    def name(self):
        return "MLP"

        
class smart(agent):
    def __init__(self, model, params, env, p=0):
        super().__init__(model, params, env)
        #
        self.optimizer  = torch.optim.SGD(self.estimator.parameters(), lr=self._alpha)
        self.p = p
        self.reset()
    
    def reset(self):
        #
        self.eligibilities  = dict()
        self.I = 1

        # initialise critic eligibilities ([re]set to zero(0))
        for i, group in enumerate(self.optimizer.param_groups):
            zs = dict()
            for p in group["params"]:
                zs[p] = torch.zeros_like(p.data)
            self.eligibilities[i] = zs
    
    def one_ply(self, env):
        # retrieve (query) list of successors
        # - indices (of implicit action/columns picked in the game):
        #                shape [n_successors, n_channels=1, height=1, width=1]
        # - successsors: shape [n_successors, n_channels=3, height, width]
        tuples = env.get_successors()
        successors, indices = zip(*tuples)
        indices = np.array(indices)
        successors = torch.Tensor(np.stack(successors))
        # evaluate successors
        # - values: shape [n_successors, n_channels=1, height=1, width=1]
        values = self.estimator(Variable(successors.view((-1, 2*6*7)))).cpu()
        values = values.data[:, 1]
        # choose an [implicit] action
        # based on an epsilon-greedy exploration scheme       
        idx = np.random.choice(np.where(values == values.max())[0])
        best_action = indices[idx]
        return best_action
        
    def two_ply(self, env):
        """Returns a list of tuples containing afterstates and actions that leads to those afterstates"""

        best_action = None
        best_value = 0
        valid_actions = env.game.get_valid_moves()
        level0 = env.game.grid
        over = env.game.over
        for action in valid_actions:
            level1 = env.game.make_move(1, action) # the state of the world won't be modified (here we only simulate)
            done = env.game.check_draw() or env.game.check_win(1) or env.game.check_win(2)
            if done:
                s = torch.Tensor(env.get_state(level1))
                v = self.estimator(Variable(s.view((-1, 2*6*7)))).cpu().data.numpy()[:, self.p]
            else:
                # retrieve (query) list of successors
                tuples = env.get_successors()
                successors, indices = zip(*tuples)
                indices = np.array(indices)
                successors = torch.Tensor(np.stack(successors))
                # evaluate successors
                values = self.estimator(Variable(successors.view((-1, 2*6*7)))).cpu().data
                # compute expected value
                v = values.numpy()[:, self.p].mean()
            # update best choice of action
            if v > best_value:
                best_value = v
                best_action = action
            # undo everything
            env.game.grid = level0
            env.game.over = over
        return best_action
    
    def three_ply(self, env):
        # retrieve (query) list of successors
        tuples = env.get_successors()
        successors, indices = zip(*tuples)
        indices = np.array(indices)
        successors = torch.Tensor(np.stack(successors))
        # evaluate successors
        values = self.estimator(Variable(successors.view((-1, 2*6*7)))).cpu().data
        #select top-4 best actions
        vs = values.numpy()[:, 1]
        n = min(4, vs.shape[0]-1)
        inds = np.argpartition(a=vs, kth=n)[-n:]
        l0_actions = indices[inds]
        
        best_action = None
        best_value = 0
        # l0_actions = env.game.get_valid_moves()
        level0 = env.game.grid        
        over = env.game.over
        for a in l0_actions:
            level1 = env.game.make_move(1, a) # the state of the world won't be modified (here we only simulate)
            done = env.game.check_draw() or env.game.check_win(1) or env.game.check_win(2)
            if done:
                s1 = torch.Tensor(env.get_state(level1))
                v = self.estimator(Variable(s1.view((-1, 2*6*7)))).cpu().data.numpy()[:, 0]
            else:
                # retrieve (query) list of successors
                l1_tuples = env.get_successors()
                l1_successors, l1_indices = zip(*l1_tuples)
                #
                l1_expectations = []
                for s in l1_successors:
                    l1_actions = env.game.get_valid_moves()  # opponent actions
                    level1 = env.game.grid
                    l2_expectations = []
                    for e in l1_actions:
                        level2 = env.game.make_move(1, e)  # opponent plays
                        done = env.game.check_draw() or env.game.check_win(1) or env.game.check_win(2)
                        if done:
                            s2 = torch.Tensor(env.get_state(level2))
                            v__ = self.estimator(Variable(s2.view((-1, 2*6*7)))).cpu().data.numpy()[:, 0]
                        else:
                            l2_tuples = env.get_successors()
                            l2_successors, l2_indices = zip(*l2_tuples)
                            l2_indices = np.array(l2_indices)
                            l2_successors = torch.Tensor(np.stack(l2_successors))
                            # evaluate successors
                            l2_values = self.estimator(Variable(l2_successors.view((-1, 2*6*7)))).cpu().data
                            # compute expected value
                            v__ = l2_values.numpy()[:, 0].mean()
                        l2_expectations.append(v__)
                        # undo e
                        env.game.grid = level1
                    #
                    l2_expectations = np.array(l2_expectations)
                    v_ = l2_expectations.mean()
                    l1_expectations.append(v_)
                #
                l1_expectations = np.array(l1_expectations)
                v = l1_expectations.mean()
            # update best choice of action
            if v > best_value:
                best_value = v
                best_action = a
            # undo a
            env.game.grid = level0
            env.game.over = over
        return best_action
    
    def select_action(self):
        # retrieve (query) list of possible moves
        action = None
        s = np.random.uniform()
        if s < self.epsilon:
            moves = self.env.game.get_valid_moves()
            action = np.random.choice(moves)
        else:
            action = self.two_ply(self.env)
        return action
    def update(self, state, reward, next_state):
        #
        # reward = reward[:, :2]
        error = Variable(torch.Tensor(reward)) + ( self._gamma * self.estimator(Variable(torch.Tensor(next_state).view((1, 2*6*7))))) - self.estimator(Variable(torch.Tensor(state).view((1, 2*6*7)))) if not self.env.game.over else Variable(torch.Tensor(reward)) - self.estimator(Variable(torch.Tensor(state).view((1, 2*6*7))))  # estimator(next_state) - estimator(state) if not done else reward - critic(state)
        #
        _delta = error.cpu().data[0, self.p]
        #
        v = self.estimator(Variable(torch.Tensor(state).view((1, 2*6*7))))#[0, self.p]
        #v.backward()
        z = torch.zeros(1, 3)
        z[0, self.p] = 1
        z[0, 2] = -1
        loss = torch.sum(Variable(z)*v)
        loss.backward
        #
        for i, group in enumerate(self.optimizer.param_groups):

            for p in group["params"]:
                if p.grad is None:
                    continue
                # retrieve current eligibility
                z = self.eligibilities[i][p]
                # retrieve current gradient
                grad = p.grad.data
                # update eligibility
                #
                z.mul_(self._gamma * self._lambda).add_(self.I, grad)
                # update parameters
                p.data.add_(self._alpha * _delta * z)
                # reset gradients
                p.grad.detach_()
                p.grad.zero_()
        #
        self.I *= self._gamma

class greed(smart):
    def __init__(self, model, params, env, p=0):
        super().__init__(model, params, env, p)
    def select_action(self):
        # retrieve (query) list of possible moves
        action = self.two_ply(self.env)
        return action

class random(agent):
    def select_action(self):
        moves = self.env.game.get_valid_moves()
        action = np.random.choice(moves)
        return action