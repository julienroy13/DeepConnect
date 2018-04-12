import torch.nn as nn
import torch.nn.functional as F
import math





class agent(object):

    def __init__(self, input_size):
        self.value_network = MLP(input_size, [512, 512], 1, "relu", "glorot", verbose=True)

    def select_action(self, afterstates):
        pass

    def update_parameters(self):
        pass



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
        output = F.sigmoid(self.out(x))

        return output

    def parameters_init(self, init_type):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant(module.bias, 0)

                elif init_type == "glorot":
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