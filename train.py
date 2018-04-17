import numpy as np
from agent import MLP, smart, greed, random
from env import Connect4Environment

from matplotlib import pyplot as plt
from tqdm import tqdm

# training parameters
params = {"epsilon": 0.01, "gamma": 1., "lambda": 0.9, "alpha": 1e-3}

# environment
env = Connect4Environment()

# some utils
def train(state, player, token):
    # print(env.game.over)
    action = player.select_action()
    # take chosen [implicit] action
    # eq. to saying to the environment which successor is picked/prefered/...etc
    # note: returns the chosen successor and the associated reward
    # print(env.game.over)
    next_state, reward = env.play(token, action)
    # print(env.game.over)
    # update state-value model [of the agent]
    # ... TD-Gammon update with eligibility-traces (and possibly discounted rewards)
    player.update(state, reward, next_state)

    #
    return next_state, reward

def play(state, player, token):
    action = player.select_action()
    # take chosen [implicit] action
    # eq. to saying to the environment which successor is picked/prefered/...etc
    # note: returns the chosen successor and the associated reward
    next_state, reward = env.play(token, action)

    #
    return next_state, reward

# exampleself-training
rs = []

for m in range(1):
    estimator = MLP(2*6*7, [160], 3, "sigmoid", "glorot", verbose=True)
    smarty = smart(model=estimator, params=params, env=env, p=0)
    opponent = smart(model=estimator, params=params, env=env, p=1)
    r = []

    for i in tqdm(range(1000)):
        # initialise (reset) the environment
        # note: returns the initial state of the environment
        # print("episode-{} start ...".format(i))
        env.reset()
        smarty.reset()
        state = np.zeros((1, 2*6*7))
        index = 1
        while not env.game.over:
            if index == 1:
                p = smarty
                # state, reward = train(state, p, index)
            elif index == 2:
                p = opponent
                # state, reward = play(state, p, index)
            # print("P{} is playing ...".format(index))
            state, reward = train(state, p, index)
            # if env.game.over:
            #     if (reward == 0).all():
            #         print("\nreward: {}".format(reward))
            #         print("state:")
            #         env.game.print_grid()
            #     else:
            #         print("episode-{} done!".format(i))
            r.append(reward)
            index = (index%2) + 1
    rs.append(r)
    smarty.save('models', 'smarty_{}.pkl'.format(i+1))

# small visualisation
# plt.clf()
for r in rs[:]:
    z = r[:]
    z = np.stack(z)
    x = z.squeeze(1).cumsum(axis=0)
    for i in range(x.shape[1]-1):
        plt.plot(x[:,i], label="P{} wins".format(i))
    plt.plot(x[:,-1], label="Draw")
plt.legend("best")
plt.show()

# small test against ranom player
greedy  = greed(model=estimator, params=params, env=env, p=0)
rand = random(model=None, params=params, env=env)
_r = 0
for i in tqdm(range(500)):
    # initialise (reset) the environment
    # note: returns the initial state of the environment
    env.reset()
    state = np.zeros((1, 2*6*7))
    index = 1
    while not env.game.over:
        if index == 1:
            p = greedy
        elif index == 2:
            p = rand
        state, reward = play(state, p, index)
        _r = _r + reward
        index = (index%2) + 1
print(_r)