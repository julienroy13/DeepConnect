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
def train(state, player):
    # print(env.game.over)
    action = player.select_action()
    # take chosen [implicit] action
    # eq. to saying to the environment which successor is picked/prefered/...etc
    # note: returns the chosen successor and the associated reward
    # print(env.game.over)
    token = player.p
    next_state, reward = env.play(token, action)
    # print(env.game.over)
    # update state-value model [of the agent]
    # ... TD-Gammon update with eligibility-traces (and possibly discounted rewards)
    player.update(state, reward, next_state)

    #
    return next_state, reward

def play(state, player):
    action = player.select_action()
    # take chosen [implicit] action
    # eq. to saying to the environment which successor is picked/prefered/...etc
    # note: returns the chosen successor and the associated reward
    token = player.p
    next_state, reward = env.play(token, action)

    #
    return next_state, reward

# example of self-training
rs = []
n_trials = 1
for m in range(n_trials):
    estimator = MLP(2*6*7, [160], 3, "sigmoid", "glorot", verbose=True)
    candidate = smart(model=estimator, params=params, env=env, p=1)
    opponent = smart(model=estimator, params=params, env=env, p=2)
    
    step = 0
    for i in tqdm(range(1000)):
        # initialise (reset) the environment
        # note: returns the initial state of the environment
        # print("episode-{} start ...".format(i))
        env.reset()
        candidate.reset()
        state = np.zeros((1, 2*6*7))
        index = 1
        while not env.game.over:
            if index == 1:
                player = candidate
            elif index == 2:
                player = opponent
            # print("P{} is playing ...".format(index))
            state, reward = train(state, player)
            # if env.game.over:
            #     if (reward == 0).all():
            #         print("\nreward: {}".format(reward))
            #         print("state:")
            #         env.game.print_grid()
            #     else:
            #         print("episode-{} done!".format(i))
            # r.append(reward)
            try:
                rs[step] = rs[step] + (step/(step + 1.))*(reward - rs[step])
            except Exception as e:
                pass
            finally:
                rs.append(reward)
            index = (index%2) + 1
            step = step + 1
        # copying nupdated state of the model to opponent
        # opponent.estimator = copy.deepcopy(candidate.estimator) 
    candidate.save('models', 'smarty_{}.pkl'.format(i+1))

# small visualisation
plt.clf()
z = np.asarray(rs)
x = z.squeeze(1)
print(x.shape)
print(x.sum(axis=0))
x = x.cumsum(axis=0)
for i in range(x.shape[1]-1):
    plt.plot(x[:,i+1], label="P{} wins".format(i+1))
plt.plot(x[:,0], label="Draw")
plt.legend(loc="best")
plt.grid(True, color="lightgrey", linestyle="--")
plt.show()

# small test against ranom player
greedy  = greed(model=estimator, params=params, env=env, p=1)
rand = random(model=None, params=params, env=env, p=2)
r = 0
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
        state, reward = play(state, p)
        r = r + reward
        index = (index%2) + 1
print(r)