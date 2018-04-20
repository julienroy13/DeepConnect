import numpy as np
from agent import MLP, smart, random
from env import Connect4Environment

from matplotlib import pyplot as plt
from tqdm import tqdm
import pdb
import os

N_GAMES = 100

def play(state, player):
    action = player.select_action()
    next_state, reward = env.play(player.p, action)

    return next_state, reward

# environment
env = Connect4Environment()
params = {"epsilon": 0., 
          "gamma": 1., 
          "lambda": 0.9, 
          "alpha": 1e-3}

# small test against random player
estimator = MLP(env.d*env.game.n_rows*env.game.n_columns, [160], 3, "sigmoid", "glorot", verbose=True)
agent  = smart(model=estimator, params=params, env=env, p=1)
agent.load(os.path.join('models', 'newshit_100k.pkl'))

randy = random(model=None, params=params, env=env, p=2)

# OUR AGENT VS RANDOM
wins = 0
for i in tqdm(range(N_GAMES)):
    
    # Resets the environment and initial state
    env.reset()
    state = np.zeros((1, env.d*env.game.n_rows*env.game.n_columns))

    # Throws a coin to decide which player starts the game
    env.game.turn = np.random.choice([1, 2])
    while not env.game.over:
        # If is the turn of player 1
        if env.game.turn == 1:
            state, reward = play(state, agent)
        
        # If is the turn of player 2
        elif env.game.turn == 2:
            state, reward = play(state, randy)

    assert reward.sum() == 1
    wins = wins + reward[0,1]

print("Our agent has won : {}/{}".format(int(wins), N_GAMES))


# RANDOM VS RANDOM
randy2 = random(model=None, params=params, env=env, p=1)
wins = 0
for i in tqdm(range(N_GAMES)):
    
    # Resets the environment and initial state
    env.reset()
    state = np.zeros((1, env.d*env.game.n_rows*env.game.n_columns))

    # Throws a coin to decide which player starts the game
    env.game.turn = np.random.choice([1, 2])
    while not env.game.over:
        # If is the turn of player 1
        if env.game.turn == 1:
            state, reward = play(state, randy2)
        
        # If is the turn of player 2
        elif env.game.turn == 2:
            state, reward = play(state, randy)

    assert reward.sum() == 1
    wins = wins + reward[0,1]

print("Randy2 has won : {}/{}".format(int(wins), N_GAMES))