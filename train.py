import numpy as np
import utils
from agent import MLP, smart, random
from env import Connect4Environment

from matplotlib import pyplot as plt
from tqdm import tqdm
import pdb
import json
import os
import torch

options = {
    "PLAYER1_LEARNS" : True,
    "PLAYER2_LEARNS" : False,
    "TRAIN_TIME" : int(5e3),
    "GRAPHS" : True,
    "TCL" : False, # False, 'v1' or 'v2'
    "FLIP" : False,
    "TRAIN_VS_RANDOM" : True,
    "EXP_NAME" : "vsRandNoTurnInfo",
    "SEED" : 1234,
    "TURN_INFO" : False}


# training parameters
params = {"epsilon": 0.1, 
          "gamma": 1., 
          "lambda": 0., 
          "alpha": 1e-2}


# Initializes the seeds
np.random.seed(options['SEED'])
torch.manual_seed(options['SEED'])

# Creates saving directory
save_dir = os.path.join('experiments', options['EXP_NAME'])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Writes parameters to text file
with open(os.path.join(save_dir, 'hyperparams.txt'), 'w+') as hyperparam_file:
    json.dump(options, hyperparam_file, indent=2)
    json.dump(params, hyperparam_file, indent=2)

# environment
env = Connect4Environment(turn_info=options['TURN_INFO'])

def play(state, player):
    action = player.select_action()
    next_state, reward = env.play(player.p, action)

    return next_state, reward

def count_wins(rewards):
    # counts the number of wins
    player1_wins = 0
    player2_wins = 0
    draws = 0
    for r in rewards:
        assert r.sum() == 1
        
        if r[0,0] == 1:
            draws += 1
        elif r[0,1] == 1:
            player1_wins += 1
        elif r[0,2] == 1:
            player2_wins += 1

    print("Player1 wins : {}\nPlayer2 wins : {}\nDraws : {}".format(player1_wins, player2_wins, draws))

# example of self-training
rewards = []
steps = []
started_game = []

all_errors = np.empty(shape=(1,3))
final_errors = np.empty(shape=(1,3))
final_steps = []
    
# Instanciate the value network
estimator = MLP(env.d*env.game.n_rows*env.game.n_columns+2, [180], 3, "relu", "glorot", verbose=True)

# Instanciates the two players
player1 = smart(model=estimator, params=params, env=env, p=1, tcl=options['TCL'])
if options['TRAIN_VS_RANDOM'] : 
    player2 = random(model=estimator, params=params, env=env, p=2)
else:
    player2 = smart(model=estimator, params=params, env=env, p=2, tcl=options['TCL'])

# TRAINING LOOP
total_step = 0
for i in tqdm(range(options['TRAIN_TIME'])):
    
    # Resets the environment and the player1's eligibility trace
    env.reset()
    if options['PLAYER1_LEARNS']: player1.reset()
    if options['PLAYER2_LEARNS']: player2.reset()

    # Initial state
    state = np.zeros((1, env.d*env.game.n_rows*env.game.n_columns+2))

    if options['FLIP']:
        # Flip coin to redefine who plays as player1 and who plays as player2
        player1.p = np.random.choice([1, 2])
        if player1.p == 1:
            player2.p = 2
        elif player1.p == 2:
            player2.p = 1

    
    # Throws a coin to decide which player starts the game
    env.game.turn = np.random.choice([1, 2])
    started_game.append(env.game.turn)

    while not env.game.over:
        # If is the turn of player 1
        if env.game.turn == player1.p:
            next_state, reward = play(state, player1)                    
        
        # If is the turn of player 2
        elif env.game.turn == player2.p:
            next_state, reward = play(state, player2)

        else:
            raise Exception("It is supposed to be either player 1's or 2's turn")

        # Learning step
        if options['PLAYER1_LEARNS']: error = player1.update(state, reward, next_state)
        if options['PLAYER2_LEARNS']: error = player2.update(state, reward, next_state)

        # Saves the error for graph
        if options['GRAPHS'] : all_errors = np.concatenate((all_errors, error), axis=0)
        
        # Ends the current step
        state = next_state
        total_step += 1
    
    # Saves the final error for graph
    if options['GRAPHS'] : final_errors = np.concatenate((final_errors, error), axis=0)
    if options['GRAPHS'] : final_steps.append(total_step)
    rewards.append(reward)

    if i+1 in [1e3, 2e3, 3e3, 4e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6]:
        player1.save(save_dir, options['EXP_NAME']+'_{}k.pkl'.format(int((i+1)/1e3)))
        count_wins(rewards)
        if options['GRAPHS'] : utils.plot_all_errors(save_dir, all_errors, final_steps)
        if options['GRAPHS'] : utils.plot_final_errors(save_dir, final_errors)

print("\nP1 started {} times\nP2 started {} times\n".format((np.array(started_game)==1).sum(), (np.array(started_game)==2).sum()))
