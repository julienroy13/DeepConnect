import numpy as np
from agent import MLP, smart, random
from env import Connect4Environment

from matplotlib import pyplot as plt
from tqdm import tqdm
import pdb

PLAYER1_LEARNS = True
PLAYER2_LEARNS = False
TRAIN_TIME = int(5e3)

# training parameters
params = {"epsilon": 0.1, 
          "gamma": 1., 
          "lambda": .5, 
          "alpha": 1e-2}

# environment
env = Connect4Environment()

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
    
# Instanciate the value network
estimator = MLP(env.d*env.game.n_rows*env.game.n_columns+2, [160], 3, "relu", "glorot", verbose=True)

# Instanciates the two players
player1 = smart(model=estimator, params=params, env=env, p=1)
player2 = smart(model=estimator, params=params, env=env, p=2)
#player2 = random(model=estimator, params=params, env=env, p=2)

for i in tqdm(range(TRAIN_TIME)):
    
    # Resets the environment and the player1's eligibility trace
    env.reset()
    if PLAYER1_LEARNS: player1.reset()
    if PLAYER2_LEARNS: player2.reset()

    # Initial state
    state = np.zeros((1, env.d*env.game.n_rows*env.game.n_columns+2))

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

        if PLAYER1_LEARNS: player1.update(state, reward, next_state)
        if PLAYER2_LEARNS: player2.update(state, reward, next_state)
        
        state = next_state
    
    rewards.append(reward)

    if i+1 in [1e3, 2e3, 3e3, 4e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6]:
        player1.save('models', 'FlipPlayers_{}k.pkl'.format(int((i+1)/1e3)))
        count_wins(rewards)

print("\nP1 started {} times\nP2 started {} times\n".format((np.array(started_game)==1).sum(), (np.array(started_game)==2).sum()))
