import numpy as np
from agent import MLP, smart, random
from env import Connect4Environment

from matplotlib import pyplot as plt
from tqdm import tqdm
import pdb

PLAYER1_LEARNS = True
PLAYER2_LEARNS = True
TRAIN_TIME = 1000

# training parameters
params = {"epsilon": 0.01, 
          "gamma": 1., 
          "lambda": 0.9, 
          "alpha": 1e-3}

# environment
env = Connect4Environment()

def play_and_learn(state, player):
    action = player.select_action()
    next_state, reward = env.play(player.p, action)
    player.update(state, reward, next_state)

    return next_state, reward

def play(state, player):
    action = player.select_action()
    next_state, reward = env.play(player.p, action)

    return next_state, reward

# example of self-training
rewards = []
steps = []
n_trials = 1
started_game = []
for m in range(n_trials):
    
    # Instanciate the value network
    estimator = MLP(2*env.game.n_rows*env.game.n_columns, [160], 3, "sigmoid", "glorot", verbose=True)
    
    # Instanciates the two players
    player1 = smart(model=estimator, params=params, env=env, p=1)
    player2 = smart(model=estimator, params=params, env=env, p=2)
    
    step = 0
    for i in tqdm(range(TRAIN_TIME)):
        
        # Resets the environment and the player1's eligibility trace
        env.reset()
        if PLAYER1_LEARNS: player1.reset()
        if PLAYER2_LEARNS: player2.reset()

        # Initial state
        state = np.zeros((1, 2*env.game.n_rows*env.game.n_columns))

        # Throws a coin to decide which player starts the game
        env.game.turn = np.random.randint(low=1, high=3)
        started_game.append(env.game.turn)

        while not env.game.over:
            # If is the turn of player 1
            if env.game.turn == 1:
                if PLAYER1_LEARNS: 
                    state, reward = play_and_learn(state, player1)
                else: 
                    state, reward = play(state, player1)
            
            # If is the turn of player 2
            elif env.game.turn == 2:
                if PLAYER2_LEARNS: 
                    state, reward = play_and_learn(state, player2)
                else: 
                    state, reward = play(state, player2)

            else:
                raise Exception("It is supposed to be either player 1's or 2's turn")
            
            step = step + 1
        
        rewards.append(reward)
        steps.append(step)

    player1.save('models', 'smarty_{}.pkl'.format(i+1))

print("\nP1 started {} times\nP2 started {} times\n".format((np.array(started_game)==1).sum(), (np.array(started_game)==2).sum()))
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
