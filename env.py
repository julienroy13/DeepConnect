import os
import numpy as np
import scipy.signal
import pdb
import time
import utils


class Connect4Environment(object):
    """A class that wraps around the Connect4 game engine to accomodate a RL agent"""

    def __init__(self, n_rows=6, n_columns=7, win_streak=4, turn_info=True):

        self.game = Connect4(n_rows, n_columns, win_streak)

        # The dimensionality of the board representation for each location
        self.d = 4
        self.turn_info = turn_info

    def get_state(self, grid):
        """Transform matrix grid representation into a 3D state of one-hot vectors (n_rows x n_columns x 3)"""
        
        # Information regarding grid positionning
        state = np.stack([grid==-1, grid==0, grid==1, grid==2]).astype(np.int).flatten()
        
        # Information regarding whose turn it is to play
        if self.game.turn == 1:
            turn_info = np.array([0, 1]) # Inverse (on purpose, because it actually looks at afterstates)
        elif self.game.turn == 2:
            turn_info = np.array([1, 0]) # Inverse (on purpose, because it actually looks at afterstates)
        else:
            raise Exception("Wrong : env.game.turn == {}".format(self.game.turn))

        if self.turn_info == False:
            turn_info = np.array([0, 0])

        state = np.concatenate((state, turn_info))
        return state

    def step(self, agent_action):
        """Makes a move in the environment for the first player (agent), and then plays for the second player (opponent). 
            Returns to a reward and a next state"""

        # Moves for player 1 (agent)
        self.game.make_move(1, agent_action, imaginary=False)

        # Moves for player 2 (opponent)
        opponent_action = self.get_opponent_move(opponent_policy="random")
        next_state = self.get_state(self.game.make_move(2, opponent_action, imaginary=False))

        # gets the reward according to a chosen reward function (mode)
        reward = self.get_reward(reward_function="win-lose-draw")

        return next_state, reward

    def play(self, player, action):
        """Makes a move in the environment for a given player. Returns to a reward and a next state"""

        # makes a move and converts resulting grid to state representation
        obs = self.game.make_move(player, action, imaginary=False)
        next_state = self.get_state(obs)

        # gets the reward according to a chosen reward function (mode)
        reward = self.get_reward(reward_function="win-lose-draw")

        return next_state, reward

    def get_reward(self, reward_function="win-lose-draw"):
        
        if reward_function == "win-lose-draw":
            reward = np.zeros(shape=(1, 3))
            # This rewarding
            player1_win = self.game.check_win(1)
            player2_win = self.game.check_win(2)
            draw = self.game.check_draw()
            if draw:
                reward[0, 0] = 1.
                #print("It's a draw game!")
            elif player1_win:
                reward[0, 1] = 1.
                #print("P1 wins!")
            elif player2_win:
                reward[0, 2] = 1.
                #print("P2 wins!")

        return reward

    def get_successors(self, player):
        """Returns a list of tuples containing afterstates and actions that leads to those afterstates"""

        afterstates = [] # list of tuples (successor, action)
        valid_actions = self.game.get_valid_moves()
        
        for action in valid_actions:
            successor = self.game.make_move(player, action, imaginary=True) # the state of the world won't be modified (here we only simulate)
            successor_state = self.get_state(successor)
            afterstates.append((successor_state, action))

        return afterstates

    def reset(self):
        """Resets the environment"""
        self.game.reset()


class InvalidMove(Exception):
    # Just creating this custom exception so we can count them if needed
    pass


class Connect4(object):
    """Our game engine"""

    def __init__(self, n_rows=6, n_columns=7, win_streak=4):

        self.n_rows = n_rows
        self.n_columns = n_columns
        self.win_streak = win_streak

        # Initializes grid
        self.grid = np.zeros(shape=(n_rows, n_columns), dtype=np.int)
        self.grid[:-1,:] = -1 
        self.turn = 1 # Player 1 starts the game

        # Recorder
        self.recorder = [self.grid]
        self.record_game = False
        
        # Termination info
        self.over = False
        self.win_type = None # String
        self.win_indices = None # Indices of winning line
        self.winner = None # Either 1 or 2

        # Creates kernels to check the different winning conditions
        h_win_kernel = np.ones(shape=(1, win_streak)) # horizontal win
        v_win_kernel = np.ones(shape=(win_streak, 1)) # vertical win
        d1_win_kernel = np.zeros(shape=(win_streak, win_streak)) # diagonal-1 win
        d2_win_kernel = np.zeros(shape=(win_streak, win_streak)) # diagonal-2 win
        for i in range(win_streak):
            d1_win_kernel[i,i] = 1
            d2_win_kernel[i, win_streak-i-1] = 1
        self.win_kernels = [(h_win_kernel, "horizontal"), (v_win_kernel, "vertical"), (d1_win_kernel, "diagonal1"), (d2_win_kernel, "diagonal2")]

    def make_move(self, player_id, column, imaginary=False):
        """Places a piece of the player's color in the given column"""

        # Initializes next grid to current state of the game
        next_grid = np.copy(self.grid)

        # Checks if the column is out of bound (should not happen)
        if column < 0 or column >= self.n_columns:
            raise InvalidMove('This move is impossible. Column {} is out of bound.'.format(column))

        # Checks if the column is full (should not happen)
        if self.grid[0, column] > 0:
            raise InvalidMove('This move is illegal. Column {} is already full.'.format(column))

        for row in range(self.n_rows):
            # If next row is empty
            if self.grid[row, column] == -1:
                continue # Piece keeps falling
            elif self.grid[row, column] == 0:
                next_grid[row, column] = player_id
                if row != 0:
                    next_grid[row-1, column] = 0
                break # Piece stops here
            else:
                raise Exception("COLLISION : There should be a layer of 0s above any piece in the grid")

        # Updates the grid only if the move wasn't imaginary 
        if not imaginary:
            self.grid = next_grid
            self.recorder.append(self.grid)

        # Updates whose turn it is to play
        if player_id == 1:
            self.turn = 2
        elif player_id == 2:
            self.turn = 1

        return next_grid

    def get_valid_moves(self):
        """Returns a list of columns in which it is possible to place an additional piece"""

        valid_columns = []
        # Checks for every column if we could add at least on additional piece
        for column in range(self.n_columns):
            if self.grid[0, column] <= 0:
                valid_columns.append(column)

        return valid_columns
    
    def check_win(self, player_id):
        """Checks if the provided player has won the game"""

        # Only keeps the position of the player's pieces we are concerned with
        player_pieces = (self.grid == player_id)
        for kernel, win_type in self.win_kernels:

            # Convolves the grid with the wining condition kernels
            win_mask = scipy.signal.convolve2d(player_pieces, kernel, mode="full")
            has_won = np.any(win_mask >= self.win_streak)
            
            if has_won:
                self.over = True
                self.winner = player_id
                self.win_type = win_type
                self.win_indices = self.get_win_indices(win_mask, win_type)
                break

        return has_won

    def check_draw(self):
        """Checks if the game ended in a draw"""
        is_draw = (not self.check_win(1)) and (not self.check_win(2)) and ((self.grid > 0).all())
        if is_draw:
            self.over = True

        return is_draw

    def get_win_indices(self, win_mask, win_type):
        """Return a list of indices representing the position of the pieces in the winning line"""
        i, j = np.unravel_index(np.argmax(win_mask), win_mask.shape)
        win_indices = []

        if win_type == 'horizontal':
            for k in range(self.win_streak):
                win_indices.append((i, j-k))

        elif win_type == 'vertical':
            for k in range(self.win_streak):
                win_indices.append((i-k, j))

        elif win_type == 'diagonal1':
            for k in range(self.win_streak):
                win_indices.append((i-k, j-k))

        elif win_type == 'diagonal2':
            for k in range(self.win_streak):
                win_indices.append((i-k, j-self.win_streak+1+k))

        return win_indices


    def reset(self, record_next_game=False):
        """Reset the game engines to prepare for a new match, optionally allows to record the next game"""
        self.grid = np.zeros(shape=(self.n_rows, self.n_columns), dtype=np.int)
        self.grid[:-1,:] = -1 
        self.turn = 1 # Player 1 starts the game
        self.over = False
        self.win_type = None # String
        self.win_indices = None # Indices of winning line
        self.winner = None # Either 1 or 2
        self.recorder = [self.grid]
        self.record_game = record_next_game

    def print_grid(self):
        """Just print the grid in the terminal with unicode characters"""
        top = '_' * (self.n_columns+2)
        print(top)
        for i in range(self.n_rows):
            row = '|'
            for j in range(self.n_columns):
                if self.grid[i,j] == -1:
                    row += 'x'
                elif self.grid[i,j] == 0:
                    row += ' '
                elif self.grid[i,j] == 1:
                    row += '\u25CF'
                elif self.grid[i,j] == 2:
                    row += '\u25CB'
                else:
                    raise ValueError("Irregular value in 'self.grid' : {}".format(self.grid))
            print(row + '|')
        bottom = '\u2588' * (self.n_columns+2) # '\u203E'
        print(bottom)
        print(' 0123456 ')
        print("\n")

    def get_record(self):
        """Returns a list of ndarray representing every grid state of the game so far"""
        return self.recorder


if __name__ == "__main__":

    option = input("HEY, you can either test the game engine (option 1) or the environment wraper (option 2) : ")

    if option == '1':
        game = Connect4()
        game.reset(record_next_game=True)

        while not game.over:

            # Player moves
            a = int(input("Your move : "))
            game.make_move(1, a)
            game.print_grid()

            if game.check_win(1):
                print("Congratulations! You have won!")
                break

            # Random move from opponent
            A_s = game.get_valid_moves()
            a_id = np.random.randint(len(A_s))
            game.make_move(2, A_s[a_id])
            game.print_grid()

            if game.check_win(2):
                print("Player 2 have won. You got beaten by a random bot...")
                break

        utils.save_game(game.recorder, "results", game.win_indices)
        print("GAME OVER")

    elif option == '2':
        env = Connect4Environment()

        while not game.over:

            # Player moves
            afterstates = env.get_successors()
            a = int(input("Your move : "))
            next_state, reward = env.step(a) # the opponent moves as part of the environment's response to the agent's action
            env.game.print_grid()

            if env.game.check_win(1):
                print("Congratulations! You have won!")

            if env.game.check_win(2):
                print("Player 2 have won. You got beaten by a random bot...")

        pdb.set_trace() # easier to inspect stuff manually than print everything in the console
        print("GAME OVER")