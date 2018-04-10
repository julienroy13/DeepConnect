import os
import numpy as np
import scipy.signal
import pdb


class Connect4Environment(object):
	"""A class that wraps around the Connect4 game engine to accomodate a RL agent"""

	def __init__(self, n_rows=6, n_columns=7, win_streak=4):

		self.game = Connect4(n_rows, n_columns, win_streak)

	def get_state(self, grid):
		"""Transform matrix grid representation into a 3D state of one-hot vectors (n_rows x n_columns x 3)"""
		n_positions = self.game.n_rows * self.game.n_columns
		state = np.zeros(shape=(n_positions, 3))
		positions = np.reshape(grid, newshape=(n_positions,)) # uses the given grid, not necessarily the actual grid of the game

		# Fills the one-hot vectors with a one at the right index
		state[np.arange(n_positions), positions] = 1

		# Reshape state
		state = np.reshape(state, newshape=(self.game.n_rows, self.game.n_columns, 3))

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

	def get_opponent_move(self, opponent_policy="random"):
		"""Makes a move in the environment for the second player (opponent). Returns to a reward and a next state"""

		if opponent_policy == "random":
			A_s = env.game.get_valid_moves()
			a_id = np.random.randint(len(A_s))
			opponent_action = A_s[a_id]

		return opponent_action

	def get_reward(self, reward_function="win-lose-draw"):
		
		if reward_function == "win-lose-draw":
			# This rewarding
			player1_win = self.game.check_win(1)
			player2_win = self.game.check_win(2)
			if player1_win:
				reward = 1
			elif player2_win:
				reward = -1
			else:
				reward = 0

		return reward

	def get_successors(self):
		"""Returns a list of tuples containing afterstates and actions that leads to those afterstates"""

		afterstates = [] # list of tuples (successor, action)
		valid_actions = self.game.get_valid_moves()
		
		for action in valid_actions:
			successor = self.game.make_move(1, action, imaginary=True) # the state of the world won't be modified (here we only simulate)
			successor_state = self.get_state(successor)
			afterstates.append((successor_state, action))

		return afterstates

	def reset():
		"""Resets the environment"""
		self.game.reset()

	def render():
		pass # Not implemented yet


class InvalidMove(Exception):
	# Just creating this custom exception so we can count them if needed
	pass


class Connect4(object):
	"""Our game engine"""

	def __init__(self, n_rows=6, n_columns=7, win_streak=4):

		self.n_rows = n_rows
		self.n_columns = n_columns
		self.win_streak = win_streak
		self.grid = np.zeros(shape=(n_rows, n_columns), dtype=np.int)

		# Creates kernels to check the different winning conditions
		h_win_kernel = np.ones(shape=(1, win_streak)) # horizontal win
		v_win_kernel = np.ones(shape=(win_streak, 1)) # vertical win
		d1_win_kernel = np.zeros(shape=(win_streak, win_streak)) # diagonal-1 win
		d2_win_kernel = np.zeros(shape=(win_streak, win_streak)) # diagonal-2 win
		for i in range(win_streak):
			d1_win_kernel[i,i] = 1
			d2_win_kernel[i, win_streak-i-1] = 1
		self.win_kernels = [h_win_kernel, v_win_kernel, d1_win_kernel, d2_win_kernel]

	def make_move(self, player_id, column, imaginary=False):
		"""Places a piece of the player's color in the given column"""

		# Initializes next grid to current state of the game
		next_grid = np.copy(self.grid)

		# Checks if the column is full (should not happen)
		if self.grid[0, column] != 0:
			raise InvalidMove('This move is illegal. Column {} is already full.'.format(column))

		for row in range(self.n_rows):
			# If next row is empty
			if row+1 < self.n_rows and self.grid[row+1, column] == 0:
				continue # Piece keeps falling
			else:
				next_grid[row, column] = player_id
				break # Piece stops here

		if not imaginary:
			self.grid = next_grid

		return next_grid

	def get_valid_moves(self):
		"""Returns a list of columns in which it is possible to place an additional piece"""

		valid_columns = []
		# Checks for every column if we could add at least on additional piece
		for column in range(self.n_columns):
			if self.grid[0, column] == 0:
				valid_columns.append(column)

		return valid_columns
	
	def check_win(self, player_id):
		"""Checks if the provided player has won the game"""

		# Only keeps the position of the player's pieces we are concerned with
		player_pieces = (self.grid == player_id)
		for kernel in self.win_kernels:

			# Convolves the grid with the wining condition kernels
			win_mask = scipy.signal.convolve2d(player_pieces, kernel, mode="full")
			has_won = np.any(win_mask >= self.win_streak)
			
			if has_won:
				break

		return has_won

	def reset(self):
		self.grid = np.zeros(shape=(self.n_rows, self.n_columns))

	def print_grid(self):
		top = '_' * (self.n_columns+2)
		print(top)
		for i in range(self.n_rows):
			row = '|'
			for j in range(self.n_columns):
				if self.grid[i,j] == 0:
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


if __name__ == "__main__":

	option = input("HEY, you can either test the game engine (option 1) or the environment wraper (option 2) : ")

	if option == '1':
		game = Connect4()

		while 1:

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

		print("GAME OVER")

	elif option == '2':
		env = Connect4Environment()

		while 1:

			# Player moves
			afterstates = env.get_successors()
			a = int(input("Your move : "))
			next_state, reward = env.step(a) # the opponent moves as part of the environment's response to the agent's action
			env.game.print_grid()

			if env.game.check_win(1):
				print("Congratulations! You have won!")
				break

			if env.game.check_win(2):
				print("Player 2 have won. You got beaten by a random bot...")
				break

		pdb.set_trace() # easier to inspect stuff manually than print everything in the console
		print("GAME OVER")


