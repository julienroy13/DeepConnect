import numpy as np
import matplotlib.pyplot as plt
import os

def pad_grid(grid, win_indices=None, coeff=21):

    grid = grid * 100 # adjust colors
    
    # Put some padding between each element (cause you know.. pretty)
    grid = np.repeat(grid, coeff, axis=0)
    grid = np.repeat(grid, coeff, axis=1)
    
    mask_i, mask_j = np.mgrid[0:grid.shape[0],0:grid.shape[1]]
    mask_i = np.invert((mask_i % coeff == 0) + (mask_i % coeff == coeff-1)).astype(np.int)
    mask_j = np.invert((mask_j % coeff == 0) + (mask_j % coeff == coeff-1)).astype(np.int)
    
    grid = grid * mask_i * mask_j

    # Final padding to make the sides pretty
    grid = np.pad(grid, pad_width=1, mode="constant")

    # Highlights the winning line (just for prettiness)
    if win_indices != None:
        grid = highlight_win_line(grid, win_indices, coeff)
    
    return grid

def highlight_win_line(grid, win_indices, coeff):

    # We draw a white line around each winning piece
    for i, j in win_indices:
        new_i = i*coeff + (coeff//2) + 1
        new_j = j*coeff + (coeff//2) + 1

        begin_i = new_i - (coeff//2) - 1
        end_i = new_i + (coeff//2) + 1
        begin_j = new_j - (coeff//2) - 1
        end_j = new_j + (coeff//2) + 1

        # Draws vertical lines
        grid[begin_i : end_i+1, [begin_j, begin_j+1, end_j, end_j-1]] = 255

        # Draws horizontal lines
        grid[[begin_i, begin_i+1, end_i, end_i-1], begin_j : end_j+1] = 255

    return grid

def save_game(recorder, save_dir, win_indices):
    """Takes a list of ndarray as input (recorder), makes it prettier and saves it"""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, grid in enumerate(recorder):

        if i == len(recorder)-1:
            image = pad_grid(grid, win_indices=win_indices)

        else:
            image = pad_grid(grid, win_indices=None)

        #plt.clim(0, 255)
        plt.imsave(os.path.join(save_dir, "{}.png".format(i)), image, vmin=0, vmax=255)



