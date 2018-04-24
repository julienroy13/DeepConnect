import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

def pad_grid(grid, win_indices=None, coeff=21):

    grid = grid * 100 # adjust colors
    grid[grid==0] = -50 # Sets the reachable but empty locations to -50 (to differentiate them from the grid that will be set to 0)
    
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

        # Change colors manually according to cmap='nipy_spectral'
        image[image == -100] = 7        # Dark violet
        image[image == -50] = 7         # Dark violet
        image[image == 0] = 5           # Dark gray
        image[image == 100] = 180       # Yellow
        image[image == 200] = 220       # Red
        plt.imsave(os.path.join(save_dir, "{}.png".format(i)), image, vmin=0, vmax=255, cmap='nipy_spectral')

def plot_all_errors(save_dir, all_errors, final_steps):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10,4))
    plt.title("TD errors", fontweight='bold')
    plt.plot(all_errors[1:, 1], color='blue', label='P1 wins')
    plt.plot(all_errors[1:, 2], color='orange', label='P2 wins')
    plt.plot(all_errors[1:, 0], color='green', label='Draw')
    #plt.vlines(final_steps, ymin=-1, ymax=1, color='grey', linestyle='--', label='Terminal steps')
    plt.xlabel('Timesteps')
    plt.ylabel('TD error')
    plt.legend(loc='best')

    plt.savefig(os.path.join(save_dir, 'errors.png'), bbox_inches='tight')
    plt.close()

def plot_final_errors(save_dir, final_errors):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10,4))
    plt.title("Final TD errors", fontweight='bold')
    plt.plot(final_errors[1:, 1], color='blue', label='P1 wins')
    plt.plot(final_errors[1:, 2], color='orange', label='P2 wins')
    plt.plot(final_errors[1:, 0], color='green', label='Draw')
    plt.xlabel('Episodes')
    plt.ylabel('Final TD error')
    plt.legend(loc='best')

    plt.savefig(os.path.join(save_dir, 'finalerrors.png'), bbox_inches='tight')
    plt.close()




