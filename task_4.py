import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *
from train_MOG import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:,0] = f1
X_full[:,1] = f2
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

# X_phonemes_1_2 = ...


X_phonemes_1_2 = np.array([])
X_phoneme_1 = np.array([])
X_phoneme_2 = np.array([])
classX = []
# X_phonemes_1_2 = ...
for index, item in enumerate(X_full):
    if phoneme_id[index] == 1:
        classX.append(1)
        if X_phonemes_1_2.size != 0:
            X_phonemes_1_2 = np.append(X_phonemes_1_2, [item], axis = 0)
            X_phoneme_1 = np.append(X_phoneme_1, [item], axis = 0)
        else:
            X_phonemes_1_2 = np.array([item])
            X_phoneme_1 = np.array([item])
for index, item in enumerate(X_full):
    if phoneme_id[index] == 2:
        classX.append(2)
        X_phonemes_1_2 = np.append(X_phonemes_1_2, [item], axis = 0)
        if X_phoneme_2.size != 0:
            X_phoneme_2 = np.append(X_phoneme_2, [item], axis = 0)
        else:
            X_phoneme_2 = np.array([item])
# X_phonemes_1_2 = ...
########################################/

# as dataset X, we will use only the samples of phoneme 1 and 2
X = X_phonemes_1_2.copy()

min_f1 = int(np.min(X[:,0]))
max_f1 = int(np.max(X[:,0]))
min_f2 = int(np.min(X[:,1]))
max_f2 = int(np.max(X[:,1]))
N_f1 = max_f1 - min_f1
N_f2 = max_f2 - min_f2
print('f1 range: {}-{} | {} points'.format(min_f1, max_f1, N_f1))
print('f2 range: {}-{} | {} points'.format(min_f2, max_f2, N_f2))

#########################################
# Write your code here
grid = np.empty((N_f1+1, N_f2+1, 3))
grid[:,:,0], grid[:,:,1] = np.mgrid[min_f1:max_f1+1:1, min_f2:max_f2+1:1]
# Create a custom grid of shape N_f1 x N_f2
# The grid will span all the values of (f1, f2) pairs, between [min_f1, max_f1] on f1 axis, and between [min_f2, max_f2] on f2 axis
# Then, classify each point [i.e., each (f1, f2) pair] of that grid, to either phoneme 1, or phoneme 2, using the two trained GMMs
# Do predictions, using GMM trained on phoneme 1, on custom grid
# Do predictions, using GMM trained on phoneme 2, on custom grid
# Compare these predictions, to classify each point of the grid
# Store these prediction in a 2D numpy array named "M", of shape N_f2 x N_f1 (the first dimension is f2 so that we keep f2 in the vertical axis of the plot)
# M should contain "0.0" in the points that belong to phoneme 1 and "1.0" in the points that belong to phoneme 2
########################################/

################################################
# Visualize predictions on custom grid
p1, mu1, s1 = train_MOG(X_phoneme_1, k)
p2, mu2, s2 = train_MOG(X_phoneme_2, k)
for x, column in enumerate(grid):
    result_1 = get_predictions(mu1,s1,p1,column[:,0:2])
    result_2 = get_predictions(mu2,s2,p2,column[:,0:2])
    for y, out in enumerate(result_1):
        if max(out)>max(result_2[y]):
            grid[x,y,2] = 1
        else:
            grid[x,y,2] = 2
datapoints = grid.reshape(-1, grid.shape[-1])
# Create a figure
#fig = plt.figure()
fig, ax = plt.subplots()
# plot_data_all_phonemes(X=datapoints[:,0:2], phoneme_id = datapoints[:,3], ax=ax1)
# use aspect='auto' (default is 'equal'), to force the plotted image to be square, when dimensions are unequal
plt.imshow(np.transpose(grid[:,:,2]), aspect='auto')
# set label of x axis
ax.set_xlabel('f1')
# set label of y axis
ax.set_ylabel('f2')

# set limits of axes
plt.xlim((0, N_f1))
plt.ylim((0, N_f2))

# set range and strings of ticks on axes
x_range = np.arange(0, N_f1, step=50)
x_strings = [str(x+min_f1) for x in x_range]
plt.xticks(x_range, x_strings)
y_range = np.arange(0, N_f2, step=200)
y_strings = [str(y+min_f2) for y in y_range]
plt.yticks(y_range, y_strings)

# set title of figure
title_string = 'Predictions on custom grid'
plt.title(title_string)

# add a colorbar
plt.colorbar()

# N_samples = int(X.shape[0]/2)
# plt.scatter(X[:N_samples, 0] - min_f1, X[:N_samples, 1] - min_f2, marker='.', color='red', label='Phoneme 1')
# plt.scatter(X[N_samples:, 0] - min_f1, X[N_samples:, 1] - min_f2, marker='.', color='green', label='Phoneme 2')

# add legend to the subplot
plt.legend()

# save the plotted points of the chosen phoneme, as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'GMM_predictions_on_grid.png')
plt.savefig(plot_filename)

# ################################################
# # enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()