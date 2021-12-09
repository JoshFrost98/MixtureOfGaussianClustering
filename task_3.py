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
No_1 = 0
No_2 = 0
for item in phoneme_id:
    if item == 1:
        No_1 +=1
    if item ==2:
        No_2 +=1

X_phoneme = np.array([])
X_phoneme_1 = np.array([])
X_phoneme_2 = np.array([])
classX = []
# X_phoneme = ...
for index, item in enumerate(X_full):
    if phoneme_id[index] == 1:
        classX.append(1)
        if X_phoneme.size != 0:
            X_phoneme = np.append(X_phoneme, [item], axis = 0)
            X_phoneme_1 = np.append(X_phoneme_1, [item], axis = 0)
        else:
            X_phoneme = np.array([item])
            X_phoneme_1 = np.array([item])
for index, item in enumerate(X_full):
    if phoneme_id[index] == 2:
        classX.append(2)
        X_phoneme = np.append(X_phoneme, [item], axis = 0)
        if X_phoneme_2.size != 0:
            X_phoneme_2 = np.append(X_phoneme_2, [item], axis = 0)
        else:
            X_phoneme_2 = np.array([item])
# X_phonemes_1_2 = ...
########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phoneme, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"
# as dataset X, we will use only the samples of the chosen phoneme
p1, mu1, s1 = train_MOG(X_phoneme_1, k)
p2, mu2, s2 = train_MOG(X_phoneme_2, k)
Result1 = get_predictions(mu1, s1, p1, X_phoneme)
Result2 = get_predictions(mu2, s2, p2, X_phoneme)
Result = np.concatenate((Result1, Result2), axis = 1)
Result_ID = np.array([])
for item in Result:
    max = 0
    maxindex = 0
    for index, val in enumerate(item):
        if val>max:
            max = val
            maxindex = index

    if maxindex>k-1:
        Result_ID = np.append(Result_ID, [2], axis = 0)
    else:
        Result_ID = np.append(Result_ID, [1], axis = 0)
incorrect = 0
tot = 0
for index, result in enumerate(Result_ID):
    if result != classX[index]:
        incorrect += 1
    tot +=1

accuracy = 100* (1 - incorrect/tot)
fig, ax1 = plt.subplots()
print(X_phoneme.shape)
print(Result_ID.shape)
plot_data_all_phonemes(X=X_phoneme, phoneme_id=Result_ID, ax=ax1)
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_1_2_predictions_k3.png')
plt.savefig(plot_filename)


print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))


################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()