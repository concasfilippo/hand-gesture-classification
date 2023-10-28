#versione del 2023-07-18 n.1
import h5py
import numpy as np
from utils import *
import threading
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

SUBJECT = 1
CHANNELS = 128

# sliding window parameters
WS = 307
SHIFT = 307


matF = h5py.File(f's{SUBJECT}.mat','r')

E = np.array(matF['emg_extensors'])
F = np.array(matF['emg_flexors'])


# Reshape the files so that the channels are aligned and no longer in a square matrix
#example
'''
X = np.array([[[155, 33, 129],[161, 218, 6]],[[215, 142, 235],[143, 249, 164]],[[221, 71, 229],[ 56, 91, 120]],[[236, 4, 177],[171, 105, 40]]])
[l1,r1,c1] = X.shape
Y = X.reshape(l1*r1, c1)
[r2,c2] = Y.shape
'''
[l,r,c] = E.shape
E = E.reshape(l*r, c)
F = F.reshape(l*r, c)

C = np.array(matF['adjusted_class'])

'''
# Plot the first sliding window of the first channel before signal noramlization/standardization
x = F[0,0:WS]
y = np.arange(0,WS)
plt.title("before noramlization/standardization")
plt.plot(y, x, color="red")
plt.show()
'''
# Normalize (range 0, 1)
# min = E.min()
# max = E.max()
# print('Min: %f, Max: %f' % (min,max))
# E = (E-min) / (max-min)
# min = F.min()
# max = F.max()
# print('Min: %f, Max: %f' % (min,max))
# F = (F-min) / (max-min)
# # or
# scaler = MinMaxScaler()
# scaler.fit(E)
# E = scaler.transform(E)
# scaler.fit(F)
# F = scaler.transform(F)
'''
'''
# Standardize
# mean = E.mean()
# std = E.std()
# print('Mean: %f, Standard deviation: %f' % (mean,std))
# E = (E-mean) / std
# mean = F.mean()
# std = F.std()
# print('Mean: %f, Standard deviation: %f' % (mean,std))
# F = (F-mean) / std
# # or
# scaler = StandardScaler()
# scaler.fit(E)
# E = scaler.transform(E)
# scaler.fit(F)
# F = scaler.transform(F)
'''
# Plot the first sliding window of the first channel after signal noramlization/standardization
x = F[0,0:WS]
y = np.arange(0,WS)
plt.title("after noramlization/standardization")
plt.plot(y, x, color="red")
plt.show()
#'''
# We find the index of the final sample of each repetition
end_reps_ind = end_ind(C[0])
# We divide the recorded samples into lists. Each list contains the samples of a single repetition.
l1 = create_sub_list(E, end_reps_ind)
l2 = create_sub_list(F, end_reps_ind)
'''
# Number of repetitions for each gesture
p = []
for z in range(0,len(end_reps_ind)-1):
  p.append(C[:,end_reps_ind[z]][0])
print(np.bincount(p))
'''

# We create sliding windows for each repetition. 
# We do not consider the case where no action takes place.
fl = [None] * math.floor((len(end_reps_ind)-1)/2)
count = 0

# We used threads to speed up the process
for k in range(1, len(end_reps_ind), 20):
  threads = list()
  for i in range(0,19,2):
    if k + i < len(end_reps_ind):
      t = threading.Thread(target=sliding_windows_attrs, args=(WS, SHIFT, l1[k + i], l2[k + i], count, fl))
      threads.append(t)
      t.start()
      count = count + 1
  for index, thread in enumerate(threads):
    thread.join()


# We create a vector containing the number of sliding windows obtained from each repetition and 
# concatenate the sliding windows of all the repetitions.
sw_per_repetition = []
sw = pd.DataFrame()

for j in range(0, len(fl)):
  sw_per_repetition.append(fl[j].shape[0])
  fl[j].to_csv('EMGdataset.sw.csv', mode='a', header=False, index=False)

pd.DataFrame(sw_per_repetition).to_csv('EMGdataset.sw_per_repetition.csv', header = False, index = False)


# We create the subrepetitions_classes and the subrepetitions_idx files
subrepetitions_classes = pd.DataFrame(columns = ['label'])
subrepetitions_idx = pd.DataFrame(columns = ['idx'])
idx = 1
idx_class = 1

for j in range(0, len(sw_per_repetition)):
  num_subrepetitions = math.floor(sw_per_repetition[j] / CHANNELS)
  # For each subrepetition we have the label
  subrepetitions_class = C[:, end_reps_ind[idx_class]]
  idx_class = idx_class+2
  i = pd.DataFrame(subrepetitions_class * np.ones(num_subrepetitions), dtype=int, columns=['label'])
  subrepetitions_classes = pd.concat([subrepetitions_classes, i])
  # For each subrepetition we have an id
  p = pd.DataFrame(idx * np.ones(num_subrepetitions), dtype=int, columns=['idx'])
  if idx == 5:
    idx =1
  else:
    idx = idx + 1
  subrepetitions_idx = pd.concat([subrepetitions_idx, p])

subrepetitions_classes.to_csv('EMGdataset.subrepetitions_classes.csv', header = False, index = False)
subrepetitions_idx.to_csv('EMGdataset.subrepetitions_idx.csv', header = False, index = False)