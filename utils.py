import pandas as pd

# Find the final index of each repetition
def end_ind(a):
  r = []
  for i in range(0, len(a)-1):
    if a[i] != a[i+1]:
      r.append(i)
  r.append(len(a)-1)
  return r


# Divide a list in sublists using indices
def create_sub_list(l,i):
  r = []
  r.append(l[:, 0:i[0]+1])
  for j in range(0, len(i)-1):
    r.append(l[:,i[j]+1:i[j+1]+1])
  return r


# Create sliding windows
def sliding_windows_attrs(WS, SHIFT, l1, l2, k, fl):
  #print(f'Starting the task {k}...')
  res = pd.DataFrame()
  [r,c] = l1.shape
  for z in range(0, c, SHIFT):
    if z + WS <= c:
      x = pd.DataFrame(l1[:, z:z + WS])
      res = pd.concat([res, x])
      x = pd.DataFrame(l2[:, z:z + WS])
      res = pd.concat([res, x])
  fl[k] = res
  #print(f'Task {k} finished')
