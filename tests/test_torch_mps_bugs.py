import numpy as np
import torch
mps_device = torch.device("mps")

## Create a numpy matrix with many zeros
np.random.seed(0)
Numpy_Test = np.random.random(200000000)
indices = np.random.choice(np.arange(Numpy_Test.size), replace=False,size=int(Numpy_Test.size * 0.6))
Numpy_Test[indices] = 0
Numpy_Matrix = Numpy_Test.reshape((20000,10000))

## Get the indices of non-zero values in the matrix, and convert these indices into a numpy array
indices = np.where(Numpy_Matrix != 0)
indices = np.asarray(indices)

## Use numpy, torch, or a torch.mps object to find where indices[1] == 8000
# Using np.where
np_w = np.where(indices[1] == 8000)[0]

# Using torch.where
t_cpu_w = torch.where(torch.from_numpy(indices)[1] == 8000)[0]

# Using torch.where with an NPS object
t_mps_w = torch.where(torch.from_numpy(indices)[1].to(mps_device) == 8000)[0]


print(np_w)
print(t_cpu_w)
print(t_mps_w)

assert (t_mps_w.to(torch.device("cpu")) == t_cpu_w).all()
