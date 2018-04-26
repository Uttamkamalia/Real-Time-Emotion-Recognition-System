
import numpy as np

labels = np.load("train_labels.npy")
data = np.load("train_data.npy")
print(labels.shape)
print(data.shape)

print(labels[0])
print(labels[2])


print("done")    
