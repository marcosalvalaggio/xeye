import xeye 
import numpy as np 
import matplotlib.pyplot as plt 

path = ['a.npz', 'b.npz']
label = [0,1]
data = xeye.BuildDataset(path=path, label=label, size = None, color=False, split=True, perc=0.2)
data.build()

data = np.load('dataset.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
print(X_train.shape, type(X_train))
print(X_test.shape, type(X_test))
print(y_train.shape, type(y_train))
print(y_test.shape, type(y_test))
print(y_train)
print(y_test)
plt.imshow(X_train[0], cmap='gray')
plt.title(f'load image: {y_train[0]}')
plt.show()
plt.imshow(X_train[2], cmap='gray')
plt.title(f'load image: {y_train[2]}')
plt.show()
plt.imshow(X_test[0], cmap='gray')
plt.title(f'load image: {y_test[0]}')
plt.show()