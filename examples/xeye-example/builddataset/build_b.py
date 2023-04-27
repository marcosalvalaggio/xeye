import xeye
import numpy as np 
import matplotlib.pyplot as plt 

index = 0
img_types = 1
label = ['b']
num = 10
height = 100
width = 100
standby_time = 0

data = xeye.ManualDataset(index=index, img_types=img_types, label=label, num=num, height=height, width=width)
data.preview()
data.gray()
data.compress_train_test(perc=0.2)
data.just_compress('b')

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