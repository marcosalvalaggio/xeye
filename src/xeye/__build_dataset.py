import cv2 
import os
from sklearn.model_selection import train_test_split
import numpy as np
import time


class BuildDataset:
    def __init__(self, path: list[str], label: list[int], size: tuple = None, color: bool = True, split: bool = True, perc: float = 0.1) -> None:
        self.path = path
        self.label = label 
        self.size = size
        self.color = color 
        self.split = split
        self.perc = perc
        self.height = 0
        self.width = 0 
        self.tensor = {}
        self.temp_tensor = []
    

    def _control(self)->None:
        """
        The method checks if the datasets to merge have the same colour space.
        """
        height = []
        width = []
        color_ch = []
        for i in range(len(self.path)):
            data = np.load(f'{self.path[i]}')
            x = data['x']
            height.append(x.shape[1]) # height
            width.append(x.shape[2]) # width
            try:
                color_ch.append(x.shape[3]) # color 
            except:
                color_ch.append(1)
        # Size control
        if self.size == None:
            self.size = []
            self.size.append(max(height))
            self.size.append(max(width))
            self.size = tuple(self.size)
        else:
            pass
        # Color channels control
        if len(set(color_ch)) != 1:
            raise ValueError("Datasets with different colour spaces...used datasets with the same colour spaces for the images")
        else:
            pass


    def build(self)->None:
        """
        Merge the datasets in a new one with the parameters instance variables indicated. 
        """
        # control method calling 
        self._control()
        # control if the path and label lists have the same length
        if len(self.path) != len(self.label):
            raise ValueError("Path and label lists doesn't have the same length...")
        # Create the tensor X
        if self.color == True:
            self.tensor['X'] = np.empty((0,self.size[0],self.size[1],3)).astype('uint8')
        else: 
            self.tensor['X'] = np.empty((0,self.size[0],self.size[1])).astype('uint8')
        # array for label y 
        self.tensor['y'] = np.empty((0))
        # loop 
        for i in range(len(self.path)):
            data = np.load(f'{self.path[i]}')
            x = data['x']
            if self.color == True:
                self.temp_tensor  = np.zeros((x.shape[0],self.size[0],self.size[1],3)).astype('uint8')
            else: 
                self.temp_tensor = np.zeros((x.shape[0],self.size[0],self.size[1])).astype('uint8')
                # inner loop for resizing the images 
            for j in range(x.shape[0]):
                new_img = cv2.resize(x[j], (self.size[1],self.size[0])) # (width, height)
                self.temp_tensor[j] = new_img
            # save the resized images 
            self.tensor['X'] = np.concatenate((self.tensor['X'],self.temp_tensor),axis=0)
            self.tensor['y'] = np.append(self.tensor['y'], np.repeat(self.label[i], x.shape[0], axis = 0))
        # create the dataset
        if self.split == True:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tensor['X'], self.tensor['y'], test_size=self.perc, random_state=123)
            np.savez('dataset.npz', X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)
        else: 
            np.savez('datasetall.npz', x = self.tensor['X'], y = self.tensor['y'])
