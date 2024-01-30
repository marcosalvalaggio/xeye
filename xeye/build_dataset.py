import cv2 
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple


class BuildDataset:
    """
    Builds a dataset by merging multiple datasets with the given parameters.

    Attributes:
        path (List[str]): List of paths to the numpy files containing the datasets.
        label (List[int]): List of labels corresponding to each dataset.
        size (tuple): Tuple specifying the size of the images in the dataset. Defaults to None.
        color (bool): Whether the images are in color or grayscale. Defaults to True.
        split (bool): Whether to split the dataset into train and test sets. Defaults to True.
        perc (float): The percentage of data to use for the test set. Defaults to 0.1.
    
    Examples:
        >>> import xeye
        >>> # list of directory (paths for the .npz files)
        >>> path = ['batch_1.npz','batch_2.npz', 'batch_3.npz']
        >>> # list of labels associated with the images inside the .npz files
        >>> label = [0,1,2]
        >>> data = xeye.BuildDataset(path=path, label=label, size = None, color=True, split=True, perc=0.2)
        >>> data.build()
    """
    def __init__(self, path: List[str], label: List[int], size: Tuple = None, color: bool = True, split: bool = True, perc: float = 0.1) -> None:
        self.path = path
        self.label = label 
        self.size = size
        self.color = color 
        self.split = split
        self.perc = perc
        self.height = 0
        self.width = 0 
        self._tensor = {}
        self._temp_tensor = []
    

    def _control(self) -> None:
        """
        Checks if the datasets to merge have the same colour space.

        Raises:
            ValueError: If the datasets have different color spaces.

        Returns:
            None
        
        Notes:
            * This method extracts the height, width, and color channels of each image in the dataset using numpy.
            * It then checks if the images have the same color space by comparing the color channel values for each image.
            * If the sizes of the images in the dataset are not specified in the instance variables, it sets the maximum height and width of all the images as the dataset size.
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


    def build(self) -> None:
        """
        Builds a new dataset by merging the datasets with the parameters indicated by the instance variables.

        Raises:
            ValueError: If the path and label lists do not have the same length.
            ValueError: If the datasets being merged have different color spaces.
        
        Returns:
            None

        Note:
            The method calls the `_control` method to check if the datasets being merged have the same color space. The resulting merged dataset is stored as a numpy array in `_tensor['X']` and `_tensor['y']`. If the `split` instance variable is set to `True`, the merged dataset is split into training and testing sets using the `train_test_split` method from scikit-learn and saved as a numpy array in the file 'dataset.npz'. Otherwise, the merged dataset is saved as a numpy array in the file 'datasetall.npz'.
        """
        # control method calling 
        self._control()
        # control if the path and label lists have the same length
        if len(self.path) != len(self.label):
            raise ValueError("Path and label lists doesn't have the same length...")
        # Create the tensor X
        if self.color == True:
            self._tensor['X'] = np.empty((0,self.size[0],self.size[1],3)).astype('uint8')
        else: 
            self._tensor['X'] = np.empty((0,self.size[0],self.size[1])).astype('uint8')
        # array for label y 
        self._tensor['y'] = np.empty((0))
        # loop 
        for i in range(len(self.path)):
            data = np.load(f'{self.path[i]}')
            x = data['x']
            if self.color == True:
                self._temp_tensor  = np.zeros((x.shape[0],self.size[0],self.size[1],3)).astype('uint8')
            else: 
                self._temp_tensor = np.zeros((x.shape[0],self.size[0],self.size[1])).astype('uint8')
                # inner loop for resizing the images 
            for j in range(x.shape[0]):
                new_img = cv2.resize(x[j], (self.size[1],self.size[0])) # (width, height)
                self._temp_tensor[j] = new_img
            # save the resized images 
            self._tensor['X'] = np.concatenate((self._tensor['X'],self._temp_tensor),axis=0)
            self._tensor['y'] = np.append(self._tensor['y'], np.repeat(self.label[i], x.shape[0], axis = 0))
        # create the dataset
        if self.split == True:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self._tensor['X'], self._tensor['y'], test_size=self.perc, random_state=123)
            np.savez('dataset.npz', X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)
        else: 
            np.savez('datasetall.npz', x = self._tensor['X'], y = self._tensor['y'])
