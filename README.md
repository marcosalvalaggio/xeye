# Introduction

Xeye is a package for creating dataset for computer vision applications based on inferencial results of deep learning models. The main reasons to use Xeye are:

* Create a dataset using only a laptop and its integrated camera (or alternatively an external usb camera);
* Create a dataset already structured like the [mnist](https://www.tensorflow.org/datasets/catalog/mnist);
* Create a dataset that can be used for building models with [Tensorflow](https://www.tensorflow.org/) or [Pytorch](https://pytorch.org/).

## Installation

To install the package, 

```
pip install xeye
```

## Dynamic API UI

First of all, load the module datapipe from the package:

```python
from xeye import datapipe as dp
```

then initialize the instance like this 

```python
data = dp.dataset()
```
set the parameters related to the images with the **init()** function

```python
data.init()
```
the execution of this function causes the starting of the user interface in **terminal** 


```console
--- CAMERA SETTING ---
Select the index of the camera that you want to use for creating the dataset: 1
``` 

the **init()** function arises multiple questions that set the parameters values


```console
--- IMAGE SETTINGS ---
Num. of types of images to scan: 2
Name of image type (1): keyboard
Name of image type (2): mouse
Num. of frames to shoot for every image type: 10
Single frame HEIGHT: 720
Single frame WIDTH:  720
num. of waiting time (in sec.) between every frame: 0
``` 

Precisely the questions refer to:

* **Select the index of the camera that you want to use for creating the dataset**: generally 0 for integrated camera, 1 for usb external camera.
* **Num. of types of images to scan**: answer 2 if you want to create a dataset with 2 objects (e.g. keyboard and mouse); answer with the number of objects types that you want to include in your dataset.
* **Name of image type**: insert the name for every specif object that you you want to include in the dataset. The **init** function creates a named folder for every image types to include. 
* **Num. of frames to shoot for every image type**: select the number of images you want to shoot and save them in every object folder. 
* **Single frame HEIGHT**: frame height values.
* **Single frame WIDTH**: frame width values.
* **num. of waiting time (in sec.) between every frame**: e.g 0.2 causes a waiting time of 0.2 seconds between every shoot.

After having set the parameters you can invoke the function in order to start shooting images. Datapipe module provides two different formats of images:

* Grayscale image with the **gray()** function;
* Color image with the **rgb()** function.
  
Let's produce a dataset based on rgb images with the rgb() function:

```python
data.rgb()
```
in the terminal press [b] to make photos for the image types passed to the **init()** function 

```console
--- START TAKING PHOTOS ---
Press [b] on keyboard to start data collection of image type [keyboard]
b
Press [b] on keyboard to start data collection of image type [mouse]
b
``` 

On the directory of the script, you can find the folders that contain the images produced by the **rbg()** function (e.g. keyboard folder and mouse folder). 


Images collected in the folders can be used for building dataset like the [mnist](https://www.tensorflow.org/datasets/catalog/mnist). The first approch to achive this result is calling the **compressTrainTest()** function:

```python
data.compressTrainTest()
```

that produces the following output in the terminal window 

```console
--- DATASET SETTING ---
percentage of images in the test set: 0.2
``` 

in which you can select the portion of images to use in the train set and in the test set (write a value between (0,1)). By doing so, the function produces a **.npz** file formed by these specific tensors:

* Train set:
  * **X_train**: matrices/tensors of every single images in the train set;
  * **y_train**: classes (ordinal values) associated to every single images in the train set.
* Test set:
  * **X_test**: matrices/tensors of every single images in the test set;
  * **y_test**: classes (ordinal values) associated to every single images in the test set.
  
(matrices for grayscale images: [Height$\times$Width$\times$1], tensors for rgb images:[Height$\times$Width$\times$3]).

An alternative approch is to use the function **compressAll()**

```python
data.compressAll()
```

in which the images is grouped in a unique tensor that containes all the frames produced previously. 

* Unique tensor:
  * **X**: matricies/tensors of every single images produced;
  * **y**: classes (ordinal values) associated to every single images produced.

### Other useful functions

* **preview**: open camera stream to check the framing. 
  ![](img/1.png)
* **varControl**: print the values of the parameters set with the init function. 
```console
--- PARAMETERS CONTROL ---
camera index: 1
num. of images types: 2
labels of images types: ['keyboard', 'mouse']
num. of images for types: 20
Single frame HEIGHT: 720
Single frame WIDTH:  720
waiting time between frames: 0.0
percentage of images in train dataset: 0.2
statusGray: 1
statusRGB: 0
``` 

#### Xeye script example 

Example of script in order to use the **dataset()** class:

```python
from xeye import datapipe as dp
data = dp.dataset()
data.init()
data.preview()
data.rgb()
data.compressTrainTest()
data.varControl()
```

## Static API 
A faster way to use the datapipe module is represented by **dataset2()** class. In this case there isn't a terminal UI that guide you in the construction of the dataset. With dataset2 you only pass the parameters to the class, and call the functions you need. 

```python
# Load datapipe module
from xeye import datapipe as dp

# define parameters values
index = 0
img_types = 2
label = ['keyboard', 'mouse']
num = 20
height = 100
width = 100
standby_time = 0
# percentage of images in the test set 
perc = 0.2

data = dp.dataset2(index = index, img_types = img_types, label = label, num = num, height = height, width = width, stand_by_time = standby_time, perc = perc)
data.init()
data.rgb()
data.compressTrainTest()
```

The parameters passed to the class dataset2:

* **index**: generally 0 for integrated camera, 1 for usb external camera.
* **img_types**: numbers of objects types that you want to include in your dataset.
* **label**: list of names for every specif object that you you want to include in the dataset. The **init** function create a named folder for every images types to include.
* **num**: number of images you want to shoot and save in every object folder that compose the dataset. 
* **height**: frame height values.
* **width**: frame width values.
* **standby_time**: e.g 0.2 cause a waiting time of 0.2 seconds between every shoot.
* **perc**: portion of images to use in the test set (write a value between (0,1)).

