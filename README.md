# Introduction

Xeye is a package for create dataset for computer vision application based on inferencial results of deep learning models. Xeye is born for these main reasons:

* Create dataset using only a laptop and its integrated camera (or alternatively an external usb camera);
* Create dataset already structured like the [mnist](https://www.tensorflow.org/datasets/catalog/mnist);
* Create dataset that can be use for build models with [Tensorflow](https://www.tensorflow.org/) or [Pytorch](https://pytorch.org/).

## Installation

To install the current release, 

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
set the parameters related to the images with the **init** function

```python
data.init()
```
the execution of this function cause the starting of the user interface in terminal

![](img/1.png)

the **init** function arise multiple questions that set the parameters values

![](img/2.png)