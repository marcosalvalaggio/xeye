 
## Xeye

<h1 align="center">
<img src="logo.png" width="200">
</h1><br>


[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://xeye.readthedocs.io/en/latest/index.html) [![PyPI version](https://badge.fury.io/py/xeye.svg)](https://badge.fury.io/py/xeye) ![PyPI - Downloads](https://img.shields.io/pypi/dm/xeye) 
  

Xeye is a package for data collection to build computer vision applications based on inferential results of deep learning models. The main reasons to use Xeye are:

* Create a dataset using either a laptop with its integrated camera, a USB camera, or by utilizing an RTSP stream.;
* Create a dataset already structured like the [mnist](https://www.tensorflow.org/datasets/catalog/mnist);
* Create a dataset that can be used for building models with [Tensorflow](https://www.tensorflow.org/) or [Pytorch](https://pytorch.org/).

<hr>

## Installation

To install the package, 

```
pip install xeye
```

<hr>

## Xeye functionalities

The Xeye package includes two major approaches for creating a dataset from scratch: Dataset, and ManualDataset.

* **FastDataset**: Uses the constructor with all the specifications of the dataset.
* **ManualDataset**: Same as FastDataset, but every image is shot manually one at a time.
  
Additionally, the package provides a method for combining datasets created with the **BuildDataset** class.

<hr>

## Xeye datasets for deep learning 

In the [example](examples) folder, you can find examples of deep learning model implementations based on datasets produced by the Xeye package (made with [Tensorflow](https://github.com/marcosalvalaggio/xeye-notebooks/tree/main/tensorflow) or [Pytorch](https://github.com/marcosalvalaggio/xeye-notebooks/tree/main/pytorch) frameworks).


* [Binary dataset](https://drive.google.com/drive/folders/1qvoFa4SRWirXj7kdWhhcqrQ8mTIHpkuJ?usp=sharing): containing two types of grayscale images (with labels: 0=keyboard, 1=mouse).
* [MultiLabel dataset](https://drive.google.com/drive/folders/1qvoFa4SRWirXj7kdWhhcqrQ8mTIHpkuJ?usp=sharing): containing three types of rgb images (three types of security cameras with labels: 0=dome, 1=bullet, 2=cube)

Additionally, the [example](example) folder contains examples of scripts that use the Xeye package to build datasets ([examples link](examples/xeye-example).

<hr>
