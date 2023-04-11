.. Xeye documentation master file, created by
   sphinx-quickstart on Tue Apr 11 17:17:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Xeye
====

Xeye is a package for data collection to build computer vision applications based on inferential results of deep learning models. The main reasons to use Xeye are:

* Create a dataset using only a laptop and its integrated camera (or alternatively an external USB camera).
* Create a dataset already structured like the `mnist <https://www.tensorflow.org/datasets/catalog/mnist>`_.
* Create a dataset that can be used for building models with `Tensorflow <https://www.tensorflow.org/>`_ or `Pytorch <https://pytorch.org/>`_.


Installation
------------

To install the package, 

``pip install xeye``


Xeye datasets for deep learning 
-------------------------------

In the `xeye-notebooks <https://github.com/marcosalvalaggio/xeye-notebooks>`_ repository, you can find examples of deep learning model implementations based on datasets produced by the Xeye package (made with `Tensorflow lib. <https://github.com/marcosalvalaggio/xeye-notebooks/tree/main/tensorflow>`_ or `Pytorch lib. <https://github.com/marcosalvalaggio/xeye-notebooks/tree/main/pytorch>`_).


* `Binary dataset <https://drive.google.com/drive/folders/1qvoFa4SRWirXj7kdWhhcqrQ8mTIHpkuJ?usp=sharing>`_: containing two types of grayscale images (with labels: 0=keyboard, 1=mouse).
* `MultiLabel dataset <https://drive.google.com/drive/folders/1qvoFa4SRWirXj7kdWhhcqrQ8mTIHpkuJ?usp=sharing>`_: containing three types of rgb images (three types of security cameras with labels: 0=dome, 1=bullet, 2=cube)

Additionally, the `xeye-notebooks <https://github.com/marcosalvalaggio/xeye-notebooks>`_ repository contains examples of scripts that use the Xeye package to build datasets (`examples link <https://github.com/marcosalvalaggio/xeye-notebooks/tree/main/xeye-example>`_).

Xeye functionalities
--------------------

The Xeye package includes three major approaches (classes) for creating a dataset from scratch: Dataset, FastDataset, and ManualDataset.

* **Dataset**: Uses the full UI terminal interface.
* **FastDataset**: Uses the constructor with all the specifications of the dataset.
* **ManualDataset**: Same as FastDataset, but every image is shot manually one at a time.
  
Additionally, the package provides a method for combining datasets created with the **BuildDataset** class.

Create a dataset with full terminal UI (Dataset)
------------------------------------------------

First of all, load the module datapipe from the package:

.. code-block:: python
   
   import xeye

then initialize the instance like this 

.. code-block:: python
   
   data = xeye.Dataset()

set the parameters related to the images with the **setup** method

.. code-block:: python
   
   data.setup()

the execution of this function causes the starting of the user interface in the **terminal**

.. code-block:: console
   
   --- CAMERA SETTING ---
   Select the index of the camera that you want to use for creating the dataset: 1

the **setup** function arises multiple questions that set the parameters' values

.. code-block:: console
   
   --- IMAGE SETTINGS ---
   Num. of types of images to scan: 2
   Name of image type (1): keyboard
   Name of image type (2): mouse
   Num. of frames to shoot for every image type: 10
   Single frame HEIGHT: 720
   Single frame WIDTH:  720
   num. of waiting time (in sec.) between every frame: 0

Precisely the questions refer to:

* **Select the index of the camera that you want to use for creating the dataset**: generally 0 for integrated camera, 1 for USB external camera.
* **Num. of types of images to scan**: answer 2 if you want to create a dataset with two objects (e.g. keyboard and mouse). In general, answer with the number of object types to include in your dataset.
* **Name of image type**: insert the name of every specific object you want to include in the dataset. The **init** function creates a named folder for every image type. 
* **Num. of frames to shoot for every image type**: select the number of images you want to shoot and save them in every object folder. 
* **Single frame HEIGHT**: frame height values.
* **Single frame WIDTH**: frame width values.
* **Num. of waiting time (in sec.) between every frame**: e.g 0.2 causes a waiting time of 0.2 seconds between every shoot.

After the parameters setting, you can invoke the function to start shooting images. Datapipe module provides two different formats of images:

* Grayscale image with the **gray** function;
* Color image with the **rgb** function.

Let's produce a dataset based on RGB images with the **rgb** function:

.. code-block:: python

   data.rgb()

In the terminal keypress [b], to make photos for the image types passed to the **setup** function.

.. code-block:: console

   --- START TAKING PHOTOS ---
   Press [b] on the keyboard to start data collection of image type [keyboard]
   b
   Press [b] on the keyboard to start data collection of image type [mouse]
   b

On the directory of the script, you can find the folders that contain the images produced by the **rbg** function (e.g. keyboard folder and mouse folder). 

Images collected in the folders can be used for building datasets like the `mnist`_:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
