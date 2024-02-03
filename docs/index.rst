.. Xeye documentation master file, created by
   sphinx-quickstart on Tue Apr 11 17:17:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Xeye
====

Xeye is a package for data collection to build computer vision applications based on inferential results of deep learning models. The main reasons to use Xeye are:

*  Create a dataset using either a laptop with its integrated camera, a USB camera, or by utilizing an RTSP stream.
* Create a dataset already structured like the `mnist <https://www.tensorflow.org/datasets/catalog/mnist>`_.
* Create a dataset that can be used for building models with `Tensorflow <https://www.tensorflow.org/>`_ or `Pytorch <https://pytorch.org/>`_.


Installation
------------

To install the package, 

.. code-block:: console
   
   pip install xeye


Functionalities
---------------

The Xeye package includes two major approaches for creating a dataset from scratch: Dataset, and ManualDataset.

* **Dataset**: Uses the constructor with all the specifications of the dataset.
* **ManualDataset**: Same as Dataset, but every image is shot manually one at a time.
  
Additionally, the package provides a method for combining datasets created with the **BuildDataset** class.


Datasets for deep learning 
--------------------------

In the `examples <https://github.com/marcosalvalaggio/xeye/tree/main/examples>`_ folder of the repository, you can find examples of deep learning model implementations based on datasets produced by the Xeye package (made with `Tensorflow`_ or `Pytorch`_).

* `Binary dataset <https://drive.google.com/drive/folders/1qvoFa4SRWirXj7kdWhhcqrQ8mTIHpkuJ?usp=sharing>`_: containing two types of grayscale images (with labels: 0=keyboard, 1=mouse).
* `MultiLabel dataset <https://drive.google.com/drive/folders/1qvoFa4SRWirXj7kdWhhcqrQ8mTIHpkuJ?usp=sharing>`_: containing three types of rgb images (three types of security cameras with labels: 0=dome, 1=bullet, 2=cube)

Additionally, the `examples <https://github.com/marcosalvalaggio/xeye/tree/main/examples>`_ folder of the repository contains scripts that use the Xeye package to build datasets (`link <https://github.com/marcosalvalaggio/xeye/tree/main/examples/xeye-example>`_).


Examples
--------

Hikvision Device
~~~~~~~~~~~~~~~~

With *xeye*, you can build a dataset using all `Hikvision <https://www.hikvision.com/en/>`_ IP cameras updated to the ISAPI firmware.

.. code-block:: python
   
   from xeye import Dataset

   data = Dataset(source='rtsp://admin:password@ip:port/ISAPI/Streaming/channels/101', 
                  img_types=2, label=['a', 'b'], num=10, height=100, width=100, stand_by_time=0)
   data.preview()
   data.gray()
   data.compress_train_test(perc=0.2)
   data.compress_all()
   data.just_compress(name="batch_test")


The RTSP stream uses, by default, port 554. If you change it in the device configuration, you need to use the port specified for the RTSP connection.

* channels/101 = main stream.
* channels/102 = sub stream.

If you want to obtain the RTSP stream from a camera connected to an Hikvision NVR, the number at the end of the RTSP stream indicates:

* channels/101 = main stream of the first camera.
* channels/102 = sub stream of the first camera.
* channels/201 = main stream of the second camera.
* channels/202 = sub stream of the second camera.

Integrated or USB connected camera
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is an example of the minimal amount of code to create a dataset using *xeye* with the laptop's integrated camera:

.. code-block:: python

   from xeye import Dataset

   data = Dataset(source=0, img_types=2, label=['a', 'b'], num=10, height=100, width=100, stand_by_time=0)
   data.preview()
   data.rgb()
   data.compress_train_test(perc=0.2)
   data.compress_all()
   data.just_compress(name="batch_test")

* source = 0 -> integrated camera.
* source = 1 -> USB connected camera.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
