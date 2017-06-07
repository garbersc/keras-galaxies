kaggle-galaxies
===============

This is the original README of the kaggle-galaxies winning solution found at  github.com/benanne/kaggle-galaxies.git by Sander Dieleman. It has not been updated to the state of this repository yet.
The solution in this repo implements the main model of the winning solution in keras (github.com/fchollet/keras)



Based on the winning solution for the Galaxy Challenge on Kaggle (http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) that is found at git://github.com/benanne/kaggle-galaxies.git .

Documentation about the method and the code is available in `doc/documentation.pdf`. Information on how to generate the solution file can also be found below.

Needed dependecies:
keras with theano backend:
      https://github.com/fchollet/keras/
      https://github.com/Theano/
pylearn2:
	git://github.com/lisa-lab/pylearn2.git

Instructions for installing Theano and getting it to run on the GPU can be found [here](http://deeplearning.net/software/theano/install.html). It should be possible to install NumPy, SciPy, scikit-image and pandas using `pip` or `easy_install`. 

### Download the training data

Download the data files from [Kaggle](http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data). Place and extract the files in the following locations:

* `data/raw/training_solutions_rev1.csv`
* `data/raw/images_train_rev1/*.jpg`
* `data/raw/images_test_rev1/*.jpg`

Note that the zip file with the training images is called `images_training_rev1.zip`, but they should go in a directory called `images_train_rev1`. This is just for consistency.


### 10 categories model

Generate solutions for 10 category model:
	 solutions_to_10cat.py
	 
Definition of 10 category model:
	 custom_keras_model_x_cat.py

Start with python for training:
 	try_convnet_keras_x_cats.py

Functions for evaluation of the 10 category:
	  predict_convnet_keras_10cat.py	
