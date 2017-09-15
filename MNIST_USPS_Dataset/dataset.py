'''
Dataset class (defined by Hvass-Labs) customised for MNIST and USPS.
Data preparation code by @leopauly

Author: @ysbecca

'''
import time
import mnist_usps as mnus
from datetime import timedelta
import numpy as np

class DataSet(object):

  def __init__(self, images, cls, set_ids):
    """ set_id = the dataset which the images belong to. 0 = MNIST; 1 = USPS """

    self._num_images = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    # Convert from [0, 255] -> [0.0, 1.0].

    # images = images.astype(np.uint8)
    # images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._cls = cls
    self._set_ids = set_ids

    # Set the labels based on cls. 
    labels = np.zeros((self._num_images, 10))
    for i, cls_ in enumerate(cls):
    	labels[i][cls_] = 1
    self._labels = labels

    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def cls(self):
    return self._cls

  @property
  def labels(self):
    return self._labels

  @property
  def set_ids(self):
    return self._set_ids

  @property
  def num_images(self):
    return self._num_images

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def set_images(self, images):
    self._images = images
    

  def remove_from_set(self, selected):
  	''' Removes the images at selected indices from the dataset. '''
  	if(len(selected) != self._num_images):
  		print("ERROR: number of selected indices does not match dataset size.")
  		return
  	num_removed = np.count_nonzero(selected)

  	# Building new arrays instead of altering old ones for efficiency.
  	new_images, new_cls, new_set_ids, new_labels = [], [], [], []
  	for i, s in enumerate(selected):
  		if s == 0:
  			new_images.append(self._images[i])
  			new_cls.append(self._cls[i])
  			new_set_ids.append(self._set_ids[i])
  			new_labels.append(self._labels[i])


  	self._images = np.array(new_images)
  	self._cls = np.array(new_cls)
  	self._set_ids = np.array(new_set_ids)
  	self._labels = np.array(new_labels)
  	self._num_images -= num_removed

  def add_to_set(self, selected, dataset, preds):
  	''' Adds the images at the selected indices to the dataset and updates the params.'''

  	num_added = np.count_nonzero(selected)
  	max_preds = np.argmax(preds, axis=1) # Indices of the winning prediction (preds[max_preds[i]])

  	images_, cls_, set_ids_, labels_ = [], [], [], []
  	for i, s in enumerate(selected):
  		if s > 0:
  			images_.append(dataset.images[i])
  			cls_.append(max_preds[i]) # Add class as PREDICTED by the CNN
  			set_ids_.append(dataset.set_ids[i])
  			new_label = np.zeros(10)
  			new_label[max_preds[i]] = 1
  			labels_.append(new_label)

  	# Add all the data.
  	self._images = np.concatenate((self._images, images_))
  	self._cls = np.concatenate((self._cls, cls_))
  	self._set_ids = np.concatenate((self._set_ids, set_ids_))
  	self._labels = np.concatenate((self._labels, labels_))
  	self._num_images += num_added
  	
  	# Reshuffle everything the same way.
  	perm = np.arange(self._num_images)
  	np.random.shuffle(perm)
  	self._images = self._images[perm]
  	self._cls = self._cls[perm]
  	self._set_ids = self._set_ids[perm]
  	self._labels = self._labels[perm]

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_images:
      # Finished epoch
      self._epochs_completed += 1

      # # Shuffle the data (maybe)
      # perm = np.arange(self._num_images)
      # np.random.shuffle(perm)
      # self._images = self._images[perm]
      # self._labels = self._labels[perm]
      # Start next epoch

      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_images
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end] # self._cls[start:end], self._set_ids[start:end]

def generate_combined_dataset(mnist_dataset, usps_dataset):
	class DataSets(object):
		pass
	dataset = DataSets()

	# Complete MNIST
	dataset.train = DataSet(np.concatenate( \
		(mnist_dataset.train.images, mnist_dataset.valid.images, mnist_dataset.test.images)), \
		np.concatenate((mnist_dataset.train.cls, mnist_dataset.valid.cls, mnist_dataset.test.cls)), \
		np.concatenate((mnist_dataset.train.set_ids, mnist_dataset.valid.set_ids, mnist_dataset.test.set_ids)), \
		)
	dataset.test = DataSet(usps_dataset.test.images, usps_dataset.test.cls, usps_dataset.test.set_ids)

	# Only used to keep track of remaining USPS images not yet added in bootstrap iterations.
	dataset.usps_train = DataSet(usps_dataset.train.images, usps_dataset.train.cls, usps_dataset.train.set_ids)
	return dataset

def read_datasets():
	class DataSets(object):
		pass
	mnist_datasets = DataSets()
	usps_datasets = DataSets()

	start_time = time.time()

	# mnist_x[i]; 0 = train, 1 = valid, 2 = test
	mnist_x, usps_x, mnist_y, usps_y = mnus.dataset(normalisation=True, store=False)

	mnist_datasets.train = DataSet(mnist_x[0], mnist_y[0], np.zeros((len(mnist_y[0]))))
	mnist_datasets.valid = DataSet(mnist_x[1], mnist_y[1], np.zeros((len(mnist_y[1]))))
	mnist_datasets.test = DataSet(mnist_x[2], mnist_y[2], np.zeros((len(mnist_y[2]))))

	# For now, the USPS train includes the validation set, so:
	# usps_x[i]; 0 = train, 1 = test
	usps_datasets.train = DataSet(usps_x[0], usps_y[0], np.ones((len(usps_y[0]))))
	usps_datasets.test = DataSet(usps_x[1], usps_y[1], np.ones((len(usps_y[1]))))


	end_time = time.time()
	time_dif = end_time - start_time

	# print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))
	return mnist_datasets, usps_datasets


