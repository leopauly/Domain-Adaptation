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

	print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))
	return mnist_datasets, usps_datasets


