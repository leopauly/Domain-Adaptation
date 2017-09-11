# Description: Function for Preparing datasets MNIST and USPS as numpy image arrays and storing as .png files if needed
# Author : Leo Pauly & Rebecca Stone | cnlp@leeds.ac.uk, @ysbecca

import os
import numpy as np
from numpy.random import seed
import scipy.misc as misc
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata, load_iris, load_digits

os.environ['PYTHONHASHSEED'] = '0'
seed(9)


def split_train_valid_test(images, targets, test_size=0.2, valid_size=0.1, no_valid=False):
    # Convert test and validation ratios into no. of images
    total_images = len(images)
    num_test_imgs = int(float(total_images * test_size))
    num_valid_imgs = int(float(total_images * valid_size))
    
    if(no_valid):
        num_valid_imgs = 0

    num_train_imgs = total_images - (num_test_imgs + num_valid_imgs)

    # Shuffle both images and targets randomly the same way.
    images, targets = shuffle(images, targets)
    if(no_valid):
        # [[train, test], [train, test]]
        return [[images[:num_train_imgs], images[num_train_imgs:]], [targets[:num_train_imgs], targets[num_train_imgs:]]]
    else:
       # [[train, valid, test], [train_target, valid_target, test_target]]
        return [[images[:num_train_imgs], images[num_train_imgs:(num_train_imgs + num_valid_imgs)], \
            images[-num_test_imgs:]], [targets[:num_train_imgs], \
            targets[num_train_imgs:(num_train_imgs + num_valid_imgs)], targets[-num_test_imgs:]]]


def dataset(normalisation=False,store=False,test_size=0.2, valid_size=0.1):
    # Tag: MNIST
    mnist = fetch_mldata("MNIST original")
    mnist_x=mnist.data
    
    mnist_x=np.reshape(mnist_x[:],[mnist_x.shape[0],28,28])
    mnist_x_new=np.zeros([70000,16,16])
    for i in range(mnist_x.shape[0]):
        mnist_x_new[i,:,:]=misc.imresize(mnist_x[i],[16,16])
        
    mnist_x_new = mnist_x_new.astype('float32')

    # Tag: USPS
    usps = fetch_mldata("USPS")
    usps_x=usps.data
        
    usps_x_new=np.zeros([9298,16,16])
    usps_x_new=np.reshape(usps_x[:],[usps_x.shape[0],16,16])
    usps_x_new=(usps_x_new-(-1))/2
    usps_x_new = usps_x_new.astype('float32')
    usps_x_new=usps_x_new*255

    # if store==True then the images will be written into the current folder
    if (store==True):
        for i in range (usps_x_new.shape[0]):
            print(i)
            c=str(i)
            scipy.misc.toimage(usps_x_new[i], cmin=0.0, cmax=255).save('./data/usps/usps'+c+'.png')
        print('Stored USPS data in the current directory')

        for i in range (mnist_x_new.shape[0]):
            c=str(i)
            print(i)
            scipy.misc.toimage(mnist_x_new[i], cmin=0.0, cmax=255).save('./data/mnist/mnist'+c+'.png')
        print('Stored MNIST data in the current directory')

    # Normalises data if normalisation arguement is true
    if (normalisation==True):
        mnist_x_new=mnist_x_new/255
        usps_x_new=usps_x_new/255

    # Splitting data into validation/testing and training data
    # Convert MNIST targets to integers and decrement the USPS targets to match MNIST.
    mnist_x, mnist_y = split_train_valid_test(mnist_x_new, np.array(mnist.target).astype(int), test_size=test_size, valid_size=valid_size)
    usps_x, usps_y = split_train_valid_test(usps_x_new, np.subtract(usps.target, 1), test_size=test_size, valid_size=valid_size, no_valid=True)
    
    
    return mnist_x, usps_x, mnist_y, usps_y
