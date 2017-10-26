'''
CNN helper functions (some originally defined by Hvass-Labs) customised for MNIST and USPS.
Author: @ysbecca

'''


import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import cv2
from sklearn.utils import shuffle
import random
import csv

def split_by_threshold(threshold, train_preds, silent=False):
    ''' Returns the indices of images with predictions above the given softmax threshold. '''
    max_preds = np.argmax(train_preds, axis=1) # Indices of the winning prediction
    selected = np.zeros((len(train_preds)))
    for i, pred in enumerate(max_preds):
        if train_preds[i][pred] >= threshold:
            selected[i] = 1
    if not silent:
        print("Found " + str(np.count_nonzero(selected)) + " images labeled with confidence >= " + str(threshold))
    return selected

def write_test_predictions(dataset, fname):
    ''' Pass data.test object. fname does not include .csv extension. '''
    prediction_root = "./test_predictions/"
    with open(prediction_root + fname + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(dataset.cls)):
            row = [cls_prob[i], dataset.cls[i], dataset.ids[i], dataset.coords[i]]
            writer.writerow(row)


def new_weights(shape, w_name="w"):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=w_name)


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   w_name="none",      # Name of weights for layer.
                   use_pooling=True):
    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

 
    weights = new_weights(shape=shape, w_name=w_name)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer

def flatten_layer(layer):
    ''' A convolutional layer produces an output tensor with 4 dimensions. We will add 
    fully-connected layers after the convolution layers, so we need to reduce the 4-dim tensor 
    to 2-dim which can be used as input to the fully-connected layer.
    '''
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 dropout_keep_rate=1.0,
                 use_relu=True):

    ''' This function creates a new fully-connected layer in the computational graph for TensorFlow. 
    Nothing is actually calculated here, we are just adding the mathematical formulas to the TensorFlow graph.

    It is assumed that the input is a 2-dim tensor of shape `[num_images, num_inputs]`. 
    The output is a 2-dim tensor of shape `[num_images, num_outputs]`.
    '''
    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    # TODO: Add dropout back in!!!
    # dropped_input = tf.nn.dropout(input, keep_prob=dropout_keep_rate)
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer, weights

''' Placeholder variables. '''

def set_x(img_size_flat):
    return tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

'''
The convolutional layers expect `x` to be encoded as a 4-dim tensor so we have to reshape it so its shape 
is instead `[num_images, img_height, img_width, num_channels]`. Note that `img_height == img_width == 
img_size` and `num_images` can be inferred automatically by using -1 for the size of the first dimension. 
So the reshape operation is:
'''
def set_x_image(img_size, num_channels):
    return tf.reshape(x, [-1, img_size, img_size, num_channels])

'''
Next we have the placeholder variable for the true labels associated with the images that were input 
in the placeholder variable `x`. The shape of this placeholder variable is `[None, num_classes]` which 
means it may hold an arbitrary number of labels and each label is a vector of length `num_classes`.
'''
def set_y_true(num_classes):
    return tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
def set_y_true_cls(y_true):
    return tf.argmax(y_true, dimension=1)

def plot_conv_weights(weights, sess, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = sess.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_confusion_matrix(data_cls_true, cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data_cls_true
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    num_classes = 10
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



def plot_conv_layer(img_size_flat, x, session, layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.
    
    image = image.reshape(img_size_flat)

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
