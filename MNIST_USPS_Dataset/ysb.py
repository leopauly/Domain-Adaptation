'''
@author ysbecca Rebecca Stone

Extra miscellaneous functions.

Simple functions which display images using matplotlib.pyplot (http://matplotlib.org/api/pyplot_api.html)
Any image in a format readable by matplotlib.pyplot's imshow() function is good.
'''

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

X, Y, W, H = 0, 1, 2, 3

def plot_with_legend(x_range, y_data, legend_labels, x_label, y_label):
    ''' Displays a single plot with multiple datasets and matching legends.
        x_range should be one single array of indices for the x axis, and y_data can 
    '''
    if(len(y_data) != len(legend_labels)):
        print("Error: the number of data sets does not match the number of labels provided.")
        return
    all_plots = []
    for i, data in enumerate(y_data):
        temp, = plt.plot(x_range, data, label=legend_labels[i])
        all_plots.append(temp)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=all_plots)
    plt.show()

def show_boxes_on_image(image, patches, colour):
    '''
    Displays red rectangles on top of image. Rectangles must be smaller than image size. 
    '''
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap="gray")

    # Create rectangles from the blocks and add them to the image
    for patch in patches:
        # Drawn from lower left, unlike our patches
        rect = pat.Rectangle((patch[X], patch[Y]+patch[H]),patch[W], patch[H],linewidth=3,edgecolor=colour,facecolor='none')
        ax.add_patch(rect)

    plt.show()

def show_images(images, per_row, per_column):
    ''' Displays up to per_row*per_column images with per_row images per row, per_column images per column.
	'''
    fig = plt.figure(figsize=(25, 25))
    data = images[:(per_row*per_column)]

    for i, image in enumerate(data):
        plt.subplot(per_column, per_row, i+1)
        plt.imshow(image)
        plt.axis("off")
    
    plt.show()

def show_labeled_patches(images, clss):
    fig = plt.figure(figsize=(20, 10))
    data = images[:50]
    labels = clss[:50]

    for i, image in enumerate(data):
        plt.subplot(5, 10, i+1)
        plt.imshow(image)
        plt.title(str(labels[i]))
        plt.axis("off")

    plt.show()

def show_labeled_patches_gray(images, clss):
    fig = plt.figure(figsize=(20, 10))
    data = images[:50]
    labels = clss[:50]

    for i, image in enumerate(data):
        plt.subplot(5, 10, i+1)
        plt.imshow(image, cmap="gray")
        plt.title(str(labels[i]))
        plt.axis("off")

    plt.show()


def show_patches(images):
    ''' A quick way of displaying up to 100 patches of 12x12px. You can easily change the default number
    	of patches below; remember to change the number of images per column to match.
    '''
    fig = plt.figure(figsize=(12, 12))
    data = images[:100]

    for i, image in enumerate(data):
        plt.subplot(10, 10, i+1)
        plt.imshow(image)
        plt.axis("off")

    plt.show()

def show_image_panels(images_1, images_2, image_ids, show_colourbar=False, hot=False):
    ''' Displays two images side by side for as many images are in each array. '''
    per_column = len(images_1)
    fig = plt.figure(figsize=(20, 15))

    j = 0
    for i in range(per_column*2):
        plt.subplot(per_column, 2, i+1)
        if(i % 2 == 0):
            plt.title(str(image_ids[j]))
            plt.imshow(images_1[j])
        else:
            if(hot):
                plt.imshow(images_2[j], cmap="hot", interpolation="nearest")
            else:
                plt.imshow(images_2[j], interpolation="nearest")
            j += 1
        plt.axis("off")
    if(show_colourbar==True):
        plt.colorbar()    
    plt.show()
    
def show_image(image, no_axis=False, hot=False):
    ''' Display one single image without axis markings. '''
    if(hot):
        plt.imshow(image, cmap="hot")
    else:
        plt.imshow(image)
    if no_axis:
        plt.axis("off")
    plt.show()

''' Fast cheating way of showing grayscale properly due to weird matplotlib bug. '''

def show_image_gray(image, no_axis=False):
    ''' Display one single image without axis markings. '''
    plt.imshow(image, cmap='gray')
    if no_axis:
        plt.axis("off")
    plt.show()

def show_images_gray(images, per_row, per_column):
    ''' Displays up to per_row*per_column images with per_row images per row, per_column images per column.
    '''
    fig = plt.figure(figsize=(25, 25))
    data = images[:(per_row*per_column)]

    for i, image in enumerate(data):
        plt.subplot(per_column, per_row, i+1)
        plt.imshow(image, cmap='gray')
        plt.axis("off")
    
    plt.show()
