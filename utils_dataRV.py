import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage
# import cv2
import time
import scipy.misc
import caffe
import seaborn as sns
from IPython import display
import time 

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def normalize_heatmap(heatmap):
    heat_map_normalize = np.zeros(heatmap.shape)
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heat_map_normalize[x,y,:] = softmax(heatmap[x,y,:])
    
    return heat_map_normalize

def do_training(solver, step_size, nb_step=0):
        solver.step(step_size)

        heat_map = solver.test_nets[0].blobs["score-final"].data[0,:,:,:].transpose(1,2,0)
        heat_map_normalize = normalize_heatmap(heat_map)
#         heat_map_normalize = heat_map
        minimum = np.min(heat_map[:,:,0])

        plt.figure(figsize=(10,10))
        image_test = solver.test_nets[0].blobs["data"].data[0].transpose(1,2,0)
        image_test_label = solver.test_nets[0].blobs["label"].data[0,0,:,:]
        plt.subplot(1,5,1)
        plt.imshow(image_test[:,:,0])
        plt.title("image test")
        plt.subplot(1,5,2)
        plt.imshow(image_test_label)
        plt.title("Label of the test image")
        plt.subplot(1,5,3)
        plt.imshow(heat_map_normalize)
        plt.title("min : " + str(minimum))
        plt.subplot(1,5,4)
        plt.imshow(solver.test_nets[0].blobs["score"].data[0,:,:,:].transpose(1,2,0))
        plt.title("score")
        plt.subplot(1,5,5)
        plt.imshow(solver.test_nets[0].blobs["score-final"].data[0,:,:,:].transpose(1,2,0).argmax(2), vmin=0, vmax=2)
        plt.title("After : " + str(nb_step+step_size) + " itterations")
        display.display(plt.gcf())
        display.clear_output(wait=True)
        time.sleep(1)
    
def display_image(img, vmin=None, vmax=None, title=''):
	plt.imshow(img, vmin=vmin, vmax=vmax)
	plt.axis('off')
	plt.title(title)
