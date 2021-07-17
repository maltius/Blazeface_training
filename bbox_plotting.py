# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 13:52:29 2021

@author: altius
"""

# draw prior boxes
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

priors=bbox_utils.convert_xywh_to_bboxes(prior_boxes)

np_prior = priors.numpy()


np_prior = prior_boxes.numpy()

blank_image = np.ones(shape=[1000,1000,3], dtype=np.uint8)*255

out = cv2.VideoWriter('prior_boxes_own_aic.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 5, (1000,1000))


for t in range(np_prior.shape[0]-1,0,-1):
    box = (np_prior[t,:]*1000).astype(int)
    box=box-1
    blank_image = np.ones(shape=[1000,1000,3], dtype=np.uint8)*255

    # blank_image[box[1]:box[3],box[0]]=blank_image[box[1]:box[3],box[0]]-100
    # blank_image[box[1]:box[3],box[2]]=blank_image[box[1]:box[3],box[2]]-100
    # blank_image[box[1],box[0]:box[2]]=blank_image[box[1],box[0]:box[2]]-100
    # blank_image[box[3],box[0]:box[2]]=blank_image[box[3],box[0]:box[2]]-100
    
    blank_image=cv2.rectangle( blank_image,(box[0],box[1]),(box[2],box[3]),(0,255,0),3)

    out.write(blank_image)

out.release()
    
    # plt.imshow((blank_image))
    # plt.show()
    # time.sleep(0.25)
    
# plt.imsave('bboxes.jpeg',blank_image)
