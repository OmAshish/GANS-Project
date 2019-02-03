# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:12:02 2019

@author: OM MISHRA
"""

#Importing the libraries
import torch  # Nuetral net, Back Prop
from torch.autograd import Variable # gradient descent dependent
import cv2 # drawing rectangles
from data import BaseTransform, VOC_CLASSES as labelmap # image will fit to the grid by baseTransform VOC classes classify 
from ssd import build_ssd # constructor of ssd nueral network
import imageio # process images and apply ssd on the images

# Defining a function that will do the detections
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1)
    x.unsqueeze(0) #batch sequence 3rd transformation
    x = Variable(x.unsqueeze(0))
    y = net(x) # feed to nuetral network frame by frame detection
    detections = y.data # extracted information
    # normalization through scale between 0 and 1
    scale = torch.Tensor([width, height, width, height])
    # [batch, number of classes, number of occurance, (score, x0, Y0, x1, Y1)] 
    # score is lower 0.6 not found and if more then found
    for i in range(detections.size(1)): # i occurance of a class
        j = 0
        while detections[0,i,j,0] >= 0.6:
            pt = (detections[0, i, j, 1:]* scale).numpy() # coordinates to normalize
            # rectangle works with numpy array
            cv2.rectangle(frame, (int(pt[0]),int(pt[1]),int(pt[2]),int(pt[3])),(255,0,0), 2) # coordinate of the upper right corner and the the other two are the coordinate of the lower right corner
            # Then we have color red and then the thickness of the color displayed that is 2
            cv2.putText(frame, labelmap[i - 1],(int(pt[0])), (int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),2, cv2.LINE_AA )
            # labelmap() classifies the name of the classes with labels, then comes position, FONT_Hershey is font style
            j += 1
    return frame

# Creating the SSD neural network
net = build_ssd('test')
# train phase and test phase
net.load_state_dict(torch.load('ssd300_map_77.43_v2.pth',map_location = lambda storage, loc: storage))

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Doing some Object Detection on a video
reader = imageio.get_reader('funny_dog.mp4') # fps = frames per seconds
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = fps) # add sequence of frames
for i, frame in enumerate(reader):
    # apply detect method
    frame = detect(frame, net.eval(), transform) # detections from each frame
    writer.append_data(frame)
    print(i) # no of processed frame
writer.close()



