### Problem Statement
Detect object tagging in the video and examine how parallel object detection on multiple patches can allow detection of smaller objects in the overall image without decreasing the resolution.

Object detection and segmentation methods are one of the most challenging problems in computer vision which aim to identify all target objects and determine the categories and position information. Numerous approaches have been proposed to solve this problem, mainly inspired by methods of computer vision and deep learning. In this project, we aim to build a model which detects multiple objects and segmentation in a moving video. For eg. Image tagging, lane detection, drivable area segmentation, road object detection, semantic segmentation, instance segmentation, multi-object detection tracking, multi-object segmentation tracking, domain adaptation, and imitation learning.

#### Definition of Terms
###### Image Tagging
Image tagging involves labelling image. So in any given image, we should be able to tag the different objects in an image.
Questions: 
1. Does this involve bounding the objects in image in a box?

<u>Algorithm to use: Yolo v3</u>


###### Lane Detection
Lane detection involves detecting the part of the road on which it is safe to drive the vehicle. Lane detection is dependent on the lane markings on the road.

On Indian roads, the lane marking are not necessarily present. So the lane detection cannot depend upon lane marking,but rather a sense of what a lane should be. So the algorithm has to figure out the width of road on which it is safe to drive the vehicle. <u>Solving lane detection on Indian roads will be out of scope of this project.</u>

<u>Algorith to use: ?</u>

###### Drivable area segmentation
Drivable area segmentation involves, detecting the surface on which vehicle can be driven. This will include all lanes.

Questions:
1. Will this include road which is drivable, but would have traffic coming from opposite direction?
2. In cases where physical divider is absent then will it include entire road?

<u>Algorith to use: U-Net (Transpose Convulation)</u>

###### Road Object detection
Road object detection is similar to image tagging with addition of bounding box around the object.

Questions:
1. How many objects should we target to detect in this project?
   Example: Car, Truck, Person, Motorcycle, Cycle, Animals, Signals (traffic lights), pot holes, roadbumps,bridges, traffic policeman
2. 

<u>Algorith to use: Yolo v3</u>

###### Semantic Segmentation

###### Instance Segmentation

###### Multi-Object Detection Tracking

###### Multi Object segmentation and Tracking

###### Domain Adaptation

###### Imitation Learning

### Task Breakdown