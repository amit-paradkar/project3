### What is IDD dataset

[IDD Dataset](http://idd.insaan.iiit.ac.in/)

[IDD Dataset Paper](http://idd.insaan.iiit.ac.in/media/publications/idd-650.pdf)

##### What BDD Dataset Provides

###### Image Tagging
Images are captured in 6 different weather conditions:
1. Clear
2. Partly Cloudy
3. Over-cast
4. Rainy
5. Snowy
6. Foggy

Images are captured in different scenes:
1. Residential
2. High-way
3. City street
4. Parking lot
5. Gas station
6. Tunnel

Images are captured in different times of a day:
1. Dawn/Dusk
2. Daytime
3. Night

###### Object Detection
10 categories of objects are provided in this dataset
1. Car
2. Sign
3. Light
4. Person
5. Truck
6. Bus
7. Bike
8. Rider
9. Motor
10. Train

BDD also provides information regarding occupancy of the vehicle. And also truncation(?)

###### Lane Marking
Lane marking has 8 main cetegories
1. Road curb
2. cross walk
3. double white
4. double yellow
5. double other color
6. single white
7. single yellow
8. single other color

###### Drivable Area
Drivable areas are divided into
1. Directly drivable
   This is the areas where vehicle is currently being driven. Or it has right of way
2. Alternetely drivable
   It is a different lane on which vehicle can be driven.
   
###### Semantic Instance Segmentation
BDD provides annotations for 10K video clips. Each pixel has a label. Labels are provided for 40 objects.

###### Multiple Object Tracking
BDD dataset has a subset called Multi Object Tracking (MOT) dataset. There are 2K videos with 400K frames. Each video is approx. 5 fps. So approx. 200K frames per video.
Objects also present occlusion and reappeare. This closely resembles real life situations. There are about 50K instance of objects tracking (occlusion and reappreance).

###### Multiple Object Tracking and Segmentation
BDD contains 90 videos which shows multiple object tracking and segmentation (MOTS). 

###### Imitation Learning
The recording also show human driver action given the visual inputs and driving trajectories. This can be used to train imitation learning on driving algorithms.
