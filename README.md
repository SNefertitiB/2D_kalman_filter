# 2D_kalman_filter
Kalman filter for object tracking

## How it works
### Object detection Steps
1) Parse video into frames<sup>[1]</sup>

2) Detect plates in each frame<sump>[2]</sup>

3) Add bounding box<sup>[3],[4]</sup> for each plate


### Prediction
1) Use model to get bounding box for each plate

2) get centroid from bounding box

3) compare each detection to know plates

    if detection matches known plate, use it to do the kalman update

    if detection does not match know plate, create new plate

4) predict location for any know plates that were not detected


### Kalman Filter
[single object](https://machinelearningspace.com/object-tracking-python/)

[multi object](https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/)
#### Starting State
We are using yolov5 as the object detection model for measurements.

The model returns two coordinates $(x_1, y_1)$ and $(x_2, y_2)$ that describe a rectangular 
bounding box around around the plate. For the Kalman filter, we will need the centroid of 
each box. Since the bounding box is a rectangle, we can use the following midpoint formula to 
find the centroid:

centroid coordinates = $ (\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2})$ 

#### Velocity
At each measurement, we update the velocity of the object as well as the predicted location.

average velocity: $\bar_v = $

current time: $t$

displacement between measurements: $\Delta z = z_t - z_{t-1}$ 

change in time: $\Delta t = t - (t - 1)$ 

$\bar_v = \frac{\Delta z}{\Delta t}$
 
#### 2 Dimensional Kalman filter with variable velocity
 
Step 1: Prediction
 - prior.x = prior.x + velocity_x
 - prior.y = prior.y + velocity_y

Step 2: Update (if the object is detected)
 - z = center of bounding box
 - velocity update
   - velocity_x = z.x - prev.x
   - velocity_y = z.y - prev.y
 - prev update
   - prev = z
 - prior update
   - new_x = (prior.x + z.x) / 2
   - new_y = (self.prior.y + z.y) / 2)
   - prior = (new_x, new_y)


### Multi Object Tracking
To track multiple objects you need to: 

1) correctly assign each measurement to a tracker

2) use the distance formula to figure out which measurement is closest to the pre-update prediction

3) use that measurement to update the prediction

4) if distance to the closest prediction is greater than some threshold, don't assign the measurement to any tracker
    
   note: any trackers that don't receive a measurement for a given frame will skip the update step of the Kalman filter

5) for any unassigned measurements, initialize a new tracker object 

## Future updates
1) get rid of duplicate measurements
2) use distance formula to find the nearest measurement for known plates
3) implement a confidence measurement for each plate location
4) discard known plate after confidence drops below some threshold

[1]: https://www.google.com/search?q=play+mp4+in+colab&rlz=1C5CHFA_enUS904US904&source=lnms&tbm=vid&sa=X&ved=2ahUKEwjd4fiIlKD7AhUoLFkFHa_aAWEQ_AUoAXoECAIQAw&biw=1332&bih=592&dpr=1#fpstate=ive&vld=cid:5e2ea0c6,vid:o3h6ptvCBYk
[2]: https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
[3]: https://pytorch.org/vision/stable/generated/torchvision.utils.draw_bounding_boxes.html
[4]: https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html#sphx-glr-auto-examples-plot-repurposing-annotations-py
