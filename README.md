# DetectLaneLines
A couple of approaches for detecting lane lines using Python wrapped OpenCV and scikit
(Based on project 1 for Udacity Self Driving Car class)
***
### In this project:
- identify lane lines on the road.  
- develop  pipeline on a series of individual images, and to a video stream. 
- extrapolate the line segments you've detected to map out the full extent of the lane lines. 
- draw just one line for the left side of the lane, and one for the right.


## Extrapolating detected lane lines
Issue: Edge detection generates spurious edges 
NOTE: Please read conclusion comments section for a better explanation. Road craks, reflectors, vehicle ahead, etc genrate, pixelwise, valid lines. Such artifacts are sources of noise, w.r.t line detection.


## Dealing with 2D sources of error
Simple fitting (LS) assumes noise only on the dependent variable (Y). In reality, noise sources from both dimensions (X, Y) - spatial perception. 

1 - Total Least Squares (TLS) fit

2 - RANSAC: fit lines using probabilistic sampling


### How to deal with individual lane lines: Split ROI into left and right ROI
Since the left and right lanes will be to the left and to the right, split the ROI into left and right


### High frquency spatio-temporal noise
1 - Increase the thresholds of the edge generation to ignore faint line patterns from asphalt

2 - Temporal rolling average filter to smoothen these fluctuations in time




## Comments Extrapolating detected lane lines

### Issue: Edge detection generates spurious edges 
`Road craks, reflectors, vehicle ahead, etc genrate, pixelwise, valid lines. Such artifacts are sources of noise, w.r.t line detection.`

### Dealing with 2D sources of error
Simple fitting (LS) assumes noise only on the dependent variable (Y). In reality, noise sources from both dimensions (X, Y) - spatial perception. Moreover, using only the points returned from the cv.Hough() funtion, we get 2 points for each line. Thus, valid "long" lines carry the same weight as small suprious lines (each line is NOT as important). 

We need to let the majorty of points which - hopefully - belong to the lane lines be used for fitting. To accomplish this, I tried two methods:

1 - Total Least Squares (TLS): fit lines directly on the edge points (or otherwise known as orthogonal fitting)

2 - RANSAC: fit lines using probabilistic sampling of shape from data. This method has shown to be fairly robust to random noise, while it picks out geometrical shapes buried in such noise. It can be not as efficient as TLS (computationally) but with today's computers, we can perform magic !!

### How to deal with individual lane lines: Split ROI into left and right ROI
Since the left and right lanes will be to the left and to the right, split the ROI into left and right


### High frquency spatio-temporal noise
It seems that the challenging video has a lot of high frequency noise in its edge generation.
The error stems from spatial spurious edges (road has faint white lie-ish patterns, switching from dark asphalt to white asphalt, other spurious objects). But in time, most of the points are fairly accurate. To deal with this, 
1 - Increase the thresholds of the edge generation to ignore faint line patterns from asphalt 
2 - Temporal rolling average filter to smoothen these fluctuations in time

Issue with priming the filter: initial values of filter are 0s. So we have floating lines until filter is filled. Better filter would be a bayesian filter (Kalman etc), or perhaps handled with care at the fusion layer.

Too high of a filterorder would not be as responsive to fast changing directions.
Issue with high thresholds for edge genration: too high of a threshold would cause performance to suffer on roads with fainting lane paint

