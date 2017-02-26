# CarND-Proj04-Advanced-Lane-Lines

##Background
This repository contains my project report for the [Udacity Self-Driving Car nano-degree](https://www.udacity.com/drive) program's project 4 - Advanced Lane Finding. The original starting files and instructions can be found [here](https://github.com/udacity/CarND-Advanced-Lane-Lines).

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard_undistort/chessboard_corners_detection.png "Chessboard corners"
[image2]: ./output_images/chessboard_undistort/undistort_12.png "Undistort Example"
[image3]: ./test_images/test1.jpg "Test Image Example"
[image4]: ./output_images/test-images-1-undistort/undistorted_6.jpg "Undistorted Test Image Example"
[image5]: ./output_images/test-images-2-threshold-binary/six_edges_1.jpg "Threshold Binary Test Image Example"
[image6]: ./output_images/test-images-3-perspective-transform/warped_7.jpg "Perspective Transformation"
[image7]: ./output_images/test-images-4-lane-pixels-fit-polynomial/res-warp_wincount_0.jpg "Sliding Window Method"
[image8]: ./output_images/test-images-4-lane-pixels-fit-polynomial/res-warp_convol_0.jpg "Convolution Method"
[image9]: ./output_images/test-images-5-results/result_0.jpg "Result projected back on real image"
[video1]: ./result.mp4 "Result Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! This project is roughly divided into three sections: 
- Camera calibration and distortion correction, 
- Pipeline for single independent images,
- Additional processing for time-sequence video frames

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the standalone file `calcam.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I have used openCV's `findChessboardCorners()` function to detect these iamge plane corners (see `calcam.py` line 32). The following shows the result of the corners found:  

![alt text][image1]

For some of the images, the corner detection failed because the image is bigger than the frame and `findChessboardCorners()` could find the required number of rows and columns.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to all the test images using the `cv2.undistort()` function and this is one of the sample result: 
![alt text][image2]

For the rest of the undistorted chessboard images, please see [here](https://github.com/lowspin/CarND-Proj04-Advanced-Lane-Lines/tree/master/output_images/chessboard_undistort).

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one (see `genimages.py` line 30):
![alt text][image3]
![alt text][image4]
The first image is the original camera distorted image and the second is the result of distortion correction. As observed, the main effect is in the border regions.

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (see `genimages.py` line 37, and `imagefunction.py` lines 74-77). I also tried converting the color-space to HLS and use the S-channel for detection. Here's an example of my output for this step.

![alt text][image5]
As labeled, these subplots are respectively from top left to bottom right:
- using only x-direction of the sobel threshold detector (`imagefunctions.py` lines 6-26)
- using only y-direction of the sobel threshold detector (`imagefunctions.py` lines 6-26)
- using both x- and y-direction of the sobel detector by take the complex magnitude (`imagefunctions.py` lines 29-45)
- using the direction of the gradient (`imagefunctions.py` lines 48-61)
- combination of above methods
- using the s-channel of the HLS color-space
- final combination, with both gradients and s-channel

From these results, the final combined results seems to include the most relevant points for lane line detection. Hence, I have decided to use this combination for the rest of the project. (Note: see the result for the other test iamges [here](https://github.com/lowspin/CarND-Proj04-Advanced-Lane-Lines/tree/master/output_images/test-images-2-threshold-binary)).

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspectiveTransform()`, which appears in lines 121 through 155 in the file `imagefunctions.py`.  The `perspectiveTransform()` function takes as inputs two undistorted image (RGB image `undst_rgb` and thresholded binary image`img_bin`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
    half_trap_top = 62
    half_trap_bot = 448
    top_left_src = [640-half_trap_top, 460]
    top_right_src = [640+half_trap_top, 460]
    bottom_right_src = [640+half_trap_bot, 720]
    bottom_left_src = [640-half_trap_bot, 720]

    top_left_dst = [320, 0]
    top_right_dst = [960, 0]
    bottom_right_dst = [960, 720]
    bottom_left_dst = [320,720]

    src = np.float32( [ top_left_src, top_right_src, bottom_right_src, bottom_left_src ] )
    dst = np.float32( [ top_left_dst, top_right_dst, bottom_right_dst, bottom_left_dst ] )
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 578, 460      | 320, 0        | 
| 702, 460      | 960, 0        |
| 1088, 720     | 960, 720      |
| 192, 720      | 320, 720      |

I verified that my perspective transform was working as expected by drawing the source and destimation polygon using `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here is an example of the persepctive transformation for both RGB and the threshold binary images. (For the result of the other test iamges see [here](https://github.com/lowspin/CarND-Proj04-Advanced-Lane-Lines/tree/master/output_images/test-images-3-perspective-transform)).

![alt text][image6]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

First, I extracted the pixels from the binary image that most likely belong to lane lines. I tried two methods - the sliding window and the convolution methods, as described below.

In the sliding window method (`imagefunctions.py` lines 157-188), I implemented a python class called `trackerwincount` (see `trackers.py` lines 4-133). First, I take a histogram of the bottom half of the warped image by vertically summing up the detected pixels in the lower half of the image (`trackers.py` line 22), adn the maximum is used as an initial estimate of where the lane lines begin. The image is then divided into 9 horizontal strips. Starting from the lowest strip (level), a rectangular window is used near the last estimated positions. For each sliding position, the number of binary pixels are counted. If a better location is found, the best estimate is updated. All the detected pixels are used to fit a 2-degree polynomial using numpy's `polyfit()` function (`imagefunctions.py` lines 166-167). After some fine-tuning of the parameters (e.g. adjust the window size to reject suprious points), I was able to detect lane lines in all the test-images. An example is shown below,
![alt text][image7]

The second using convolution (`imagefunctions.py` lines 190-234) uses a similar concept of dividing the image into 9 horizontal level. However, instead of counting the pixels in a sliding window, perform a convolution of the horizontal pixel count with a rectangular window and select the window x-position with the highest correlation, or max convlution result. The centroid of the final windows (total 9 points for each lane line) are used to fir a 2-degree polynomial using numpy's `polyfit()` function (`imagefunctions.py` lines 211-212). After some fine-tuning of the parameters (e.g. reject results with low correlation), I was able to detect lane lines in all the test-images. An example is shown below,
![alt text][image8]

In the end, I decided to go with the sliding window method, because it seems more accurate.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 99 through 121 in my code in `genimages.py`. First I used the standard pixel to meters conversion ratios for both x- and y-direction and scale the original x and y coordinates to real-world values and redo the `polyfit` operation to get the polynomial coefficients in real-world dimensions. Then I used the formula from the lecture to calculate the radius of curvatures for the left and right lane lines separately (`genimages.py` lines 109-110):
```
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```
To calculate the offset with respect to lane center, I first find the x-coordinates of the lane lines in the lowest point of the image frame (`genimages.py` llines 116-117) and calculate the mean, which gives the x-cordinate of the center of the lane right in front of the vehicle. Then, assuming the camera is mounted in the center of the vehicle, I find the difference between the horizontal center of the image to this lane center, which gives the offset.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 130 through 167 in my code in `genimages.py` in the function `processOneFrame()`.  Here is an example of my result on a test image:

![alt text][image9]

The labels "Detected Left" and "Detected Right" are sanity check returned boolean values used in sequential frames video inputs and are always "False" for still images. For the results of the rest of the test images, please see [here](https://github.com/lowspin/CarND-Proj04-Advanced-Lane-Lines/tree/master/output_images/test-images-5-results).

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result file](./result.mp4), or view it on [Youtube](https://youtu.be/AELMPOjgoOs)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For sequentially correlated frames in a video file, we can do a few more things to improve reliability in the detection. 

First, I used a faster search algorithm if the lane lines were detected in the previous frame, by searching for pixels within a margin from the previous frame's fitted polynomial lines (see `trackers.py` lines 45-47 for left lane lines and lines 84-86 for right lane lines). If lane lines were classified as "not detected" (see next paragraph), the full sliding window algorithm mentioned in the previous section is used. This is done separately for the left and right lane lines, hence we can have only one of the lane lines use the fast algorithm while the other uses the full search.

Second, I implemented a simple sanity check to see if the base of the detected lane line polynomials are roughly where the left and right lane lines are supposed to be, assuming the vehicle is generally in the middle of the lane. If this sanity result passes, we'll update the polynomial coeeficients and the coordinates used in the `polyfit` function in the Linetracker object (`trackers.py` lines 228-249), otherwise we used the previous frame's results and discard the current frame's detection results (see `genimages.py` lines 60-92)

In general, the pipeline fails when the image is not cleaned up enough by the extraction phase. This could be due to shadows or extra line-like features on the ground, e.g. in the challenge videos. In order to make the pipeline more robust, I would isolate a few sequence of frames that fails and try to adjust the image processing parameters and/or more strigent sanity check as well as a softer fall-back like a multi-frame average instead of just reverting to the previous frame's result if the sanity check fails.
