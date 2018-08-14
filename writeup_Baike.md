# Advance Lane Lines Finding Project

## Goals

The goals/steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position on center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) points

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera calibration

#### 1. Briefly, state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration code could be found on the first code shell [Advance_Lane_Find notebook](Advance_Lane_Find.ipynb). Using `cv2.findChessboardCorners`, the corners points are stored in an array `imgpoints` for each calibration image where the chessboard could be found. The object points will always be the same as the known coordinates of the chessboard with zero as 'z' coordinate because the chessboard is flat. The object points are stored in an array called `objpoints`. I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera` function.

The image below depicts the rsults of applying `cv2.undistort`, using the calibration and distortion coefficients, to one of the chessboar images:

![Camera calibration](./misc/Camera_Calibration.JPG)

The camera calibration and distortion coefficients are stored on file [Camera calibration coefficients.p](camera_calibrate_coefficients.p) using `pickle` to be used on the main [notebook](Advance_Lane_Find.ipynb)

### Pipeline (single images)



#### 1. Provide an example of a distortion-corrected image.

The following image shows the result of applying the camera calibration to one of the test images:

![distortion-corrected imgae](./misc/dashcam_undist.JPG)

As seen from images, the effect of `cv2.undistort` is subtle, but can be perceived from the difference in shape of the hood of the car at the bottom coeners of the image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

The code used to experiment with color, gradients, and thresholds could be found on the [Advance_Lane_Find notebook](Advance_Lane_Find.ipynb).

I explore several combinations of sobel gradient thresholds and color channel thresholds in mutiple coloe spaces. Below is the demonstration of applying different thresholds and combination of sobel, direction, and mutiple color spaces for the same image: 

![Thresholds imgae1](./misc/Threshold_Image_1.JPG)
![Thresholds imgae1](./misc/Threshold_Image_2.JPG)

Ultimately, I chosed the following :

- Sobel X and Sobel Y: 
- Magnitude : 
- Gradient direction 
- S-channel of HLS
- V-channel of HSV
- Combination of all the above (Sobel X and Sobel Y) or (Magnitude and Gradient)

The full combination of these gradients leads to a 'noisy' image as shown in below:

![Stack_Binarized imgae](./misc/Original_Binarized_Stacked_Img.JPG)

On the [Advance_Lane_Find notebook](Advance_Lane_Find.ipynb), all the code could be found from the cell just below `Step 2： Use color transforms, gradients, etc, to create a thresholded binary image`

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provided an example of a transformed image.

The code for my perspective transform is titled as` Step 3: Apply a perspective transform to rectify binary image ` in the Jupyter notebook.  The ` bird_eye_view ` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

```
offset = 250
corners = np.float32([[190,720],[600,447],[685,447],[1125,720]])
src = corners

dst = np.float32([[corners[0][0] + offset, corners[0][1]],
                      [corners[0][0] + offset, 0],
                      [corners[3][0] - offset, 0],
                      [corners[3][0] - offset, corners[3][1]]])   

```

Using `cv2.getPerspectiveTransform`, a transformation matrix was calculated, and an inverse transformation matrix was also calculated to map the points back to the original space . Four points where selected on the first image as the source of the perspective transformation. Those points are highlighted on the following image, after transformation, the result of the transformation on a test image is the following as well:

![Transformation](./misc/Transformer_Perspective_Warped_img.JPG)

The transformation matrix and the inverse transformation matrix was stored using `pickle` to be used on the main notebook [Advanced Lane Lines notebook](Advance_Lane_Find.ipynb). The following picture shows the binary images results after the perspective transformation:

![Binary images transformed](./misc/Perspective_Transform_After_Binarization.JPG)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The codes, which are used to identify lane-line, sit in the cell [75] on the main notebook [Advanced Lane Lines notebook](Advance_Lane_Find.ipynb). Within this cell all the functions are clearly labeled.The first of these computes a histogram of the bottom half of the image and finds the bottom-most x position (or "base") of the left and right lane lines. Originally these locations were identified from the local maxima of the left and right halves of the histogram, but in my final implementation I changed these to quarters of the histogram just left and right of the midpoint. This helped to reject lines from adjacent lanes. The function then identifies ten windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below. This effectively "follows" the lane lines up to the top of the binary image, and speeds processing by only searching for activated pixels over a small portion of the image. Pixels belonging to each lane line are identified and the Numpy polyfit() method fits a second order polynomial to each set of pixels. The image below demonstrates how this process works:

![Fit_Lane_Line](./misc/Fit_Lane-line_Process.JPG)

The imgae below depicts the histogram generated in the cell [75] on the main notebook [Advanced Lane Lines notebook], the resulting base points for the left and right lanes - the two peaks nearst the center - are clearly visible:

![Polynomial fit](./misc/Poly_Fit_Histogram.JPG)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle on the center.

On the step 4 a polynomial was calculated on the meters space to be used here to calculate the curvature. The formula is the following:

```
((1 + (2*fit[0]*yRange*ym_per_pix + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```

where `fit` is the the array containing the polynomial, `yRange` is the max Y value and `ym_per_pix` is the meter per pixel value.

To find the vehicle position on the center:

- Calculate the lane center by evaluating the left and right polynomials at the maximum Y and find the middle point.
- Calculate the vehicle center transforming the center of the image from pixels to meters.
- The sign between the distance between the lane center and the vehicle center gives if the vehicle is on to the left or the right.

The code used to calculate this could be found at `In [20]`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

To display the lane lines on the image, the polynomials where evaluated on a lineal space of the Y coordinates. The generated points where mapped back to the image space using the inverse transformation matrix generated by the perspective transformation. The code used for this operation could be found on `In [19]`, and the following images are examples of this mapping:

![Lane lines fit](images/lanelines.png)


### Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

After some refactoring of the code found at `In [21]`, the project video was processed and the results at [video_output](./video_output/project_video.mp4)

### Discussion

#### 1. Briefly, discuss any problems/issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

- There are a few improvements that could be done on the performance of the process due to repetitive calculations.
- More information could be use from frame to frame to improve the robustness of the process.
- Other gradients could be use (Ex. Laplacian) to improve the line detection.
