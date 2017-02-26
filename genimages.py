import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imagefunctions

# keep track of parameters from frames to frames
from trackers import Line
linetrackerleft = Line()
linetrackerright = Line()
videomode = True

# get camera calibration results
f = open('calib_pickle.p','rb')
mtx, dist = pickle.load(f)
f.close()

def processOneFrame(imgRGB, plotid=0):
    ###############################################
    # Keep track of things across frames
    ###############################################
    firstframe = True if (linetrackerleft.allx is None) else False
    ###############################################

    ###############################################
    # Undistort camera distortion
    ###############################################
    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
    undst = cv2.undistort(imgBGR, mtx, dist, None, mtx)
    undst_rgb = cv2.cvtColor(undst, cv2.COLOR_BGR2RGB)
    ###############################################

    ###############################################
    # Extract pixels for detection
    ###############################################
    img_bin = imagefunctions.extractpixels(undst)
    ###############################################

    ###############################################
    # Perspective Transform
    ###############################################
    M, Minv, warpedimg, warpedbin = imagefunctions.perspectiveTransform(undst_rgb,img_bin)
    ###############################################

    ###############################################
    # Find Lane Lines
    ###############################################
    # #### Method 1 - Counting with Sliding Window
    # Use Fast search if BOTH lane lines were detected in the previous frame (Note: if not videomode, detected will not be True)
    leftx, lefty, rightx, righty, left_fit, right_fit, out_img, confidenceleft, confidenceright = imagefunctions.slidingWindowCount(warpedbin, linetrackerleft.detected, linetrackerright.detected, linetrackerleft.current_fit, linetrackerright.current_fit)
    #leftx, lefty, rightx, righty, left_fit, right_fit, out_img, confidenceleft, confidenceright = imagefunctions.slidingWindowCount(warpedbin, False, False, [], [], True, plotid )
    # #### Method 2 - Convolutions
    #leftx, lefty, rightx, righty, left_fit, right_fit, out_img, confidenceleft, confidenceright = imagefunctions.slidingConvolution(warpedbin, True, plotid )

    ###############################################
    # Sanity Check for Video Processing ONLY
    ###############################################
    # choose the maximum y-value, corresponding to the bottom of the image
    ploty = np.linspace(0, warpedbin.shape[0]-1, warpedbin.shape[0] )
    y_eval = np.max(ploty)
    left_warpnear_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_warpnear_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

    if (videomode == True):
        # left_nearlane_x is between 1.60m to 1.90m
        if (left_warpnear_x < 300) | (left_warpnear_x > 400):
            linetrackerleft.detected = False
            # use last frame's results if exists
            if not firstframe:
                left_fit = linetrackerleft.current_fit
                leftx = linetrackerleft.allx
                lefty = linetrackerleft.ally
        else:
            linetrackerleft.detected = True
            linetrackerleft.current_fit = left_fit
            linetrackerleft.allx = leftx
            linetrackerleft.ally = lefty

        # right_nearlane_x is betwwen 5.25 to 5.55
        if  (right_warpnear_x < 980) | (right_warpnear_x > 1080):
            linetrackerright.detected = False
            # use last frame's results if exists
            if not firstframe:
                right_fit = linetrackerright.current_fit
                rightx = linetrackerright.allx
                righty = linetrackerright.ally
        else:
            linetrackerright.detected = True
            linetrackerright.current_fit = right_fit
            linetrackerright.allx = rightx
            linetrackerright.ally = righty
    ###############################################

    ###############################################
    # Calculate Curvature and Displacement
    ###############################################
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # choose the maximum y-value, corresponding to the bottom of the image
    # ploty = np.linspace(0, warpedbin.shape[0]-1, warpedbin.shape[0] )
    # y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

    # find drift (Offset) from center
    view_width = xm_per_pix * warpedimg.shape[1]
    # detected lane lines at the bottom of image
    left_nearlane_x = left_fit_cr[0]*(y_eval*ym_per_pix)**2 + left_fit_cr[1]*y_eval*ym_per_pix + left_fit_cr[2]
    right_nearlane_x = right_fit_cr[0]*(y_eval*ym_per_pix)**2 + right_fit_cr[1]*y_eval*ym_per_pix + right_fit_cr[2]
    # mid point between detected lane lines
    midpointdet = np.mean([left_nearlane_x,right_nearlane_x])
    # offset from camera center
    driftdet = midpointdet - 0.5*view_width


    ###############################################

    ###############################################
    # Project back to real world
    ###############################################
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warpedimg).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp = warp_zero

    # Generate x and y values for plotting
    # ploty = np.linspace(0, warpedbin.shape[0]-1, warpedbin.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (imgBGR.shape[1], imgBGR.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undst_rgb, 1, newwarp, 0.3, 0)

    label1 = 'Lane Drift: {0:.2f}'.format(driftdet) + 'm'
    # label2 = 'Left x: {0:.2f}'.format(left_warpnear_x) + 'px'
    # label3 = 'Right x: {0:.2f}'.format(right_warpnear_x) + 'px'
    label2 = 'Left Curvature: {0:.2f}'.format(left_curverad) + 'm'
    label3 = 'Right Curvature: {0:.2f}'.format(right_curverad) + 'm'
    # label4 = 'Confidence Left: {0:.2f}'.format(confidenceleft)
    # label5 = 'Confidence Right: {0:.2f}'.format(confidenceright)
    label4 = 'Detected Left: ' + str(linetrackerleft.detected)
    label5 = 'Detected Right: ' + str(linetrackerright.detected)
    cv2.putText(result, label1, org=(30, 40), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.25, color=(0,255,0))
    cv2.putText(result, label2, org=(30, 80), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.25, color=(0,255,0))
    cv2.putText(result, label3, org=(30, 120), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.25, color=(0,255,0))
    cv2.putText(result, label4, org=(660, 80), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.25, color=(0,255,255))
    cv2.putText(result, label5, org=(660, 120), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.25, color=(0,255,255))

    return result

############################################################

# Part 1 - Test Images
images = glob.glob('./test_images/*.jpg')
# images = glob.glob('./some_folder/*.jpg')
for idx, fname in enumerate(images):
    # load image
    img = cv2.imread(fname)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = processOneFrame(img_rgb, plotid=idx)

    plt.figure(figsize=(15,10))
    plt.imshow(result)
    outfname = 'result_' + str(idx) + '.jpg'
    plt.savefig(outfname)
    plt.close()

############################################################

# Part 2 - Video File
# from moviepy.editor import VideoFileClip
# result_output = 'result.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# # clip1 = VideoFileClip("challenge_video.mp4")
# white_clip = clip1.fl_image(processOneFrame) # NOTE: this function expects color images!!
# white_clip.write_videofile(result_output, audio=False)
