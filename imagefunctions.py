import numpy as np
import cv2
import matplotlib.pyplot as plt
from trackers import trackerwincount, trackerconv

# Sobel gradient in one direction and thresholding
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    thresh_min = thresh[0]
    thresh_max = thresh[1]

    # Convert BGR to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Magnitude of the gradient
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Direction of the Gradient
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def extractpixels(imgBGR, plotall=False, plotid=0):
    ###############################################
    # Threholding to extract pixels
    ###############################################
    # Convert to grayscale
    gray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(imgBGR, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(imgBGR, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(imgBGR, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(imgBGR, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # extract s-channel of HLS colorspace
    hls = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    combined2 = np.zeros_like(s_binary)
    combined2[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

    if(plotall==True):
        outfname = 'six_edges_' + str(plotid) + '.jpg'
        plt.figure(figsize=(25,15))
        plt.subplot(3, 2, 1)
        plt.imshow(gradx, cmap='gray')
        plt.title('gradx')
        plt.subplot(3, 2, 2)
        plt.imshow(grady, cmap='gray')
        plt.title('grady')
        plt.subplot(3, 2, 3)
        plt.imshow(mag_binary, cmap='gray')
        plt.title('mag_binary')
        plt.subplot(3, 2, 4)
        plt.imshow(combined, cmap='gray')
        plt.title('combined gradient')
        plt.subplot(3, 2, 5)
        plt.imshow(s_binary, cmap='gray')
        plt.title('s_binary')
        plt.subplot(3, 2, 6)
        plt.imshow(combined2, cmap='gray')
        plt.title('Combined S channel and gradient')
        plt.savefig(outfname)

    # Return the combined image
    return combined2

def perspectiveTransform(undst_rgb, img_bin, plotall=False, plotid=0):
    # Define perspective transformation area
    img_size = (img_bin.shape[1], img_bin.shape[0]) #(1280, 720)

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

    # perform transformation
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undst_rgb, M, img_size, flags=cv2.INTER_LINEAR)
    warped_bin = cv2.warpPerspective(img_bin, M, img_size, flags=cv2.INTER_LINEAR)

    if(plotall==True):
        outfname = 'warped_' + str(plotid) + '.jpg'
        plt.subplot(131),plt.imshow(origimg),plt.title(fname) #('Input')
        plt.subplot(132),plt.imshow(warpedimg),plt.title('Warped Image')
        plt.subplot(133),plt.imshow(warpedbin, cmap='gray'),plt.title('Warped Lines')
        # plt.show()
        outfname = 'warped_' + str(plotid) + '.jpg'
        plt.savefig(outfname)

    return M, Minv, warped, warped_bin

def slidingWindowCount(warpedbin, leftdetected=False, rightdetected=False, left_fit=[], right_fit=[], plotall=False, plotid=0):
    # Instantiate tracker object
    curve_points = trackerwincount(mynwindows=9, mywinmargin = 60, myminpix = 50)

    #### Method 1 - Counting with Sliding Window
    leftx, lefty, rightx, righty, out_img, confidenceleft, confidenceright = curve_points.find_lane_points(warpedbin, leftdetected, rightdetected, left_fit, right_fit)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warpedbin.shape[0]-1, warpedbin.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if(plotall==True):
        # out_img already contains the windows
        plt.figure()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='magenta', linewidth=4.0)
        plt.plot(right_fitx, ploty, color='cyan', linewidth=4.0)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.title('Results - Counting Sliding Window')
        #plt.show()
        outfname = 'res-warp_wincount_' + str(plotid) + '.jpg'
        plt.savefig(outfname)
        plt.close()

    #return leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty, out_img, confidenceleft, confidenceright
    return leftx, lefty, rightx, righty, left_fit, right_fit, out_img, confidenceleft, confidenceright

def slidingConvolution(warpedbin, plotall=False, plotid=0):
    # Instantiate tracker object
    curve_centers = trackerconv()

    window_width = 50
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching
    mincovl = 50 # min covolution strength to accept
    maxslide = 50 # max acceptable window slide

    window_centroids, out_img, confidenceleft, confidenceright = curve_centers.find_window_centroids(warpedbin, window_width, window_height, margin, mincovl, maxslide) #margin)

    leftx = np.asarray([int(i[0]) for i in window_centroids])
    rightx = np.asarray([int(i[1]) for i in window_centroids])

    #lefty = range(int(window_height/2.):out_img.shape[0]:window_height)
    lefty = np.linspace(out_img.shape[0]-int(window_height/2.),int(window_height/2.),9, dtype=np.int16)
    righty = lefty

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warpedbin.shape[0]-1, warpedbin.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if(plotall==True):
        # out_img already contains the windows
        plt.figure()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='magenta', linewidth=4.0)
        plt.plot(right_fitx, ploty, color='cyan', linewidth=4.0)
        plt.plot(leftx, lefty, 'ro')
        plt.plot(rightx, righty, 'bo')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.title('Results - Convolution')
        #plt.show()
        outfname = 'res-warp_convol_' + str(plotid) + '.jpg'
        plt.savefig(outfname)
        plt.close()

    return leftx, lefty, rightx, righty, left_fit, right_fit, out_img, confidenceleft, confidenceright

def fastersearch(warpedbin, left_fit, right_fit):
    nonzero = warpedbin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # define confidence as the number of pixels used for polyfit
    confidenceleft = len(leftx)
    confidenceright = len(rightx)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warpedbin, warpedbin, warpedbin))*255
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return leftx, lefty, rightx, righty, left_fit, right_fit, out_img, confidenceleft, confidenceright
