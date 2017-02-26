import numpy as np
import cv2

class trackerwincount():
    def __init__(self, mynwindows=9, mywinmargin = 60, myminpix = 50):
        # number of sliding windows
        self.nwindows= mynwindows
        # Set the width of the windows +/- margin
        self.winmargin = mywinmargin
        # Set minimum number of pixels found to recenter window
        self.minpix = myminpix
        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []

    def find_lane_points(self, warpedbin, leftdetected=False, rightdetected=False, left_fit=[], right_fit=[]):
        # Reset lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []

        # Take a histogram of the bottom half of the warped image
        histogram = np.sum(warpedbin[int(warpedbin.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((warpedbin, warpedbin, warpedbin))*255
        # print(histogram)
        # plt.plot(histogram)
        # plt.show()

        # Find the peak of the left and right halves of the histogram
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Calc height of windows
        window_height = np.int(warpedbin.shape[0]/self.nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warpedbin.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ###############################################
        # LEFT LANE LINES
        ###############################################
        if (leftdetected==True & len(left_fit)>0): # use fast detection around polynomial
            margin = 100
            self.left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        else: # revert to full detection
            # Current positions to be updated for each window
            leftx_current = leftx_base

            # Step through the windows one by one
            for window in range(self.nwindows):

                # Identify window boundaries in x and y (and right and left)
                win_y_low = warpedbin.shape[0] - (window+1)*window_height
                win_y_high = warpedbin.shape[0] - window*window_height
                win_xleft_low = leftx_current - self.winmargin
                win_xleft_high = leftx_current + self.winmargin

                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

                # Append these indices to the lists
                self.left_lane_inds.append(good_left_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > self.minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

                #Update windows
                win_xleft_low = leftx_current - self.winmargin
                win_xleft_high = leftx_current + self.winmargin

                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)

            # Concatenate the arrays of indices
            self.left_lane_inds = np.concatenate(self.left_lane_inds)

        ###############################################
        # RIGHTT LANE LINES
        ###############################################
        if (rightdetected==True & len(right_fit)>0): # use fast detection around polynomial
            margin = 100
            self.right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
        else: # revert to full detection
            # Current positions to be updated for each window
            rightx_current = rightx_base

            # Step through the windows one by one
            for window in range(self.nwindows):

                # Identify window boundaries in x and y (and right and left)
                win_y_low = warpedbin.shape[0] - (window+1)*window_height
                win_y_high = warpedbin.shape[0] - window*window_height
                win_xright_low = rightx_current - self.winmargin
                win_xright_high = rightx_current + self.winmargin

                # Identify the nonzero pixels in x and y within the window
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                # Append these indices to the lists
                self.right_lane_inds.append(good_right_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_right_inds) > self.minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

                #Update windows
                win_xright_low = rightx_current - self.winmargin
                win_xright_high = rightx_current + self.winmargin

                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,255), 2)

            # Concatenate the arrays of indices
            self.right_lane_inds = np.concatenate(self.right_lane_inds)

        # Extract left and right line pixel positions
        leftx = np.asarray(nonzerox[self.left_lane_inds])
        lefty = np.asarray(nonzeroy[self.left_lane_inds])
        rightx = np.asarray(nonzerox[self.right_lane_inds])
        righty = np.asarray(nonzeroy[self.right_lane_inds])

        out_img[nonzeroy[self.left_lane_inds], nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[self.right_lane_inds], nonzerox[self.right_lane_inds]] = [0, 0, 255]

        # define confidence as the number of pixels used for polyfit
        confidenceleft = len(leftx)
        confidenceright = len(rightx)

        return leftx, lefty, rightx, righty, out_img, confidenceleft, confidenceright


class trackerconv():
    def __init__(self): #, My_ym=1, My_xm=1, Mysmooth=15):
        self.recent_centers = []
        # self.ym_per_pix = My_ym
        # self.xm_per_pix = My_xm
        # self.smooth_factor = Mysmooth

    def window_mask(self, width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),\
               max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    def find_window_centroids(self, warped, window_width, window_height, margin, mincovl, maxslide):
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((warped, warped, warped))*255
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        # draw first window
        win_y_low = warped.shape[0]-window_height
        win_y_high = warped.shape[0]
        win_xleft_low = int(max(l_center+offset-margin,0))
        win_xleft_high = int(min(l_center+offset+margin,warped.shape[1]))
        win_xright_low = int(max(r_center+offset-margin,0))
        win_xright_high = int(min(r_center+offset+margin,warped.shape[1]))
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,255), 2)

        # Go through each layer looking for max pixel locations
        cummconvleft = 0.0
        cummconvright = 0.0
        for level in range(1,(int)(warped.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
            conv_temp = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # reject low-correlation results
            if((conv_signal[int(conv_temp)] > mincovl) & (abs(window_centroids[-1][0]-conv_temp) < maxslide)):
                l_center = conv_temp
                cummconvleft = cummconvleft + conv_signal[int(conv_temp)]
                #print(conv_signal[l_center])
            else:
                l_center = window_centroids[-1][0]
                #print('left - low corr')
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
            conv_temp = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            if((conv_signal[int(conv_temp)] > mincovl) & (abs(window_centroids[-1][1]-conv_temp) < maxslide)):
                r_center = conv_temp
                cummconvright = cummconvright + conv_signal[int(conv_temp)]
                #print(conv_signal[r_center])
            else:
                r_center = window_centroids[-1][1]
                #print('right - low corr')
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

            # Draw the windows on the visualization image
            win_y_low = warped.shape[0]-(level+1)*window_height
            win_y_high = warped.shape[0]-level*window_height
            win_xleft_low = l_min_index
            win_xleft_high = l_max_index
            win_xright_low = r_min_index
            win_xright_high = r_max_index
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,255), 2)

        # define confidence as the correlation level (height of convolution)
        confidenceleft = cummconvleft
        confidenceright = cummconvright

        return window_centroids, out_img, confidenceleft, confidenceright

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
