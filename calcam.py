import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt

# prepare object points
nx = 9 #number of inside corners in x
ny = 6 #number of inside corners in y

# Arrays to store object and image points
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# prepare object points, (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

images = glob.glob('./camera_cal/*.jpg')

plt.figure(figsize=(25,15))

for idx, fname in enumerate(images):
    # load image
    img = cv2.imread(fname)
    img_orig = img

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If corners are found, add object points, image points
    if(ret == True):
        imgpoints.append(corners)
        objpoints.append(objp)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.subplot(5, 4, idx+1)
        plt.imshow(img)
        print(idx, ret)
    else:
        plt.subplot(5, 4, idx+1)
        plt.imshow(img)
        print(ret)

plt.show()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# save camera calibration results
# dist_pickle = []
# dist_pickle['mtx'] = mtx
# dist_pickle['dist'] = dist
pickle.dump( [mtx,dist], open( "./calib_pickle.p", "wb"))

# plot original and undistorted images
for idx, fname in enumerate(images):
    print(fname)
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(dst)
    plt.title('Undistorted')
    outfname = 'undistort_' + str(idx) + '.png'
    plt.savefig(outfname)
