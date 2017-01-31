#importing some useful packages
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from ImageUtils import draw_lines
from ImageUtils import extract_lines_naiive_hough
from ImageUtils import extract_edges

import cv2
from sklearn import linear_model
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from scipy.odr import Model, ODR, Data
from scipy.stats import linregress
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from collections import deque


import math

def orthoregress(x, y):
    """Perform an Orthogonal Distance Regression on the given data,
    using the same interface as the standard scipy.stats.linregress function.
    Arguments:
    x: x data
    y: y data
    Returns:
    [m, c]
    Uses standard ordinary least squares to estimate the starting parameters
    then uses the scipy.odr interface to the ODRPACK Fortran code to do the
    orthogonal distance calculations.

    Source: http://blog.rtwilson.com/orthogonal-distance-regression-in-python/
    """
    linreg = linregress(x, y)
    mod = Model(f)
    dat = Data(x, y)
    od = ODR(data = dat, model = mod, beta0=linreg[0:2], maxit=10)
    out = od.run()
    return list(out.beta)

def f(p, x):
    """
    Basic linear regression 'model' for use with ODR
    Source: http://blog.rtwilson.com/orthogonal-distance-regression-in-python/
    """
    return (p[0] * x) + p[1]


def lines_from_edge_points_RANSAC(edges, roi):

    roi = np.array(roi).reshape(-1,2)
    min_x= min(roi[:,0]); max_x = max(roi[:,0])
    min_y = min(roi[:,1]); max_y = max(roi[:,1])

    # Generate X,Y coordinates of the locations of non-zero values of edges
    res = np.nonzero(edges)
    x = np.array(res[1]).reshape(len(res[1]), -1)
    y = np.array(res[0]).reshape(len(res[0]), -1)

    # Fit line using all data
    model = linear_model.LinearRegression()
    # model = make_pipeline(PolynomialFeatures(2), Ridge())
    # model.fit(x, y)

    # Robustly fit linear model with RANSAC algorithm
    model_ransac = linear_model.RANSACRegressor(base_estimator=model)
    model_ransac.fit(x, y)

    inlier_mask = model_ransac.inlier_mask_
    # outlier_mask = np.logical_not(inlier_mask)

    # Compute the points generated within the x-range (min/max)
    x_range = np.array(range(min_x, max_x + 1))
    x_range = x_range.reshape(len(x_range), -1)

    # Predict data of estimated models
    y_range = np.array(np.around(model_ransac.predict(x_range), decimals=0), dtype=np.int64)


    # Y points in the x-ROI
    idc = (y_range >= min_y) & (y_range <= max_y)
    x_range = x_range[idc]
    y_range = y_range[idc]

    # min point / max point of line
    pt1 = (int(min(x_range)), int(y_range[x_range == min(x_range)][0]))
    pt2 = (int(max(x_range)), int(y_range[x_range == max(x_range)][0]))

    return pt1, pt2


def line_from_edge_points_TLS(edges, roi):
    '''
    Using the bounds of ROI and edge (pixels), fit TLS line model to edges.
    Then, generate the enpoints which are boun by the ROI
    :param edges: Edge image
    :return: fitted line endpoint segments as determined by ROI

    Author: Gerti Tuzi
            01/23/2017
    '''
    roi = np.array(roi).reshape(-1,2)
    min_x= min(roi[:,0]); max_x = max(roi[:,0])
    min_y = min(roi[:,1]); max_y = max(roi[:,1])
    # Generate X,Y coordinates of the locations of non-zero values of edges
    res = np.nonzero(edges)
    [m, c] = orthoregress(res[1], res[0])

    # Compute the points generated within the x-range (min/max)
    (x_range, y_range) = (np.array(range(min_x, max_x+1)), np.array(range(min_x, max_x+1))*m + c)

    # Y points in the x-ROI
    idc = (y_range >= min_y) & (y_range <= max_y)
    x_range = x_range[idc]
    y_range = y_range[idc]

    # min point / max point of line
    pt1 = (int(min(x_range)), int(y_range[x_range == min(x_range)][0]))
    pt2 = (int(max(x_range)), int(y_range[x_range == max(x_range)][0]))

    return pt1,pt2


def extract_lines_RANSAC_fitting(image, filtarr = None):
    '''
            Generate lane lines and draw on input image (returned copy)
            Split roi into L|R
            Use RANSAC fitting on edge pixels
            Draw fitted line on ime

        :param image: Source raw image
        :return: Raw image with lines drawn, raw line image
        '''

    x_idc = 0
    y_idc = 1

    min_x = 0
    min_y = 0
    ctr_x = int(math.floor(image.shape[1] / 2.))
    ctr_y = int(math.floor(image.shape[0] / 2.))
    max_x = int(image.shape[1])
    max_y = image.shape[0]

    # ROI Definition

    left_bottom = (0, max_y)
    right_bottom = (max_x, max_y)
    center_bottom = (ctr_x, max_y)
    apex = (ctr_x, ctr_y + 50)  # Push it down a bit to vanishing point
    apex_x_offset = int(math.floor(50 / 2.))

    # **************** Split the ROI into Left and Right
    L_verts = np.array([[(left_bottom[x_idc], left_bottom[y_idc]),
                         (apex[x_idc] - apex_x_offset, apex[y_idc]),
                         (apex[x_idc], apex[y_idc]),
                         (center_bottom[x_idc], center_bottom[y_idc])]])

    R_verts = np.array([[(center_bottom[x_idc], center_bottom[y_idc]),
                         (apex[x_idc], apex[y_idc]),
                         (apex[x_idc] + apex_x_offset, apex[y_idc]),
                         (right_bottom[x_idc], right_bottom[y_idc])]])

    # Obtain the edges for L|R ROI
    roi_edges_L = extract_edges(image=image, roi_verteces=L_verts)
    roi_edges_R = extract_edges(image=image, roi_verteces=R_verts)

    # Get the line endpoints from edges for each L|R side - using RANSAC fitting
    pt1L, pt2L = lines_from_edge_points_RANSAC(roi=L_verts, edges=roi_edges_L)
    pt1R, pt2R = lines_from_edge_points_RANSAC(roi=R_verts, edges=roi_edges_R)

    L = []
    if filtarr is not None:
        # Shift the filter
        for i in range(0, len(filtarr) - 1):
            filtarr[i, :] = filtarr[i + 1, :]
        filtarr[len(filtarr) - 1, :] = [pt1L[0], pt1L[1], pt2L[0], pt2L[1],
                                        pt1R[0], pt1R[1], pt2R[0], pt2R[1]]
        LL = np.around(filtarr.mean(axis=0), decimals=0).astype(int)
        L = [LL[0:4].tolist(), LL[4:8].tolist()]
    else:
        # Collect line endpoints
        L = [[pt1L, pt2L], [pt1R, pt2R]]

    # Combine both edges to plot for debugging
    roi_edges = roi_edges_L + roi_edges_R

    # Empty image to draw only the detected lines
    fitted_line_image = np.zeros_like(image)
    draw_lines(img=fitted_line_image, lines=L, color=[0, 0, 255], thickness=4)

    # Draw lines on the original image
    raw_image_with_lines = np.copy(image)
    draw_lines(img=raw_image_with_lines, lines=L, color=[255, 0, 0], thickness=4)

    return raw_image_with_lines, fitted_line_image, roi_edges


def extract_lines_TLS_fitting(image, filtarr = None):
    '''
        Generate lane lines and draw on input image (returned copy)
        Split roi into L|R
        Use TLS fitting to edge pixels
        Draw fitted line on ime

    :param image: Source raw image
    :return: Raw image with lines drawn, raw line image
    '''

    x_idc = 0
    y_idc = 1

    min_x = 0
    min_y = 0
    ctr_x = int(math.floor(image.shape[1] / 2.))
    ctr_y = int(math.floor(image.shape[0] / 2.))
    max_x = int(image.shape[1])
    max_y = image.shape[0]

    # ROI Definition

    left_bottom = (0, max_y)
    right_bottom = (max_x, max_y)
    center_bottom = (ctr_x, max_y)
    apex = (ctr_x, ctr_y + 50) # Push it down a bit to vanishing point
    apex_x_offset = int(math.floor(50 / 2.))

    # **************** Split the ROI into Left and Right
    L_verts = np.array([[(left_bottom[x_idc], left_bottom[y_idc]),
                         (apex[x_idc] - apex_x_offset, apex[y_idc]),
                         (apex[x_idc], apex[y_idc]),
                         (center_bottom[x_idc], center_bottom[y_idc])]])

    R_verts = np.array([[(center_bottom[x_idc], center_bottom[y_idc]),
                         (apex[x_idc], apex[y_idc]),
                         (apex[x_idc] + apex_x_offset, apex[y_idc]),
                         (right_bottom[x_idc], right_bottom[y_idc])]])

    # Obtain the edges for L|R ROI
    roi_edges_L = extract_edges(image=image, roi_verteces=L_verts)
    roi_edges_R = extract_edges(image=image, roi_verteces=R_verts)

    # Get the line endpoints from edges for each L|R side - using TLS fitting
    pt1L, pt2L = line_from_edge_points_TLS(roi=L_verts, edges=roi_edges_L)
    pt1R, pt2R = line_from_edge_points_TLS(roi=R_verts, edges=roi_edges_R)

    L = []
    if filtarr is not None:
        # Shift the filter
        for i in range(0, len(filtarr)-1):
            filtarr[i, :] = filtarr[i+1, :]
        filtarr[len(filtarr)-1, :] = [pt1L[0], pt1L[1], pt2L[0], pt2L[1],
                         pt1R[0], pt1R[1], pt2R[0], pt2R[1]]
        LL = np.around(filtarr.mean(axis=0), decimals=0).astype(int)
        L = [LL[0:4].tolist(), LL[4:8].tolist()]
    else:
        # Collect line endpoints
        L = [[pt1L, pt2L], [pt1R, pt2R]]

    # Combine both edges to plot for debugging
    roi_edges = roi_edges_L + roi_edges_R

    # Empty image to draw only the detected lines
    fitted_line_image = np.zeros_like(image)
    draw_lines(img=fitted_line_image, lines=L, color=[0, 0, 255], thickness=4)

    # Draw lines on the original image
    raw_image_with_lines = np.copy(image)
    draw_lines(img=raw_image_with_lines, lines=L, color=[255, 0, 0], thickness=4)

    return raw_image_with_lines, fitted_line_image, roi_edges

# *************************************************
# *************************************************
# *************************************************

imgfolder = "repo/test_images/"
image = mpimg.imread( imgfolder + 'whiteCarLaneSwitch.jpg')



x_idc = 0
y_idc = 1

min_x = 0
min_y = 0
ctr_x = int(math.floor(image.shape[1] / 2.))
ctr_y = int(math.floor(image.shape[0] / 2.))
max_x = int(image.shape[1])
max_y = image.shape[0]

# ROI Definition
# <x,y>

left_bottom = (0, max_y)
right_bottom = (max_x, max_y)
center_bottom = (ctr_x, max_y)
apex = (ctr_x, ctr_y + 50)
apex_x_offset = int(math.floor(50/2.))


# Create a trapezius for ROI
verts = np.array([[(left_bottom[x_idc], left_bottom[y_idc]),
                   (apex[x_idc] - apex_x_offset, apex[y_idc]),
                   (apex[x_idc] + apex_x_offset, apex[y_idc]),
                   (right_bottom[x_idc], right_bottom[y_idc])]])



# ************ Draw ROI bounding lines on the raw image
l1 = [left_bottom[x_idc], left_bottom[y_idc], apex[x_idc] - apex_x_offset, apex[y_idc]]
l2 = [apex[x_idc] - apex_x_offset, apex[y_idc], apex[x_idc] + apex_x_offset, apex[y_idc]]
l3 = [apex[x_idc] + apex_x_offset, apex[y_idc], right_bottom[x_idc], right_bottom[y_idc]]
l4 = [right_bottom[x_idc], right_bottom[y_idc], left_bottom[x_idc], left_bottom[y_idc]]
L = [l1, l2, l3, l4]
roi_lines_image = np.copy(image)
draw_lines(img=roi_lines_image, lines=L, color=[0, 0, 255], thickness=4)
# *************************************************************

# Extract lane lines using Canny edges and Hough Transforms - in ROI defined above
# raw_image_with_edges, edge_line_image, lines, roi_edges = extract_lines_naiive_hough(image, roi_verteces = verts)


# # Plots
# plt.subplot(321)
# plt.imshow(image)
# plt.title('Original Image')
# plt.subplot(322)
# plt.imshow(roi_edges)
# plt.title('Edges in ROI')
# plt.subplot(323)
# plt.imshow(roi_lines_image)
# plt.title('ROI Lines')
# plt.subplot(324)
# plt.imshow(edge_line_image)
# plt.title('Raw Lines')
# plt.subplot(325)
# plt.imshow(raw_image_with_edges)
# plt.title('Lines on original image')



# ****************** TLS fitting on edges ************************ #

raw_image_with_lines, fitted_line_image, roi_edges = extract_lines_TLS_fitting(image)


# Plots
plt.subplot(321)
plt.imshow(image)
plt.title('Original Image')
plt.subplot(322)
plt.imshow(roi_edges)
plt.title('Edges in ROI')
plt.subplot(323)
plt.imshow(roi_lines_image)
plt.title('ROI Bounding Lines')
plt.subplot(324)
plt.imshow(fitted_line_image)
plt.title('Fitted Lines')
plt.subplot(325)
plt.imshow(raw_image_with_lines)
plt.title('Lines on original image')

plt.pause(0)

filtsize = 10
filtarr = np.zeros((filtsize, 8), dtype=int)
# filtarr = None

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    # raw_image_with_lines, \
    # fitted_line_image, \
    # roi_edges = extract_lines_TLS_fitting(image=image, filtarr=filtarr)

    raw_image_with_lines, \
    fitted_line_image, \
    roi_edges = extract_lines_RANSAC_fitting(image=image, filtarr=filtarr)

    return raw_image_with_lines


try:

    currdir = os.getcwd() + '/repo/'
    white_output = currdir + 'chall.mp4'
    clip1 = VideoFileClip(currdir + "challenge.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
except Exception as ex:
    print str(ex)

print 'Done'