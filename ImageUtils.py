import matplotlib.image as mpimg
import numpy as np
import math
import cv2


def image_rgb_threshold(image, thsh):
    '''
        Extract from RGB image the values greater than threshold

    :param image: Image input (RGB)
    :param thsh: Threshold below which values are set to 0
    :return: thresholded image (a thresholded copy),
             threshold values (logical) - the pixels kept
    '''

    # GT: Note the image dimensions
    ysize = image.shape[0]
    xsize = image.shape[1]
    # Note: always make a copy rather than simply using "="
    color_select = np.copy(image)

    # Define our color selection criteria
    # Note: if you run this code, you'll find these are not sensible values!!
    # But you'll get a chance to play with them soon in a quiz
    red_threshold = thsh
    green_threshold = thsh
    blue_threshold = thsh
    r_channel = 0
    g_channel = 1
    b_channel = 2
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Next, I'll select any pixels below the threshold and set them to zero.
    # After that, all pixels that meet my color criterion will be retained,
    # and those that do not will be blacked out.

    # Identify pixels below the threshold
    # Do a boolean or with the "|" character to identify
    # pixels below the thresholds
    # Thresholds is a logical indexer
    thresholds = (image[:, :, r_channel] < rgb_threshold[r_channel]) \
                 | (image[:, :, g_channel] < rgb_threshold[g_channel]) \
                 | (image[:, :, b_channel] < rgb_threshold[b_channel])

    # Set the colors BELOW the thresholds to matrix values (0 - black)
    color_select[thresholds] = [0, 0, 0]

    return color_select, ~thresholds


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""

    '''  Or use BGR2GRAY if you read an image with cv2.imread()
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):

    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    Gerti - Note: Vertices are ordered as <x, y> and they must be numpy
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in np.array(line).reshape(1, 4):
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, thickness=2):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn and line objects
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=thickness)
    return line_img, lines


def weighted_img(img, initial_img, a=0.8, b=1., l=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * a + img * b + l
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, l)



def extract_edges(image, roi_verteces):
    '''
        Given a ROI, return edges on image with same size as original
        Function filters image first (low pass) to avoid high frequency
        spurious line detection. Then edges are detected on the whole
        image. ROI is implemented AFTER - to avoid introducing artificially, high
        frequencies into the edge detection process.
    :param image: Raw Image
    :param roi_verteces: Eges image <x, y> ordered
    :return:
    '''

    min_x = 0
    min_y = 0
    ctr_x = int(math.floor(image.shape[1] / 2.))
    ctr_y = int(math.floor(image.shape[0] / 2.))
    max_x = int(image.shape[1])
    max_y = image.shape[0]

    # Convert to gray scale image
    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Edge detect (returns image with high pixels indicating edges)
    # 50 / 150
    low_threshold = 150
    high_threshold = 220
    edges = canny(blur_gray, low_threshold, high_threshold)

    # ****** Extract ROI
    # Important: extract ROI only AFTER the edges have been extracted.
    #            ROI extraction introduces high frequency components to the
    #            image. This ends up causing artificual edges at the ROI eges
    try:
        roi_edges = region_of_interest(edges, roi_verteces)
    except Exception as ex:
        print str(ex)
        raise Exception(ex)

    return roi_edges


def hough_lines_from_edges(roi_edges):
    '''
        Generate lines using Hough transforms on edge image
    :param roi_edges:
    :return:
    '''
    # **************** Raw Line Extraction
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    try:
        # Draw lines on the edge image. Obtain the segment line objects
        edge_line_image, lines = hough_lines(roi_edges, rho, theta, threshold,
                                             min_line_length, max_line_gap, thickness=4)
    except Exception as ex:
        print str(ex)
        raise Exception(ex)

    return edge_line_image, lines


def extract_lines_naiive_hough(image, roi_verteces):
    '''
        Extract lines from ROI in image. Function extracts
        edges in ROI using canny edge detection. Then using
        OpenCV's Hough transform, extract the lines
    :param image: Raw image to process
    :param roi_verteces: ROI definition
    :return: Image with lines drawn, line objects (line segment endpoints), edge pixels used for line extraction
    '''

    # 1 - Extract edges in roi
    roi_edges = extract_edges(image=image, roi_verteces=roi_verteces)

    edge_line_image, lines = hough_lines_from_edges(roi_edges)

    # Draw the lines on the raw image
    raw_image_with_edges = weighted_img(img = edge_line_image,
                                        initial_img= np.copy(image),
                                        a=0.8, b=1., l=0.)

    return raw_image_with_edges, edge_line_image, lines, roi_edges


def image_roi(image, polygon_apex):
    '''
    :param image: Input image
    :param polygon_apex: Listo of apex (1x2 tuples) of the polygon
    :return: ROI image and ROI indeces of the image as logical (bool) array
    '''
    xaxis_dim = 1
    yaxis_dim = 0

    nchannels = 1

    if len(image.shape) == 3:
        nchannels = 3
    elif len(image.shape) != 2:
        raise Exception('Incorrect image shape')

    left_bottom = polygon_apex[0]
    right_bottom = polygon_apex[1]
    apex = polygon_apex[2]

    ysize = image.shape[yaxis_dim]
    xsize = image.shape[xaxis_dim]

    roi_image = np.copy(image)

    # Gerti: Defining the line equations of the ROI triangle. Line std. slope-itcpt form
    # Fit lines (y=Ax+B) to identify the  3 sided region of interest
    # np.polyfit() returns the coefficients [A, B] of the fit

    # Equation: y = slope*x + line_itcpt
    slope_idc = 0
    line_itcpt_idc = 1
    # fit_left = np.polyfit((left_bottom[yaxis_dim], apex[yaxis_dim]),
    #                       (left_bottom[xaxis_dim], apex[xaxis_dim]), 1)
    # fit_right = np.polyfit((right_bottom[yaxis_dim], apex[yaxis_dim]),
    #                        (right_bottom[xaxis_dim], apex[xaxis_dim]), 1)
    # fit_bottom = np.polyfit((left_bottom[yaxis_dim], right_bottom[yaxis_dim]),
    #                         (left_bottom[xaxis_dim], right_bottom[xaxis_dim]), 1)

    fit_left = np.polyfit((left_bottom[xaxis_dim], apex[xaxis_dim]),
                          (left_bottom[yaxis_dim], apex[yaxis_dim]), 1)
    fit_right = np.polyfit((right_bottom[xaxis_dim], apex[xaxis_dim]),
                           (right_bottom[yaxis_dim], apex[yaxis_dim]), 1)
    fit_bottom = np.polyfit((left_bottom[xaxis_dim], right_bottom[xaxis_dim]),
                            (left_bottom[yaxis_dim], right_bottom[yaxis_dim]), 1)

    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX * fit_left[slope_idc] + fit_left[line_itcpt_idc])) & \
                        (YY > (XX * fit_right[slope_idc] + fit_right[line_itcpt_idc])) & \
                        (YY < (XX * fit_bottom[slope_idc] + fit_bottom[line_itcpt_idc]))

    roi_image[~region_thresholds] = np.zeros(nchannels).tolist();
    return roi_image, region_thresholds

