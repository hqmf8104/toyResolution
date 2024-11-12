' spatial functions contains functions needed to turn spatial info from Yolov8 into red dots on maps'
import numpy as np
import cv2
import tkinter

def F1(box, model, frame):
    """ 
    F1: GET DETECTED OBJECT DETAILS'
    Takes the results from a yolo model run over a single frame and the model itself and returns the x,y coords, label and confidence level

    results = model.predict(frame)
    for result in results:
        for box in result.boxes
    
    
     a. get most recent detected object set'
       i. coordinates, class label'
       ii. coordimate transformation'
       iii. colour'
    """
    #----------------------------------------------------------------------------------------------------
    # a.i.
    #----------------------------------------------------------------------------------------------------
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
    label = model.names[int(box.cls)]  # Get the class label
    confidence = box.conf[0]  # Confidence score
    #****************************************************************************************************

    #----------------------------------------------------------------------------------------------------
    # a.ii.
    #---------------------------------------------------------------------------------------------------- 
    # Get webcam data
    pixelHeight, pixelWidth, _ = frame.shape


    K = homographyMatrix(pixelWidth,pixelHeight,[0,0],[500,0],[0,300],[500,300])
    actualCentre = pixelToRealCoord(((x1 + x2) // 2, (y1 + y2) // 2),K)  # Object center point in pixel coords
    actualCentre = ((x1 + x2) // 2, (y1 + y2)) 
    #****************************************************************************************************
    return actualCentre[0], actualCentre[1], label, confidence, x1, y1, x2, y2
    
def homographyMatrix(pixelWidth, pixelHeight, bottomLeft, bottomRight, topLeft, topRight):
    """
    Computes a homography matrix to map pixel coordinates in an image to real-world coordinates.
    
    This function takes the dimensions of an image and maps its corners to corresponding real-world
    coordinates using a homography transformation. 

    Parameters:
    - pixelWidth (int or float): The width of the image in pixels.
    - pixelHeight (int or float): The height of the image in pixels.
    - bottomLeft (list of float): The (x, y) real-world coordinates of the bottom-left corner.
    - bottomRight (list of float): The (x, y) real-world coordinates of the bottom-right corner.
    - topLeft (list of float): The (x, y) real-world coordinates of the top-left corner.
    - topRight (list of float): The (x, y) real-world coordinates of the top-right corner.

    Returns:
    - homography_matrix (numpy.ndarray): A 3x3 homography matrix that maps pixel coordinates to
      real-world coordinates based on the specified correspondences.
    
    Usage:
    - This matrix can be used to transform pixel coordinates to the real-world plane, assuming
      the input image plane is a flat, planar surface being observed.
    """
 
    # Image dimensions (you can replace these with actual dimensions of your image)
    image_width = pixelWidth  # Width of the image in pixels
    image_height = pixelHeight  # Height of the image in pixels

    # Define points in the image plane (in pixels)
    image_points = np.array([
        [0, image_height],            # Bottom-left
        [image_width, image_height],  # Bottom-right
        [image_width, 0],            # Top-right
        [0, 0]                       # Top-left
    ], dtype=np.float32)


    
    world_points = np.array([
        bottomLeft,          # Bottom-left
        bottomRight,        # Bottom-right
        topRight,      # Top-right
        topLeft         # Top-left
    ], dtype=np.float32)

    # Compute the homography matrix
    homography_matrix, _ = cv2.findHomography(image_points, world_points)

    return homography_matrix

# Function to transform a point from image to world coordinates
def pixelToRealCoord(image_point, homography_matrix, scaleFactor = 1):
    """
    Transforms a point from image (pixel) coordinates to real-world coordinates using a given homography matrix.
    
    This function takes an image point in pixel coordinates, applies a homography transformation, and
    returns the corresponding coordinates in the real-world plane. Optionally, a scaling factor can be 
    applied to the output coordinates to adjust for unit conversions or other scaling requirements.

    Parameters:
    - image_point (tuple or list of float): The (x, y) coordinates of the point in the image plane.
    - homography_matrix (numpy.ndarray): A 3x3 matrix representing the homography transformation 
      from the image plane to the real-world plane.
    - scaleFactor (float, optional): A scaling factor to be applied to the transformed coordinates.
      Default is 1 (no scaling).

    Returns:
    - numpy.ndarray: A 1D array containing the (x, y) coordinates of the transformed point in 
      the real-world plane, scaled by the provided scaleFactor.

    Usage:
    - This function assumes that the image point is represented in (x, y) format and applies a 
      homogeneous coordinate transformation followed by a normalization step to obtain Cartesian 
      coordinates.
    """
    point = np.array([image_point[0], image_point[1], 1.0], dtype=np.float32).reshape(3, 1)
    # Apply the homography matrix
    transformed_point = np.dot(homography_matrix, point)
    # Convert from homogenous coordinates to Cartesian coordinates
    transformed_point /= transformed_point[2]
    return scaleFactor * transformed_point[0:2].flatten()