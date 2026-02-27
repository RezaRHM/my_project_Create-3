# Get the top-down view of the maze
import cv2 as cv
import numpy as np


# Spot removal configuration: (x_offset_from_edge, y_offset_from_edge, diameter, reference_corner)
UNWANTED_SPOTS = [
    {'x_from': 'right', 'x_offset': 200, 'y_from': 'bottom', 'y_offset': 295, 'diameter': 32},
    {'x_from': 'right', 'x_offset': 2080, 'y_from': 'bottom', 'y_offset': 320, 'diameter': 34},
    {'x_from': 'left', 'x_offset': 500, 'y_from': 'top', 'y_offset': 0, 'diameter': 49}
]


def cover_unwanted_spots(image, spot_config):
    result = image.copy()
    h, w = image.shape

    for spot in spot_config:
        # Calculate absolute position from relative reference
        if spot['x_from'] == 'right':
            x = w - spot['x_offset']
        else:
            x = spot['x_offset']

        if spot['y_from'] == 'bottom':
            y = h - spot['y_offset']
        else:  # 'top'
            y = spot['y_offset']

        radius = int(spot['diameter'] * 3.5)
        cv.circle(result, (int(x), int(y)), radius, 255, -1)

    return result


# Morphology
def erosion(maze, white_path: bool = True, kernel_size: int = 3, iterations: int = 1):
    if white_path:
        maze = cv.bitwise_not(maze)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv.erode(maze, kernel, iterations=iterations)
    if white_path:
        erosion = cv.bitwise_not(erosion)
    return erosion


def inflation(maze, white_path: bool = True, kernel_size: int = 3, iterations: int = 1):
    if white_path:
        maze = cv.bitwise_not(maze)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    inflation = cv.dilate(maze, kernel, iterations=iterations)
    if white_path:
        inflation = cv.bitwise_not(inflation)
    return inflation


# Maze Image Processing: Feel free to add, remove, or modify any layer
def maze_image_processing(maze=None, path=None):
    # 1. Read the image
    if maze is None and path:
        maze = cv.imread(path, cv.IMREAD_COLOR)

    # 2. Transform to GrayScale
    maze_gray = cv.cvtColor(maze, cv.COLOR_BGR2GRAY)

    # Experimental Layer - Dilation
    maze_inf = inflation(maze_gray, white_path = False, kernel_size = 3, iterations = 2)

    # 3. Contrast Stretching
    clahe = cv.createCLAHE(clipLimit = 2, tileGridSize=(8, 8))
    clahe_contrast = clahe.apply(maze_inf)

    # 4. Blurring
    m_blur_cl = cv.medianBlur(clahe_contrast, 3)

    # 5. Binarization
    mean_median_cl = cv.adaptiveThreshold(m_blur_cl, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 3)

    # 6. Blurring
    median_mean_median_cl = cv.medianBlur(mean_median_cl, 7)

    # 7. Removing Aruco Markers
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()

    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(maze)

    if corners:
        for corner in corners:
            corner = corner.reshape((4, 2))
            center = corner.mean(axis=0)

            # Expand corners outward from center
            margin_ratio = 1.1  # Expand by 110%
            expanded_corner = []
            for point in corner:
                direction = point - center
                expanded_point = center + direction * (1 + margin_ratio)
                expanded_corner.append(expanded_point)

            expanded_corner = np.array(expanded_corner, dtype=np.int32)
            cv.fillPoly(median_mean_median_cl, [expanded_corner], (255, 255, 255))

    # 8. Covering Spots
    maze_covered = cover_unwanted_spots(median_mean_median_cl, UNWANTED_SPOTS)

    # 9. Morphology: Opening
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    maze_cleaned = cv.morphologyEx(maze_covered, cv.MORPH_OPEN, kernel, iterations = 5)

    # 10. Morphology: erosion followed by dilation with different kernel sizes and iterations
    maze_cleaned_ero = erosion(maze_cleaned, white_path = True, kernel_size=5, iterations=3)
    maze_cleaned_ero_inf = inflation(maze_cleaned_ero, white_path = True, kernel_size=9, iterations=7)

    # 11. Adding 1-pixel width padding for A* optimization
    maze_padded = cv.copyMakeBorder(maze_cleaned_ero_inf, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, value=0)

    return maze_padded
