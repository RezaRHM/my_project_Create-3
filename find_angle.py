import cv2
import numpy as np

ROBOT_ID = 13  # ArUco ID attached to the robot
IMAGE_PATH = "out/cam2_birdseye.jpg"  # latest bird's-eye image


def get_robot_heading_deg(image_path=IMAGE_PATH, robot_id=ROBOT_ID):
    """
    Returns:
        heading_deg (float): robot's heading angle in degrees,
                             relative to +X axis of the bird's-eye image,
                             normalized to [-180, 180].

    Raises:
        FileNotFoundError: if the image cannot be loaded.
        RuntimeError: if the robot's ArUco marker is not detected.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ArUco detection setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()

    # Detect markers
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is None:
        raise RuntimeError("No ArUco markers detected in image.")

    ids = ids.flatten()

    # Find the robot's marker
    for i, cid in enumerate(ids):
        if int(cid) == int(robot_id):
            pts = corners[i][0]  # shape (4,2): [tl, tr, br, bl]
            p0 = pts[0]          # top-left corner
            p1 = pts[1]          # top-right corner

            # Orientation vector of the marker's local +X axis
            vx = p1[0] - p0[0]
            vy = p1[1] - p0[1]

            angle_deg = np.degrees(np.arctan2(vy, vx))

            # Normalize angle to [-180, 180]
            while angle_deg > 180:
                angle_deg -= 360
            while angle_deg < -180:
                angle_deg += 360

            return float(angle_deg)

    raise RuntimeError(f"Robot marker (ID={robot_id}) not found in image.")