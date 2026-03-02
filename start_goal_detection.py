import cv2
import numpy as np
from pathlib import Path

# ==== DEFAULT CONFIG ====
DEFAULT_IMAGE_PATH = "out/cam2_birdseye.jpg"   # Bird's-eye image
DEFAULT_START_ID   = 13                    # ArUco ID for start
DEFAULT_GOAL_ID    = 45                    # ArUco ID for goal
DEFAULT_OUT_PREFIX = "cam2_run"            # Prefix for output files
# ================================================================


def detect_aruco_centers(gray, aruco_dict, aruco_params):
    """
    Detect ArUco markers and return:
      centers: dict {id: (cx, cy)}
      corners: list of 4-corner arrays
      ids:     np.array of marker IDs
    """
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    centers = {}
    if ids is not None:
        ids = ids.flatten()
        for i, cid in enumerate(ids):
            c = corners[i][0]  # shape (4,2)
            centers[int(cid)] = c.mean(axis=0)
    return centers, corners, ids


def annotate_and_save(bird_img, start_xy, goal_xy, out_path):
    """
    Draw start/goal on a copy of bird_img and save.
    Returns the annotated image (np.ndarray).
    """
    vis = bird_img.copy()

    if start_xy is not None:
        sx, sy = int(start_xy[0]), int(start_xy[1])
        cv2.circle(vis, (sx, sy), 8, (0, 255, 0), -1)
        cv2.putText(vis, "START", (sx + 10, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if goal_xy is not None:
        gx, gy = int(goal_xy[0]), int(goal_xy[1])
        cv2.circle(vis, (gx, gy), 8, (0, 0, 255), -1)
        cv2.putText(vis, "GOAL", (gx + 10, gy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(str(out_path), vis)
    return vis


def extract_start_goal(
    image_path: str = DEFAULT_IMAGE_PATH,
    start_id: int = DEFAULT_START_ID,
    goal_id: int  = DEFAULT_GOAL_ID,
    out_prefix: str = DEFAULT_OUT_PREFIX,
):
    """
    High-level function:
    1. Load the bird's-eye image.
    2. Detect all ArUco markers in that image.
    3. Find (x,y) for start_id and goal_id.
    4. Save:
        - annotated image with START/GOAL drawn
        - debug image with all markers + IDs
        - text file with coordinates
    5. RETURN a dict with everything useful.

    Returns (dict):
    {
        "start_id": int,
        "start_xy": (x,y) or None,
        "goal_id": int,
        "goal_xy": (x,y) or None,
        "all_centers": {marker_id: (cx,cy), ...},
        "all_ids": [ ... ] or [],
        "annotated_img_path": "cam2_run_with_points.png",
        "debug_img_path": "cam2_run_aruco_debug.png",
        "coords_txt_path": "cam2_run_start_goal_coords.txt"
    }
    """

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    bird = cv2.imread(str(img_path))
    if bird is None:
        raise RuntimeError(f"Failed to read image as BGR: {img_path}")

    gray = cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY)

    # Prepare aruco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = (
        cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "DetectorParameters")
        else cv2.aruco.DetectorParameters_create()
    )

    # Detect all ArUco markers
    centers, corners, ids = detect_aruco_centers(gray, aruco_dict, aruco_params)

    # Get specific markers (start / goal)
    start_xy = centers.get(start_id)
    goal_xy  = centers.get(goal_id)

    # Build output filenames (same folder as script run)
    out_txt  = Path(f"{out_prefix}_start_goal_coords.txt")
    out_img  = Path(f"{out_prefix}_with_points.png")
    debug_out = Path(f"{out_prefix}_aruco_debug.png")

    # Save annotated start/goal overlay image
    annotate_and_save(bird, start_xy, goal_xy, out_img)

    # Save debug overlay with all detected markers and their IDs
    if ids is not None:
        debug = bird.copy()
        cv2.aruco.drawDetectedMarkers(debug, corners, ids)
        # draw ID next to each center
        for mid, (cx, cy) in centers.items():
            cv2.putText(
                debug,
                str(mid),
                (int(cx) + 5, int(cy) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )
        cv2.imwrite(str(debug_out), debug)
    else:
        # no markers at all, but we still want predictable output
        debug = bird.copy()
        cv2.imwrite(str(debug_out), debug)

    # Save coordinates text file (for logging / downstream)
    with open(out_txt, "w") as f:
        f.write(f"start_id={start_id} start_xy={start_xy}\n")
        f.write(f"goal_id={goal_id} goal_xy={goal_xy}\n")

    # Prepare return payload
    result = {
        "start_xy": None if start_xy is None else (round(start_xy[1]), round(start_xy[0])),
        "goal_xy": None if goal_xy is None else (round(goal_xy[1]), round(goal_xy[0])),
    }

    return result