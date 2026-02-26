import cv2
import numpy as np
from pathlib import Path
import time

# ============== SETTINGS (edit if needed) ==============

CAM_URLS = {
    "cam1": "rtsp://SmartSystemsEngineering:IntroductionProject!@10.149.24.18:554/stream1",
    "cam2": "rtsp://SmartSystemsEngineering:IntroductionProject!@10.149.24.15:554/stream1",
}

UNDISTORT_ALPHA = 1   # 0..1
ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_50
MARKER_ID_ORDER = [0, 1, 2, 3]

WIDTH_CM, HEIGHT_CM = 474, 415

TL_LEFT_CM, TL_TOP_CM       = 67.0, 65.0   # ID0 = TL
TR_RIGHT_CM, TR_TOP_CM      = 54.0, 67.0   # ID1 = TR
BR_RIGHT_CM, BR_BOTTOM_CM   = 51.0, 90.0   # ID2 = BR
BL_LEFT_CM,  BL_BOTTOM_CM   = 68.0, 93.0   # ID3 = BL

SCALE_PX_PER_CM = 5
EXTRA_MARGIN_CM = 0
ROTATE = None
FLIP   = None

# =======================================================

SCRIPT_DIR = Path(__file__).parent
OUT_DIR    = SCRIPT_DIR / "out"
DATA_DIR   = SCRIPT_DIR / "data"
INTR_DIR   = SCRIPT_DIR

OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------ helpers ------------

def fix_orientation(frame):
    if ROTATE == "90cw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif ROTATE == "90ccw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif ROTATE == "180":
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    if FLIP == "h":
        frame = cv2.flip(frame, 1)
    elif FLIP == "v":
        frame = cv2.flip(frame, 0)
    return frame


def load_intrinsics(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    K = None
    for k in ["camera_matrix", "K", "mtx"]:
        if k in d.files:
            K = d[k].astype(float).reshape(3,3)
            break
    dist = None
    for k in ["dist_coeffs", "dist", "distCoeffs"]:
        if k in d.files:
            dist = d[k].astype(float).ravel().reshape(-1,1)
            break
    size = tuple(d["image_size"]) if "image_size" in d.files else None
    if K is None or dist is None:
        raise RuntimeError(f"Could not find K/dist in {npz_path}. Keys={list(d.files)}")
    return K, dist, size


def prepare_undistort_maps(K, dist, frame_size, calib_size=None, alpha=0.7):
    w, h = frame_size
    if calib_size and calib_size != (w, h):
        w0, h0 = calib_size
        sx, sy = w / w0, h / h0
        S = np.diag([sx, sy, 1.0])
        K = (S @ K)
        K[2, 2] = 1.0
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)
    return map1, map2


def detect_aruco_centers(gray, aruco_dict, aruco_params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    centers = {}
    if ids is not None:
        ids = ids.flatten()
        for i, cid in enumerate(ids):
            c = corners[i][0]
            centers[int(cid)] = c.mean(axis=0)
    return centers, corners, ids


def order_src_points_from_ids(centers, id_order):
    pts = []
    for i in id_order:
        if i not in centers:
            return None
        pts.append(centers[i])
    return np.array(pts, dtype=np.float32)


def draw_ids(frame, centers):
    for cid, p in centers.items():
        p = tuple(np.int32(p))
        cv2.circle(frame, p, 6, (0, 255, 0), -1)
        cv2.putText(frame, str(cid), (p[0] + 6, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)


def compute_table_dst_and_size_per_corner():
    TL = [TL_LEFT_CM,                    TL_TOP_CM]
    TR = [WIDTH_CM - TR_RIGHT_CM,        TR_TOP_CM]
    BR = [WIDTH_CM - BR_RIGHT_CM,        HEIGHT_CM - BR_BOTTOM_CM]
    BL = [BL_LEFT_CM,                    HEIGHT_CM - BL_BOTTOM_CM]
    pts_dst_cm = np.array([TL, TR, BR, BL], dtype=np.float32)
    out_w = int(round(WIDTH_CM  * SCALE_PX_PER_CM))
    out_h = int(round(HEIGHT_CM * SCALE_PX_PER_CM))
    pad_px = int(round(EXTRA_MARGIN_CM * SCALE_PX_PER_CM)) if EXTRA_MARGIN_CM > 0 else 0
    pts_dst_px = (pts_dst_cm * SCALE_PX_PER_CM).astype(np.float32) + np.array([pad_px, pad_px], np.float32)
    out_w += 2 * pad_px
    out_h += 2 * pad_px
    return pts_dst_px, (out_w, out_h)

# ------------ AUTO RUN CAMERA 2 ------------

def _capture_cam2_frame_undist():
    cam_name = "cam2"
    url = CAM_URLS[cam_name]

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"[{cam_name}] ERROR: cannot open stream")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"[{cam_name}] ERROR: could not read frame")

    frame = fix_orientation(frame)
    intrinsics_path = INTR_DIR / f"{cam_name}_intrinsics.npz"
    if intrinsics_path.exists():
        K, dist, calib_size = load_intrinsics(str(intrinsics_path))
        h0, w0 = frame.shape[:2]
        maps = prepare_undistort_maps(K, dist, (w0, h0), calib_size, alpha=UNDISTORT_ALPHA)
        frame_u = cv2.remap(frame, maps[0], maps[1],
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        return frame_u, K, dist
    else:
        return frame, None, None


def auto_run_cam2():
    cam_name = "cam2"
    frame_u, K, dist = _capture_cam2_frame_undist()

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
    aruco_params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") \
                   else cv2.aruco.DetectorParameters_create()

    gray = cv2.cvtColor(frame_u, cv2.COLOR_BGR2GRAY)
    centers, corners, ids = detect_aruco_centers(gray, aruco_dict, aruco_params)

    vis = frame_u.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    draw_ids(vis, centers)
    ann_path = OUT_DIR / f"{cam_name}_undist_aruco.jpg"
    cv2.imwrite(str(ann_path), vis)

    pts_src = order_src_points_from_ids(centers, MARKER_ID_ORDER)
    if pts_src is None:
        raise RuntimeError(f"[{cam_name}] Can't compute homography. Need all IDs {MARKER_ID_ORDER} visible.")

    pts_dst_px, OUTPUT_SIZE = compute_table_dst_and_size_per_corner()
    out_w, out_h = OUTPUT_SIZE
    H, _ = cv2.findHomography(pts_src, pts_dst_px, method=cv2.RANSAC)
    if H is None:
        raise RuntimeError(f"[{cam_name}] Homography failed (RANSAC None).")

    bird = cv2.warpPerspective(frame_u, H, (out_w, out_h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    out_img = OUT_DIR / f"{cam_name}_birdseye.jpg"
    cv2.imwrite(str(out_img), bird)

    np.savez(OUT_DIR / f"{cam_name}_homography_aruco.npz",
             H=H,
             ids=np.array(MARKER_ID_ORDER),
             output_size=np.array([out_w, out_h]),
             mode="TABLE",
             scale_px_per_cm=SCALE_PX_PER_CM,
             table_size_cm=np.array([WIDTH_CM, HEIGHT_CM]),
             tl=np.array([TL_LEFT_CM, TL_TOP_CM], dtype=float),
             tr=np.array([TR_RIGHT_CM, TR_TOP_CM], dtype=float),
             br=np.array([BR_RIGHT_CM, BR_BOTTOM_CM], dtype=float),
             bl=np.array([BL_LEFT_CM, BL_BOTTOM_CM], dtype=float),
             extra_margin_cm=EXTRA_MARGIN_CM)

    print(f"[{cam_name}] ✅ Done.")
    print(f"  -> Annotated detections: {ann_path.name}")
    print(f"  -> Bird's-eye image:     {out_img.name}")
    print(f"  -> Homography file:      {cam_name}_homography_aruco.npz")
    print(f"  -> Size: {out_w}x{out_h} px  |  Scale: {SCALE_PX_PER_CM} px/cm")

# ------------ entry point ------------

#def main():
auto_run_cam2()
