import cv2
import config as cfg
import numpy as np

intrinsic_data_folder =""
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1e-4)
cols, rows = (6,9)
cell_size = 10
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
objp *= cell_size
objpoints = []
imgpoints = []
imgs_paths = glob(path.join(intrinsic_data_folder, '*.png'))
h, w = cv2.imread(imgs_paths[0]).shape[:2]
for fname in imgs_paths:
img = cv2.imread(fname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the chess board corners
ret, corners = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_ACCURACY)
# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    imgpoints.append(corners)
_, camera_matrix, dist, _, _, _, _, _ = cv2.calibrateCameraExtended(
objpoints,
imgpoints, (w, h),
None,
None,
flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_RATIONAL_MODEL,
criteria=criteria)
np.save(cfg.M_int_path, camera_matrix)

print('calibration completed')
print(camera_matrix)