import apriltag
import numpy as np
import cv2
import pyrealsense2 as rs
from calibrate import calibration
import config as cfg
from glob import glob
import os
from os import path
from scipy.optimize import leastsq
import math
import argparse

pipeline = (
    rs.pipeline()
)  # <- Объект pipeline содержит методы для взаимодействия с потоком
config = rs.config()  # <- Дополнительный объект для хранения настроек потока
colorizer = rs.colorizer()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
profile = pipeline.start(config)
detector = apriltag.Detector()
M_int = np.load(cfg.M_int_path)
vs = cv2.VideoCapture(6)
camera_params = [M_int[0][0], M_int[1][1], M_int[0][-1], M_int[1][-1]]
Rt_cam2eye = None


def opt_func(optimized_parameters, y_true, xyz_orig):
    fx, fy, cx, cy = optimized_parameters[:4]
    M_int = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    rvec = optimized_parameters[4:7]
    tvec = optimized_parameters[7:]
    tvec = np.array(tvec)
    R = cv2.Rodrigues(rvec)[0]
    Rt = np.append(R, np.expand_dims(tvec, 1), axis=1)
    optimized_MRt = M_int.dot(Rt)
    points = np.array([optimized_MRt.dot(point) for point in xyz_orig])
    pixels = np.array([[u / w, v / w] for u, v, w in points])

    return np.linalg.norm((y_true - pixels), axis=1)


def read_data(camera_params):
    fx, fy, cx, cy = camera_params[:4]
    camera_matrix = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    images_path = glob(path.join(cfg.cam2eye_data_folder, "*.png"))
    Rt_foreach = {}
    data = {}
    XYZ = {}
    Coordinates_path = glob(path.join(cfg.cam2eye_data_folder, "*.txt"))
    RSCoordinates_path = glob(path.join(cfg.cam2eye_data_folder, "*rs.txt"))
    for img_path in images_path:
        img = cv2.imread(img_path, 0)
        detect_result = detector.detect(img)
        if len(detect_result) != 0:

            pse = detector.detection_pose(detect_result[0], camera_params, tag_size=11)
            pose_params = np.asanyarray(pse)
            Rt = pose_params[0][:3]
            filename = path.splitext(path.basename(img_path))[0]

            Rt_foreach[filename] = camera_matrix.dot(Rt)
    for coord_file in RSCoordinates_path:
        with open(coord_file) as f:
            filename = path.splitext(path.basename(coord_file))[0]
            filename = filename.replace("_rs", "")
            Points = f.read()

            Points = (
                Points.replace("x=", "")
                .replace("y=", "")
                .replace("z=", "")
                .replace(",", "")
                .split("\n")
            )

            Points = Points[0].split(" ")

            Points = [float(Points[0]), float(Points[1]), float(Points[2]), 1.0]
            XYZ[filename] = Points
    for coord_file in Coordinates_path:
        if coord_file in RSCoordinates_path:
            continue
        else:
            with open(coord_file) as f:
                filename = path.splitext(path.basename(coord_file))[0]
                Points = f.read()

                Points = Points.replace("x=", "").replace("y=", "").split("\n")
                print(coord_file)
                Points = [int(Points[0].split(" ")[0]), int(Points[0].split(" ")[1])]

                if filename in Rt_foreach:
                    if filename in XYZ:
                        assert filename not in data
                        Rt = Rt_foreach[filename]
                        xyz = XYZ[filename]
                        data[filename] = [Rt, Points, xyz]
    return data


def optimizer(data, camera_params):
    M_RTpoints = []
    y_true = []
    xyz_orig = []
    for RTpoint, pixel, xyz in data.values():
        M_RTpoints.append(RTpoint)
        y_true.append(pixel)
        xyz_orig.append(xyz)
    xyz_orig = np.array(xyz_orig)
    y_true = np.array(y_true)
    M_RTpoints = np.array(M_RTpoints)
    rvec = [0.003, 0.003, 0.003]
    tvec = [0.003, 0.003, 0.003]
    camera_params
    init = [*camera_params, *rvec, *tvec]
    x, _ = leastsq(
        opt_func,
        init,
        args=(y_true, xyz_orig),
        ftol=1e-8,
        gtol=1e-8,
        maxfev=100000,
    )

    fx, fy, cx, cy = x[:4]
    m_int = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    rvec = x[4:7]
    tvec = x[7:]
    np.save(cfg.rvec_cam2eye, rvec)
    np.save(cfg.tvec_cam2eye, tvec)
    R = cv2.Rodrigues(rvec)[0]

    Rt = np.hstack((R, tvec.reshape(3, 1)))

    residuals = opt_func(x, y_true, xyz_orig)

    MRT_optimize = m_int.dot(Rt)
    print("Mean error: {}".format(residuals.mean()))
    np.save(cfg.MRt_cam2eye, MRT_optimize)
    return MRT_optimize


def coordinate_shift(coordinates, depth_point, Rt_cam2eye):
    x = int(coordinates[0])
    y = int(coordinates[1])
    depth = depth_frame.get_distance(int(depth_point[0]), int(depth_point[1]))
    points = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
    UVW = Rt_cam2eye.dot(np.hstack((points, 1.0)))
    pixel = (UVW / UVW[-1])[:-1]
    return pixel


def show(rs_frame, cam_frame, coordinates, newX, newY):
    cv2.rectangle(
        rs_frame,
        (int(coordinates[0]), int(coordinates[1])),
        (int(coordinates[0]) + 3, int(coordinates[1]) + 3),
        (255, 30, 0),
        4,
    )
    cv2.rectangle(
        cam_frame,
        (int(newX), int(newY)),
        (int(newX) + 3, int(newY) + 3),
        (255, 30, 0),
        4,
    )


ap = argparse.ArgumentParser()
# ap.add_argument('-p', '--prototxt', default='/Users/siddhantbansal/Desktop/Python/Personal_Projects/Object_Detection/MobileNetSSD_deploy.prototxt.txt', help='path to Caffe deploy prototxt file')
# ap.add_argument('-m', '--model', default='/Users/siddhantbansal/Desktop/Python/Personal_Projects/Object_Detection/MobileNetSSD_deploy.caffemodel', help='path to the Caffe pre-trained model')
ap.add_argument(
    "-p", "--prototxt", required=True, help="path to Caffe deploy prototxt file"
)
ap.add_argument(
    "-m", "--model", required=True, help="path to the Caffe pre-trained model"
)
ap.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=0.2,
    help="minimum probability to filter weak detections",
)
args = vars(ap.parse_args())
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
h, w = 480, 640
cc = 0
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    rs_frame = np.asanyarray(color_frame.get_data())
    gray_rs_frame = cv2.cvtColor(rs_frame, cv2.COLOR_BGR2GRAY)
    _, cam_frame = vs.read()
    key = cv2.waitKey(1) & 0xFF
    gray_rs_frame = cv2.cvtColor(rs_frame, cv2.COLOR_BGR2GRAY)
    result = detector.detect(gray_rs_frame)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    # Intrinsics & Extrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
    color_to_depth_extrin = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_extrinsics_to(profile.get_stream(rs.stream.depth))
    )

    blob = cv2.dnn.blobFromImage(
        cv2.resize(rs_frame, (300, 300)), 0.007843, (300, 300), 127.5
    )

    # pass the blob through the neural network
    if cc == 20:
        net.setInput(blob)
        detections = net.forward()
        cc = 0
    cc += 1
    """Load data"""
    if key == ord("l"):
        Rt_cam2eye = np.load(cfg.MRt_cam2eye)
        print("success load")
    """Calculate data"""
    if key == ord("c"):
        data = read_data(camera_params)
        Rt_cam2eye = optimizer(data, camera_params)
    if Rt_cam2eye is not None:
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., the probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the classes label from the 'detections',
                # then compute the (x, y)-coordinates of the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

                cv2.rectangle(rs_frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    rs_frame,
                    label,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLORS[idx],
                    2,
                )

                center = int(startX + (endX - startX) / 2), int(
                    startY + (endY - startY) / 2
                )
                depth_point = rs.rs2_project_color_pixel_to_depth_pixel(
                    depth_frame.get_data(),
                    depth_scale,
                    0,
                    5.0,
                    depth_intrin,
                    color_intrin,
                    depth_to_color_extrin,
                    color_to_depth_extrin,
                    [center[0], center[1]],
                )
                newX1, newY1 = coordinate_shift(
                    [startX, startY], depth_point, Rt_cam2eye
                )
                newX2, newY2 = coordinate_shift([endX, endY], depth_point, Rt_cam2eye)
                cv2.rectangle(
                    cam_frame,
                    (int(newX1), int(newY1)),
                    (int(newX2), int(newY2)),
                    COLORS[idx],
                    2,
                )

    both_frames = np.hstack((rs_frame, cam_frame))
    cv2.imshow("window", both_frames)
    if key == ord("q"):
        break