#detects pose, nose location using facemerk, sends prev pose if no head pose/nose detected in current frame and uses Kalman filter to track

import os
import cv2
import sys
sys.path.append('..')


import numpy as np
from math import *
from math import cos, sin
# from moviepy.editor import *
from lib.FSANET_model import *
#from mtcnn.mtcnn import MTCNN
import cv2.aruco as aruco
import tensorflow as tf
# add to the top of your code under import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.test.is_gpu_available()
tf.config.experimental.list_physical_devices('GPU')
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
tf.debugging.set_log_device_placement(True)

from keras.layers import Average
from keras.models import Model
import pyrealsense2 as rs
import face_alignment
#from mlxtend.image import extract_face_landmarks
import dlib
from imutils import face_utils
import socket
import pickle
import transforms3d
import struct
from scipy.spatial.transform import Rotation as R
from arucocalibclass import arucocalibclass
from numpy.linalg import inv
from KalmanFilter import KalmanFilter
import time
##from tensorflow.python.client import device_lib
##print(device_lib.list_local_devices())
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=80):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll)
                 * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch)
                 * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def draw_results_ssd(detected, input_img, faces, ad, img_size, img_w, img_h, model, time_detection, time_network, time_plot):
    RotMat = [[0,0,0],[0,0,0],[0,0,0]]
    euler_angles = [0,0,0]
    if len(detected) > 0:
        for i, d in enumerate(detected):
            #x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detected[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                (h0, w0) = input_img.shape[:2]
                box = detected[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
                (startX, startY, endX, endY) = box.astype("int")
                # print((startX, startY, endX, endY))
                x1 = startX
                y1 = startY
                w = endX - startX
                h = endY - startY

                x2 = x1 + w
                y2 = y1 + h

                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)

                faces[i, :, :, :] = cv2.resize(
                    input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces[i, :, :, :] = cv2.normalize(
                    faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                face = np.expand_dims(faces[i, :, :, :], axis=0)
                p_result = model.predict(face)

                faceSq = face.squeeze()
                img = draw_axis(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :],
                                p_result[0][0], p_result[0][1], p_result[0][2])
                #print("euler", p_result[0][0], p_result[0][1], p_result[0][2])
                # note the change in sign  of middle term
                r = R.from_euler('zyx', [p_result[0][0],-p_result[0][1],p_result[0][2]], degrees=True)
                euler_angles = [p_result[0][0], -p_result[0][1], p_result[0][2]]
                # rotation matrix
                #RotMat = r.as_matrix()
                input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img

    cv2.imshow("result", input_img)
    
    return input_img, euler_angles # ,time_network,time_plot

def ConvertToArucoSys(RotMat, Posarr):
    # matrix to transform FSA pose into Aruco space
    RotConvAruco = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
    RotConvAruco = RotConvAruco.reshape(3, 3)
    RotMatTemp = np.matmul(RotMat, RotConvAruco)
    RotMat = np.matmul(inv(RotConvAruco), RotMatTemp)
    # now combine the rotation matrix with the Pos array
    RSFSATr = np.hstack([RotMat, np.transpose(Posarr)])
    RSFSATr = np.vstack([RSFSATr, [0, 0, 0, 1]])
    return RSFSATr

def main():
    # create a socket object if socket_connect = 1
    # flag set to 1 if testing with MagicLeap, else set to 0 if testing only the FSANet code with Realsense camera
    socket_connect = 1
    kalman_flag = 1
    arucoposeflag = 1
    N_samples = 10 # number of samples for computing transformation matrix using homography

    if socket_connect == 1:
        s = socket.socket()
        print ("Socket successfully created")
        # reserve a port on your computer - for example 2020 - but it can be anything
        port = 2020
        s.bind(('', port))
        print ("socket binded to %s" %(port))

        # put the socket into listening mode
        s.listen(5)
        print ("socket is listening")
        c,addr = s.accept()
        print('got connection from ',addr)

        #if socket_connect = 1, call the aruco calibration instance and claibrate with MagicLeap

        arucoinstance = arucocalibclass()
        ReturnFlag = 1
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        markerLen = 0.0645
        MLRSTr = arucoinstance.startcamerastreaming(c, ReturnFlag, markerLen, aruco_dict, N_samples)
        print(MLRSTr)
    else:
      MLRSTr = np.array((1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1))
      MLRSTr = MLRSTr.reshape(4,4)
      print(MLRSTr)

    # FSANet related params
    K.set_learning_phase(0)  # make sure its testing mode
    # face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    #detector = MTCNN()
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detector",
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # kalman filter initialization
    stateMatrix = np.zeros((12, 1), np.float32)  # [x, y, delta_x, delta_y]
    estimateCovariance = np.eye(stateMatrix.shape[0])
    transitionMatrix = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
    processNoiseCov = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], np.float32) * 0.001
    measurementStateMatrix = np.zeros((6, 1), np.float32)
    observationMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], np.float32)
    measurementNoiseCov = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32) * 1
    kalman = KalmanFilter(X=stateMatrix,
                          P=estimateCovariance,
                          F=transitionMatrix,
                          Q=processNoiseCov,
                          Z=measurementStateMatrix,
                          H=observationMatrix,
                          R=measurementNoiseCov)


    # load model and weights
    img_size = 64
    stage_num = [3, 3, 3]
    lambda_local = 1
    lambda_d = 1
    img_idx = 0
    img_sent = 0
    reinit_thresh = 10
    proc_frame_count = 0
    detected = ''  # make this not local variable
    time_detection = 0
    time_network = 0
    time_plot = 0
    skip_frame_detect = 2
    skip_frame_send = 4  # every 5 frame do 1 detection and network forward propagation
    skip_frame_reinit = 120 #after every 150 frames, reinitialize detection
    ad = 0.6

    # Parameters
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3, 3, 3]
    lambda_d = 1
    num_classes = 3
    image_size = 64
    num_primcaps = 7*3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model1 = FSA_net_Capsule(image_size, num_classes,
                             stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(
        image_size, num_classes, stage_num, lambda_d, S_set)()

    num_primcaps = 8*8*3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model3 = FSA_net_noS_Capsule(
        image_size, num_classes, stage_num, lambda_d, S_set)()

    print('Loading models ...')

    weight_file1 = '../pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    model1.load_weights(weight_file1)
    print('Finished loading model 1.')

    weight_file2 = '../pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    model2.load_weights(weight_file2)
    print('Finished loading model 2.')

    weight_file3 = '../pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model3.load_weights(weight_file3)
    print('Finished loading model 3.')

    inputs = Input(shape=(64, 64, 3))
    x1 = model1(inputs)  # 1x1
    x2 = model2(inputs)  # var
    x3 = model3(inputs)  # w/o
    avg_model = Average()([x1, x2, x3])
    model = Model(inputs=inputs, outputs=avg_model)

    # Configure depth and color streams of Realsense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    #load lbp face cascde for face detection followed by facemark for getting nose coordinates
    lbp_face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")#"/home/supriya/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel('/home/supriya/supriya/FSA-Net/demo/lbfmodel.yaml')
        
    print('Start detecting pose ...')
    detected_pre = []

    ArrToSendPrev = np.array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])
    depth_point = [0,0,0]
    start_time = time.time()
    #start the camera streams

    while True:
        try:
            start_time_rs = time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # Intrinsics & Extrinsics of camera streams
            depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(
                color_frame.profile)

            input_img = np.asanyarray(color_frame.get_data())
            img_idx = img_idx + 1
            #print(img_idx, time.time() - start_time)
            img_h, img_w, _ = np.shape(input_img)
            #print("realsense streaming time ", time.time() - start_time_rs )
            #do the following for first frame and every frame after skip_frames
            if img_idx == 1 or img_idx % skip_frame_detect == 0:
                start_time_frame = time.time()
                time_detection = 0
                time_network = 0
                time_plot = 0

                # detect faces using LBP detector
                gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
                                             (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detected = net.forward()

                depth_point = [0,0,0]

                #OPenCV face detection and landmark detection
                if len(detected_pre) > 0 and len(detected) == 0:
                    detected = detected_pre
                face_detect_start = time.time()
                faces = np.empty((len(detected), img_size, img_size, 3))
                face_detect_time = time.time() - face_detect_start
                ##print("face detection time: ", face_detect_time)
                facemark_start = time.time()
                facesLBP = lbp_face_cascade.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 5)
                facemark_time = time.time() - facemark_start
                ##print("facemark time", facemark_time)

                #detect facial landmarks using OpenCV's facemerk and deproject the nose coordinates to the 3D space
                if len(facesLBP) > 0:
                   #start_time_frame = time.time()
                   status, landmarks = facemark.fit(gray_img, facesLBP)
                   cv2.circle(input_img, (landmarks[0][0][30][0],landmarks[0][0][30][1]), 3, (0, 0, 255), -1)
                   depth = aligned_depth_frame.get_distance(landmarks[0][0][30][0],landmarks[0][0][30][1] )
                   depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [landmarks[0][0][30][0],landmarks[0][0][30][1]], depth)
                #print("nose coord", depth_point)
                #translation vector is the translation vector of the nose location in 3D space
                Posarr = np.array([depth_point[0], depth_point[1], depth_point[2]], np.float32)
                Posarr = Posarr.reshape(1, 3)
                FSAnet_start = time.time()
                #FSANet head pose detection
                input_img, euler_angles = draw_results_ssd(
                    detected, input_img, faces, ad, img_size, img_w, img_h, model, time_detection, time_network, time_plot)
                FSANet_time = time.time() - FSAnet_start
                ##print("FSANet time", FSANet_time)
               
                # Send the transformed pose to the AR Headset only if the person detected is within 0.8 m (to avoid) wrongly detecting the pose of someone else in the room 
                # other than the subject we want to track.
                if ((depth_point[2]!=0 and depth_point[2]< 0.8) and (euler_angles[0]!=0 and euler_angles[1] !=0)):

                    ##print("img_idx img_sent", img_idx, img_sent)
                    if kalman_flag == 0 or img_idx - img_sent > reinit_thresh or img_idx % skip_frame_reinit == 0:
                        print("reinit", img_idx, skip_frame_reinit)
                        Posarr = np.array([depth_point[0], depth_point[1], depth_point[2]])
                        Posarr = Posarr.reshape(1, 3)
                        print("Posarr", Posarr)
                        r = R.from_euler('zyx', euler_angles, degrees=True)
                        RotMat = r.as_matrix()
                        img_sent = img_idx


                    else:
                        # execute the following if kalman filter to be applied

                        current_measurement = np.array(
                            [np.float32(depth_point[0]), np.float32(depth_point[1]), np.float32(depth_point[2]),
                             np.float32(euler_angles[0]), np.float32(euler_angles[1]),
                             np.float32(euler_angles[2])]).reshape([1, 6])
                        current_prediction = kalman.predict()
                        current_prediction = np.array(current_prediction, np.float32)
                        current_prediction = current_prediction.transpose()[0]
                        # predicted euler angles
                        euler_angles_P = current_prediction[3:6]
                        # predicted posarr
                        Posarr_P = np.array(current_prediction[:3]).reshape([1, 3])
                        # convert to rotation matrix using r function
                        r = R.from_euler('zyx', euler_angles_P, degrees=True)
                        RotMat = r.as_matrix()
                        # update the estimate of the kalman filter
                        kalman.correct(np.transpose(current_measurement))
                        Posarr_noKalman = np.array([depth_point[0], depth_point[1], depth_point[2]])
                        Posarr = Posarr_P
                        print("Posarr_P", Posarr, Posarr_noKalman, euler_angles, euler_angles_P)

                    # transform the FSANet coord sytem to OpenCV coordinate system (since calibration was done between OpenCV coordinate system and FSANet coordinate system
                    RSFSATr = ConvertToArucoSys(RotMat, Posarr)
                    RSTr = RSFSATr

                    # If arucoposeflag = 1, an Aruco marker will get detected and its transformed pose will be streamed to the AR headset and the pose
                   # of the tracking parent will be updated to reflect Aruco marker pose. This can be used to verify/test the accuracy of teh calibration

                    if arucoposeflag == 1:
                                 print("aruco")
                                 gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                                 # set dictionary size depending on the aruco marker selected
                                 aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
                                 # detector parameters can be set here (List of detection parameters[3])
                                 parameters = aruco.DetectorParameters_create()
                                 parameters.adaptiveThreshConstant = 10
                                 # lists of ids and the corners belonging to each id
                                 corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                                 # font for displaying text (below)
                                 font = cv2.FONT_HERSHEY_SIMPLEX
                                 # check if the ids list is not empty
                                 # if no check is added the code will crash
                                 if np.all(ids != None):
                                   # estimate pose of each marker and return the values
                                   intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()#profile.as_video_stream_profile().get_intrinsics()
                                   mtx = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]])
                                   dist = np.array(intr.coeffs)
                                   rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.045, mtx, dist)
                                   for i in range(0, ids.size):
                                     # draw axis for the aruco markers
                                     aruco.drawAxis(input_img, mtx, dist, rvec[i], tvec[i], 0.1)
                                   # draw a square around the markers
                                   aruco.drawDetectedMarkers(input_img, corners)
                                   R_rvec = R.from_rotvec(rvec[0])
                                   R_rotmat = R_rvec.as_matrix()
                                   RSTr = np.hstack([R_rotmat[0],tvec[0].transpose()])
                                   RSTr = np.vstack([RSTr,[0,0,0,1]])

                    # Since pose detected by FSANet will be right handed coordinate system, it needs to be converted to left-handed coordinate system of Unity
                    RSTr_LH = np.array([RSTr[0][0],RSTr[0][2],RSTr[0][1],RSTr[0][3],RSTr[2][0],RSTr[2][2],RSTr[2][1],RSTr[2][3],RSTr[1][0],RSTr[1][2],RSTr[1][1],RSTr[1][3],RSTr[3][0],RSTr[3][1],RSTr[3][2],RSTr[3][3]])# converting to left handed coordinate system
                    RSTr_LH = RSTr_LH.reshape(4,4)
                    # compute transformed pose to send to MagicLeap
                    HeadPoseTr = np.matmul(MLRSTr,RSTr_LH)
                    #Head Pose matrix in the form of array to be sent
                    ArrToSend = np.array([HeadPoseTr[0][0],HeadPoseTr[0][1],HeadPoseTr[0][2],HeadPoseTr[0][3],HeadPoseTr[1][0],HeadPoseTr[1][1],HeadPoseTr[1][2],HeadPoseTr[1][3],HeadPoseTr[2][0],HeadPoseTr[2][1],HeadPoseTr[2][2],HeadPoseTr[2][3],HeadPoseTr[3][0],HeadPoseTr[3][1],HeadPoseTr[3][2],HeadPoseTr[3][3]])

                    print("Arr to send",ArrToSend)
                    ArrToSendPrev = ArrToSend
                    # pack the array to be sent before sending
                    dataTosend = struct.pack('f'*len(ArrToSend),*ArrToSend)
                    proc_frame_count += 1
                    print("time taken for 1 frame:", proc_frame_count, "th frame in", time.time() - start_time_frame, "seconds")
                    print("time taken for n frame:", proc_frame_count, "th frame in", time.time() - start_time, "seconds")

                    # print(dataTosend)
                    if socket_connect == 1 and img_idx % skip_frame_send == 0: #
                        img_sent = img_idx
                        c.send(dataTosend)

                    # if no nose location detected, send transformation matrix that was sent most recently
                elif img_idx > 1:
                    #print("sending prev pose detected")
                    dataTosendPrev = struct.pack('f'*len(ArrToSendPrev),*ArrToSendPrev)
                    if socket_connect == 1:
                        c.send(dataTosendPrev)

            else:

                input_img, euler_angles = draw_results_ssd(
                    detected, input_img, faces, ad, img_size, img_w, img_h, model, time_detection, time_network, time_plot)

            if len(detected) > len(detected_pre) or img_idx % (skip_frame_detect*3) == 0:
                detected_pre = detected

            key = cv2.waitKey(1)
        except KeyboardInterrupt:
            print(img_idx, time.time() - start_time)
            print("time taken to process {0:d} frames was {1:2f} seconds".format(img_idx, time.time() - start_time))
            break


if __name__ == '__main__':
    main()
