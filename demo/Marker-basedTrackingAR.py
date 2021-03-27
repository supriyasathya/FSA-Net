# This script can be used for the following:
# 1. aruco marker detection with Realsense camera (set socket_connect = 0 to test Realsense tracking Aruco pose only)
# 2. calibrate the realsense and AR headset spaces using marker 1, and test the calibration by tracking an Aruco marker (set socket_connet = 1). This can be used to
# perform marker-based head tracking by attaching an Aruco marker to subject's head and tracking. Details in the Google doc.

import numpy as np
import cv2
import cv2.aruco as aruco
from arucocalibclass import arucocalibclass
import os
import pyrealsense2 as rs
import matplotlib.pyplot as plt 
import socket
import struct
from scipy.spatial.transform import Rotation as R
from charucocalibclass import charucocalibclass

# function that is called whenever either an aruco marker or charuco board needs to be tracked by Realsense and the transformed pose is streamed to MagicLeap
def startTracking(marker_type, MLRSTr, n_sq_y, n_sq_x, squareLen, markerLen, aruco_dict, socket_connect, c):
    RSTr = None
    #if Charuco board is the marker type, execute following if statement
    if marker_type == 1:
        rvec = np.zeros((1, 3))  
        tvec = np.zeros((1, 3))  
        objectPoints = np.zeros((1,3))
    #configure Realsense camera stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    #get realsense stream intrinsics
    intr = profile.get_stream(
        rs.stream.color).as_video_stream_profile().get_intrinsics()  # profile.as_video_stream_profile().get_intrinsics()
    mtx = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    dist = np.array(intr.coeffs)

    while (True):
        #start streaming
        frames = pipeline.wait_for_frames()
        #get color frame
        color_frame = frames.get_color_frame()
        input_img = np.asanyarray(color_frame.get_data())
        # operations on the frame - convert to grayscale
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        # detector parameters can be set here.
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # font for displaying text (below)
        font = cv2.FONT_HERSHEY_SIMPLEX
      
        if marker_type == 1:
            # check if the ids list is not empty
            if (ids is not None):
                # create Charudo board
                charuco_board = aruco.CharucoBoard_create( n_sq_x, n_sq_y, squareLen, markerLen, aruco_dict)
                ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
                # if there are enough corners to get a reasonable result
                if (ret > 3):
                    aruco.drawDetectedCornersCharuco(input_img, ch_corners, ch_ids, (0, 0, 255))
                    #estimate Charuco board pose
                    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, charuco_board, mtx,
                                                                            dist, None,
                                                                            None)  # , useExtrinsicGuess=False)

                    print("pose", retval, rvec, tvec)
                    # if a pose could be estimated
                    if( retval ) :
                       # draw axis showing pose of board
                       rvec = np.reshape(rvec, (1, 3))
                       tvec = np.reshape(tvec, (3, 1))
                       aruco.drawAxis(input_img, mtx, dist, rvec, tvec, 0.1)
                       #convert to rotation matrix from rotation vector
                       R_rvec = R.from_rotvec(rvec)
                       R_rotmat = R_rvec.as_matrix()
                       # get transformation matrix combining tvec and rotation matrix
                       RSTr = np.hstack([R_rotmat.reshape((3,3)), tvec])
                       RSTr = np.vstack([RSTr, [0, 0, 0, 1]])
                       print("RSTr", RSTr)
            
            else:
                # show 'No Ids' when no markers are found
                cv2.putText(input_img, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
        # following part is executed if single Aruco marker is selected
        else:

            if np.all(ids != None):
                # estimate pose of each marker and return the values
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLen, mtx, dist)
                # (rvec-tvec).any() # get rid of that nasty numpy value array error
                for i in range(0, ids.size):
                    # draw axis for the aruco markers
                    aruco.drawAxis(input_img, mtx, dist, rvec[i], tvec[i], 0.1)
                    # draw a square around the markers
                aruco.drawDetectedMarkers(input_img, corners)
                #select only the first marker in the list (assumes single Aruco marker is in the scene)
                #convert to rotation matrix from rotation vector
                R_rvec = R.from_rotvec(rvec[0])
                R_rotmat = R_rvec.as_matrix()
                # get transformation matrix combining tvec and rotation matrix
                RSTr = np.hstack([R_rotmat[0], tvec[0].transpose()])
                RSTr = np.vstack([RSTr, [0, 0, 0, 1]])


        if socket_connect == 1 and RSTr is not None:
            # transform the pose from OpenCV's right-handed rule to Unity's left-handed rule
            RSTr_LH = np.array(
                [RSTr[0][0], RSTr[0][2], RSTr[0][1], RSTr[0][3], RSTr[2][0], RSTr[2][2], RSTr[2][1], RSTr[2][3],
                 RSTr[1][0], RSTr[1][2], RSTr[1][1], RSTr[1][3], RSTr[3][0], RSTr[3][1], RSTr[3][2],
                 RSTr[3][3]])  # converting to left handed coordinate system
            RSTr_LH = RSTr_LH.reshape(4, 4)
            # compute transformed pose to send to MagicLeap
            PoseTr = np.matmul(MLRSTr, RSTr_LH)
            # Head Pose matrix in the form of array to be sent
            ArrToSend = np.array(
                [PoseTr[0][0], PoseTr[0][1], PoseTr[0][2], PoseTr[0][3], PoseTr[1][0],
                 PoseTr[1][1], PoseTr[1][2], PoseTr[1][3], PoseTr[2][0], PoseTr[2][1],
                 PoseTr[2][2], PoseTr[2][3], PoseTr[3][0], PoseTr[3][1], PoseTr[3][2],
                 PoseTr[3][3]])

            # pack the array to be sent before sending
            dataTosend = struct.pack('f' * len(ArrToSend), *ArrToSend)
            
            if socket_connect == 1:  # and img_idx % skip_frame_send == 0:  #
                # img_sent = img_idx
                c.send(dataTosend)

        # display the resulting frame
        cv2.imshow('frame', input_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    #set flag whether we want to detect 1 - charuco board or else - single aruco marker
    marker_type = 3
    # set number of samples for computing transformation matrix
    n_samples = 10
    # create a socket object if socket_connect = 1
    # flag set to 1 if testing with AR Headset, else set to 0 if testing only with Realsense 
    socket_connect = 1
    kalman_flag = 0
    if marker_type == 1: #charuco board
        squareLen = 0.03710 # make changes to square length and marker length, number of squares based on the marker you are using for calibration
        markerLen = 0.02250
        n_sq_y = 7
        n_sq_x = 5
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    else: # single aruco
        squareLen = None
        markerLen = 0.0645
        n_sq_y = None
        n_sq_x = None
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    
    c = 0
    # if socket_connect = 1, call the aruco calibration instance and claibrate with MagicLeap
    if socket_connect == 1:
        s = socket.socket()
        print("Socket successfully created")
        # reserve a port on your computer - for example 2020 - but it can be anything
        port = 2020
        s.bind(('', port))
        print("socket binded to %s" % (port))

        # put the socket into listening mode
        s.listen(5)
        print("socket is listening")
        c, addr = s.accept()
        print('got connection from ', addr)

        if marker_type == 1:
           charucoinstance = charucocalibclass()
           ReturnFlag = 1
           #MLRSTr is the transformation matrix computed after calibration
           MLRSTr = charucoinstance.startcamerastreaming(c, ReturnFlag,  n_sq_y, n_sq_x, squareLen, markerLen, aruco_dict, n_samples)
           print(MLRSTr)

        else:
            arucoinstance = arucocalibclass()
            ReturnFlag = 1
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
            #MLRSTr is the transformation matrix computed after calibration
            MLRSTr = arucoinstance.startcamerastreaming(c, ReturnFlag, markerLen, aruco_dict, n_samples)
            print(MLRSTr)
    else:
        MLRSTr = np.array((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))
        MLRSTr = MLRSTr.reshape(4, 4)
        print(MLRSTr)
    print(socket_connect, c)
    #Once calibration is done, start tracking - remember to set the marker length, square length and aruco dict based on the marker we use for tracking
    # This step can be used to verify the calibration.
    marker_type = 3
    squareLen = None 
    markerLen = 0.0645 
    n_sq_y = None 
    n_sq_x = None 
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250) 
    startTracking(marker_type, MLRSTr, n_sq_y, n_sq_x, squareLen, markerLen, aruco_dict, socket_connect, c)


if __name__ == '__main__':
    main()


