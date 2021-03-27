
Head tracking using an external camera and aligning the virtual rendering of head with its real-world counterpart while visualizing through an AR headset.

The AR headset used: MagicLeap One 

External camera used: Intel Realsense D415.

UnityProject-HeadTracking-0.24.0-9.3.6: Unity project to build the app on MagicLeap. Unity project was built and tested on: Unity version: 9.3.6, mlsdk 24.0 and Lumin OS 0.98.10

The FSA-Net was used for estimating head pose. Navigate to the /demo folder. Run this Python code to interface with Realsense and perform head tracking on MagicLeap.:

python3 demo_FSANET_SSDML_Tracking.py

Set socket_connect = 0 if you want to test FSANet head pose integrated with Realsense camera without streaming the pose to MagicLeap.

Run the following code to test Aruco marker detection using Realsense:

python3 Marker-basedTrackingAR.py 

The head pose estimation is performed based on SolvePnP. More details about SolvePnP can be found here: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

Details of the project in the Google docs.

