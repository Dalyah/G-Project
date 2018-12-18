#!/usr/bin/ python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#This code visualize the yaw,roll, pitch angle results obtained from deepgaze CnnHeadPoseEstimator
# by drawing lines with different colores on the  images

import os
import tensorflow as tf
import cv2
import numpy as np
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
from deepgaze.face_detection import HaarFaceDetector

hfd = HaarFaceDetector("../etc/xml/haarcascade_frontalface_alt.xml", "../etc/xml/haarcascade_profileface.xml")
sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object

# Load the weights from the configuration folders
my_head_pose_estimator.load_roll_variables(os.path.realpath("../etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_pitch_variables(os.path.realpath("../etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_yaw_variables(os.path.realpath("../etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))

def yaw2rotmat(yaw):
    x = 0.0
    y = 0.0
    z = yaw
    ch = np.cos(z)
    sh = np.sin(z)
    ca = np.cos(y)
    sa = np.sin(y)
    cb = np.cos(x)
    sb = np.sin(x)
    rot = np.zeros((3,3), 'float32')
    rot[0][0] = ch * ca
    rot[0][1] = sh*sb - ch*sa*cb
    rot[0][2] = ch*sa*sb + sh*cb
    rot[1][0] = sa
    rot[1][1] = ca * cb
    rot[1][2] = -ca * sb
    rot[2][0] = -sh * ca
    rot[2][1] = sh*sa*cb + ch*sb
    rot[2][2] = -sh*sa*sb + ch*cb
    return rot

def roll2rotmat(roll):
    x = roll
    y = 0.0
    z = 0.0
    ch = np.cos(z)
    sh = np.sin(z)
    ca = np.cos(y)
    sa = np.sin(y)
    cb = np.cos(x)
    sb = np.sin(x)
    rot = np.zeros((3,3), 'float32')
    rot[0][0] = ch * ca
    rot[0][1] = sh*sb - ch*sa*cb
    rot[0][2] = ch*sa*sb + sh*cb
    rot[1][0] = sa
    rot[1][1] = ca * cb
    rot[1][2] = -ca * sb
    rot[2][0] = -sh * ca
    rot[2][1] = sh*sa*cb + ch*sb
    rot[2][2] = -sh*sa*sb + ch*cb
    return rot

def pitch2rotmat(pitch):
    x = 0.0
    y = pitch
    z = 0.0
    ch = np.cos(z)
    sh = np.sin(z)
    ca = np.cos(y)
    sa = np.sin(y)
    cb = np.cos(x)
    sb = np.sin(x)
    rot = np.zeros((3,3), 'float32')
    rot[0][0] = ch * ca
    rot[0][1] = sh*sb - ch*sa*cb
    rot[0][2] = ch*sa*sb + sh*cb
    rot[1][0] = sa
    rot[1][1] = ca * cb
    rot[1][2] = -ca * sb
    rot[2][0] = -sh * ca
    rot[2][1] = sh*sa*cb + ch*sb
    rot[2][2] = -sh*sa*sb + ch*cb
    return rot

for i in range (16,18):
    file_name = str(i) + ".jpg"
    file_save = str(i) + "_axes.jpg"
    print("Processing image ...." + file_name)
    image = cv2.imread(file_name) #Read the image with OpenCV
    img_gray = cv2.imread(file_name, 0)
    #cv2.imshow("gray", img_gray)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    print img_gray
    # STEP 1
    # detect the face if needed
    allTheFaces = hfd.returnMultipleFacesPosition(img_gray, runFrontal=True, runFrontalRotated=True,
                        runLeft=True, runRight=True,
                        frontalScaleFactor=1.2, rotatedFrontalScaleFactor=1.2,
                        leftScaleFactor=1.15, rightScaleFactor=1.15,
                        minSizeX=64, minSizeY=64,
                        rotationAngleCCW=30, rotationAngleCW=-30)
    print "all faces"
    print allTheFaces
    if allTheFaces: # Crop the image
        # Iterating all the faces
        for element in allTheFaces:
            face_x1 = int(element[0])
            face_y1 = int(element[1])
            face_x2 = int(face_x1+element[2])
            face_y2 = int(face_y1+element[3])
            #cv2.rectangle(image, (face_x1, face_y1), (face_x2, face_y2), [255, 0, 0])
        # Drawing a rectangle around the face
        #cv2.rectangle(image, (face_x1, face_y1), (face_x2, face_y2), [0,0,255])
        # crop the image
        image = image[face_y1:face_y2, face_x1:face_x2]
        cv2.imshow("Cropped Image", image)
        cv2.waitKey(0)

    # Get the angles for roll, pitch and yaw
    h, w = image.shape[:2]
    print w
    print h
    roll = my_head_pose_estimator.return_roll(image, radians=True)  # Evaluate the roll angle using a CNN
    pitch = my_head_pose_estimator.return_pitch(image, radians=True)  # Evaluate the pitch angle using a CNN
    yaw = my_head_pose_estimator.return_yaw(image, radians=True)  # Evaluate the yaw angle using a CNN
    print("Estimated [roll, pitch, yaw] ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")
    print("")

    # visualization YAW Angle
    # getting camera params
    cam_w = image.shape[1]
    cam_h = image.shape[0]
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60/2 * np.pi / 180)
    f_y = f_x

    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])

    print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")

    axis_y = np.float32([[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.5]])
    axis_r = np.float32([[0.5, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])
    axis_p = np.float32([[0.0, 0.0, 0.0],
                             [0.0, 0.5, 0.0],
                             [0.0, 0.0, 0.0]])

    tvec_y = np.array([0.0, 0.0, 1.0], np.float) # translation vector
    tvec_r = np.array([1.0, 0.0, 0.0], np.float) # translation vector
    tvec_p = np.array([0.0, 1.0, 0.0], np.float) # translation vector
    rot_matrix_y = yaw2rotmat(-yaw[0,0,0])
    rot_matrix_r = roll2rotmat(roll[0,0,0])
    rot_matrix_p = pitch2rotmat(-pitch[0,0,0])
    rvec_y, jacobian_y = cv2.Rodrigues(rot_matrix_y)
    rvec_r, jacobian_r = cv2.Rodrigues(rot_matrix_r)
    rvec_p, jacobian_p = cv2.Rodrigues(rot_matrix_p)
    imgpts_y, jac_y = cv2.projectPoints(axis_y, rvec_y, tvec_y, camera_matrix, None)
    imgpts_r, jac_r = cv2.projectPoints(axis_r, rvec_r, tvec_r, camera_matrix, None)
    imgpts_p, jac_p = cv2.projectPoints(axis_p, rvec_p, tvec_p, camera_matrix, None)

    p_start_y = (int(c_x), int(c_y))
    p_start_r = (int(c_x), int(c_y))
    #Yaw Point
    p_stop_y = (int(imgpts_y[2][0][0]), int(imgpts_y[2][0][1]))
    print imgpts_y
    print p_stop_y
    #pitch Point
    p_stop_p = (int(imgpts_p[1][0][0]), int(imgpts_p[1][0][1]))
    #roll Point
    p_stop_r = (int(imgpts_r[0][0][0]) , int(imgpts_r[0][0][1]))
    print imgpts_r
    print p_stop_r
    #print imgpts

    #yaw line
    cv2.line(image, p_start_y, p_stop_y, (0,0,255), 3) #Red
    # pitch line
    cv2.line(image, p_start_y, p_stop_p, (0,255,0), 3) #Green
    #roll line
    cv2.line(image, p_start_r, p_stop_r, (255,0,0), 3) # blue
    cv2.circle(image, p_start_y, 1, (0,255,0), 3) #Green
    cv2.imshow("Image withe axes", image)
    cv2.imwrite(file_save, image)
