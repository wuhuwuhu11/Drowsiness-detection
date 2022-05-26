# Import the necessary packages 
import datetime as dt
from EAR_calculator import *
from imutils import face_utils 
from imutils.video import VideoStream
import matplotlib.pyplot as plt
from matplotlib import style 
import imutils 
import dlib
import time 
import argparse 
import cv2 
from playsound import playsound
import os
import pandas as pd

style.use('fivethirtyeight')
# Creating the dataset 
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


#all eye  and mouth aspect ratio with time
ear_list=[]
total_ear=[]
mar_list=[]
total_mar=[]
ts=[]
total_ts=[]
# Construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser() 
ap.add_argument("-p", "--shape_predictor", required = True, help = "path to dlib's facial landmark predictor")
ap.add_argument("-r", "--picamera", type = int, default = -1, help = "whether raspberry pi camera shall be used or not")
args = vars(ap.parse_args())

# Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink 
EAR_THRESHOLD = 0.25
# Declare another costant to hold the consecutive number of frames to consider for a blink 
CONSECUTIVE_FRAMES = 22
CONSECUTIVE_FRAMES_m =15
# Another constant which will work as a threshold for MAR value
MAR_THRESHOLD = 14

# Initialize two counters 
BLINK_COUNT = 0 
FRAME_COUNT = 0
count_yawn = 0

# Now, intialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
print("[INFO]Loading the predictor.....")
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(args["shape_predictor"])

# Grab the indexes of the facial landamarks for the left and right eye respectively 
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Now start the video stream and allow the camera to warm-up
print("[INFO]Loading Camera.....")
vs = VideoStream(usePiCamera = args["picamera"] > 0).start()
time.sleep(2) 

assure_path_exists("dataset/")
count_sleep = 0

count_sleep2 = 0


 
# Now, loop over all the frames and detect the faces现在，对所有的帧进行循环并检测面部
while True: 
	# Extract a frame 提取一个框架
	frame = vs.read()
	cv2.putText(frame, "PRESS 'ESC' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) #图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
	# Resize the frame 调整框架大小
	frame = imutils.resize(frame, width = 500)
	# Convert the frame to grayscale   将框架转换为灰度
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Detect faces 检测人脸
	rects = detector(frame, 1)

	# Now loop over all the face detections and apply the predictor 现在遍历所有面部检测并应用预测器
	for (i, rect) in enumerate(rects): 
		shape = predictor(gray, rect)
		# Convert it to a (68, 2) size numpy array 将其转换为（68，2）大小的numpy数组
		shape = face_utils.shape_to_np(shape)

		# Draw a rectangle over the detected face 在检测到的脸上绘制一个矩形
		(x, y, w, h) = face_utils.rect_to_bb(rect) 
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)	
		# Put a number 
		cv2.putText(frame, "Student", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		leftEye = shape[lstart:lend]
		rightEye = shape[rstart:rend] 
		mouth = shape[mstart:mend]
		# Compute the EAR for both the eyes 计算两只眼睛的EAR
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# Take the average of both the EAR   取两个EAR的平均值
		EAR = (leftEAR + rightEAR) / 2.0
		#live datawrite in csv 实时数据写入CSV
		ear_list.append(EAR)
		#print(ear_list)  打印（ear_list）
		

		ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
		# Compute the convex hull for both the eyes and then visualize it 计算双眼的凸包，然后对其进行可视化
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		# Draw the contours  画轮廓
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

		MAR = mouth_aspect_ratio(mouth)
		mar_list.append(MAR/10)
		# Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 添加帧到数据集ar瞌睡驾驶证明
		# Thus, count the number of frames for which the eye remains closed 因此，计算眼睛保持闭着的帧数
		if EAR < EAR_THRESHOLD: 
			FRAME_COUNT += 1

			cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

			if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
				count_sleep += 1
				# Add the frame to the dataset ar a proof of drowsy driving 将帧添加到数据集或昏昏欲睡的证明
				cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
				playsound('sound files/alarm.mp3')
				cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else: 
			if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
				playsound('sound files/warning.mp3')
			FRAME_COUNT = 0
		#cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# Check if the person is yawning检查该人是否在打哈欠
		if MAR > MAR_THRESHOLD:
			count_yawn += 1
			cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1)
			if count_yawn >= CONSECUTIVE_FRAMES_m:
				count_sleep2 += 1
				# Add the frame to the dataset ar a proof of drowsy driving 将帧添加到数据集或昏昏欲睡的证明
				cv2.imwrite("dataset/frame_yawn%d.jpg" % count_sleep2, frame)
				playsound('sound files/alarm.mp3')
				cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			if count_yawn >= CONSECUTIVE_FRAMES_m:
				playsound('sound files/warning.mp3')
			count_yawn = 0

			# cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# # Add the frame to the dataset ar a proof of drowsy driving 将帧添加到数据集或昏昏欲睡的证明
			# cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)
			# playsound('sound files/alarm.mp3')
			# playsound('sound files/qifei.mp3')
	#total data collection for plotting 绘制总数据
	for i in ear_list:
		total_ear.append(i)
	for i in mar_list:
		total_mar.append(i)			
	for i in ts:
		total_ts.append(i)
	#display the frame  显示框架
	cv2.imshow("Output", frame)
	key = cv2.waitKey(1) & 0xFF 
	
	

	if key == ord('q'):
		break

a = total_ear
b=total_mar
c = total_ts

df = pd.DataFrame({"EAR" : a, "MAR":b,"TIME" : c})
df.to_csv("op_webcam.csv", index=False)
df=pd.read_csv("op_webcam.csv")

df.plot(x='TIME',y=['EAR','MAR'])
#plt.xticks(rotation=45, ha='right')

plt.subplots_adjust(bottom=0.30)
plt.title('EAR & MAR calculation over time of webcam')
plt.ylabel('EAR & MAR')
plt.gca().axes.get_xaxis().set_visible(False)
plt.show()
cv2.destroyAllWindows()
vs.stop()
