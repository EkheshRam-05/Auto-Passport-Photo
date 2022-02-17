import cv2
import mediapipe as mp
import imageEdit


cap = cv2.VideoCapture(0)


# pTime=0
img_counter = 0
score = 0


mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)


BGremove = input("Remove BackGround ??")


while True:
	SUCCESS, img = cap.read()
	# img = cv2.flip(img, 1)
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	frame = faceDetection.process(imgRGB)
	# print(results)

	if BGremove:
		img = imageEdit.RamoveBackground(img, "white")
		
	
	if frame.detections:
		for id, detection in enumerate(frame.detections):
			score = detection.score[0]
	print(score)
	k = cv2.waitKey(1)
	cv2.imshow("Frame Face Detection", img)
	if k % 256 == 27:
	    # ESC pressed
		print("Escape hit, closing...")
		break

	if score > 0.98:
		img_name = "TestImage_{}.jpg".format(img_counter)
		cv2.imwrite(img_name, img)
		print("{} written!".format(img_name))
		img_counter += 1
		imageEdit.generatePP(img_name)
		print('FaceDetection: 1')

