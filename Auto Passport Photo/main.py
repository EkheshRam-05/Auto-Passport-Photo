import cv2
import mediapipe as mp
from numpy import source
import imageEdit
import sys
from math import sqrt


def Boundary(img, bbox, l=20, t=5, rt=1):
	x, y, w, h = bbox
	x1, y1 = x+w, y+h
	cv2.rectangle(img, bbox, (255, 0, 255), rt)
	#Top Left
	cv2.line(img, (x, y), (x+l, y), (255, 0, 0), t)
	cv2.line(img, (x, y), (x, y+l), (255, 0, 0), t)
	#Top Right
	cv2.line(img, (x1, y), (x1-l, y), (255, 0, 0), t)
	cv2.line(img, (x1, y), (x1, y+l), (255, 0, 0), t)
	#Bottom Left
	cv2.line(img, (x, y1), (x+l, y1), (255, 0, 0), t)
	cv2.line(img, (x, y1), (x, y1-l), (255, 0, 0), t)
	#Bottom Right
	cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 0), t)
	cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 0), t)
	return img


def main(bgColor):
	yes = 0
	cap = cv2.VideoCapture(0)


	# pTime=0
	img_counter = 0
	score = 0


	mpFaceDetection = mp.solutions.face_detection
	faceDetection = mpFaceDetection.FaceDetection(0.75)


	BGremove = input("\n\nRemove BackGround ??(Yes: 1 No: 0)\n\n")
	print()


	while True:

		SUCCESS, img = cap.read()
		# img = cv2.flip(img, 1)
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		frame = faceDetection.process(imgRGB)
		# print(results)

		if int(BGremove):
			img = imageEdit.RamoveBackground(img, int(bgColor))

		if yes:
			img_name = "TestImage_{}.jpg".format(img_counter)
			cv2.imwrite(img_name, img)
			print("{} written!".format(img_name))
			img_counter += 1
			imageEdit.generatePP(img_name)

		yes = 0
		if frame.detections:
			for id, detection in enumerate(frame.detections):
				score = detection.score[0]
				bboxC = detection.location_data.relative_bounding_box
				ih, iw, ic = img.shape
				source_diagonal = sqrt((iw)**2 + (ih)**2)
				source_PPI = source_diagonal / sqrt(52)
				source_crop_width = 1.5 * source_PPI
				source_crop_height = 2.0 * source_PPI
				print(ih, iw, ic)
				left = int((iw // 2) - (source_crop_width / 2))
				upper = int((ih // 2) - (source_crop_height / 2))
				bbox = int(left), int(upper), \
					int(source_crop_width), int(source_crop_height)
				Boundary(img, bbox)


		print(score)
		k = cv2.waitKey(1)
		cv2.imshow("Frame Face Detection", img)
		if k % 256 == 27:
			# ESC pressed
			print("Escape hit, closing...")
			break		

		if score > 0.90:
			yes = 1




if __name__ == '__main__':
	main(sys.argv[1])
