from turtle import right
import cv2
from PIL import Image
import os
from math import sqrt
import numpy as np
import mediapipe as mp


def generatePP(imgFile, left, upper, maxiFile = "maxi.jpg"):
	imgSource = Image.open(imgFile)
	maxi = Image.open(maxiFile)

	# in Inches
	# sqrt((6 ** 2) + (4 ** 2)) Diagonal of a maxi in inch
	DIAGONAL_INCHES = sqrt(52)

	# Maxi dimensions
	# depends upon the display dimension in inches
	maxi_width, maxi_height = maxi.size
	maxi_diagonal = sqrt((maxi_width)**2 + (maxi_height)**2)
	Maxi_PPI = maxi_diagonal / DIAGONAL_INCHES
	maxi_crop_width = 1.5 * Maxi_PPI
	maxi_crop_height = 2.00 * Maxi_PPI

	

	# Camera/Video or Image's dimention
	source_width, source_height = imgSource.size
	source_diagonal = sqrt((source_width)**2 + (source_height)**2)
	source_PPI = source_diagonal / DIAGONAL_INCHES
	source_crop_width = 1.96 * source_PPI
	source_crop_height = 2.60 * source_PPI


	# left = int((source_width // 2) - (source_crop_width / 2))
	# upper = int((source_height // 2) - (source_crop_height / 2))
	# right = int((maxi_width // 2) + (maxi_crop_width / 2))
	# lower = int((maxi_height // 2) + (maxi_crop_height / 2))
	right = left + source_crop_width
	lower = upper + source_crop_height


	coord = left, upper, right, lower

	# print(left, upper, right, lower)
	
	img_crop = imgSource.crop(coord)

	img_crop.save("crop_img_{}.jpg".format(imgFile))

	img_crop = img_crop.resize((int(maxi_crop_width - Maxi_PPI*0.1),
                             int(maxi_crop_height - Maxi_PPI*0.1)), resample=None, box=None, reducing_gap=None)

	img_crop.save("crop_img_{}.jpg".format(imgFile))

	# For Border
	crop_img = cv2.imread("crop_img_{}.jpg".format(imgFile))
	border  = cv2.copyMakeBorder(crop_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value = [0, 0, 0])
	cv2.imwrite("border_crop_img_{}.jpg".format(imgFile), border)



	img_crop = Image.open("border_crop_img_{}.jpg".format(imgFile))

	resized_width, resized_height = img_crop.size

	Xgap = (maxi_width - (resized_width * 4)) / 5
	Ygap = (maxi_height - (resized_height * 2)) / 3

	x = Xgap
	y = Ygap
	i = 1
	while i < 5:
		maxi.paste(img_crop, (int(x), int(y)))
		x += (Xgap+resized_width)
		i += 1

	x = Xgap
	y = resized_height+(Ygap*2)
	i = 1
	while i < 5:
		maxi.paste(img_crop, (int(x), int(y)))
		x += (Xgap+resized_width)
		i += 1

	os.remove("border_crop_img_{}.jpg".format(imgFile))
	os.remove("crop_img_{}.jpg".format(imgFile))
	os.remove(imgFile)
	maxi.save("{}_pp.jpg".format(imgFile))


def RamoveBackground(frameImg, image_index = 0):
	mp_selfie_segmentation = mp.solutions.selfie_segmentation
	selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
		model_selection=1)

	image_path = 'BackGroundImages'
	images = os.listdir(image_path)
	bg_image = cv2.imread(image_path+'/'+images[image_index])
	results = selfie_segmentation.process(frameImg)

	mask = results.segmentation_mask
	height, width, channel = frameImg.shape

	condition = np.stack((results.segmentation_mask, ) * 3, axis=-1) > 0.5
	bg_image = cv2.resize(bg_image, (width, height))
	output_image = np.where(condition, frameImg, bg_image)

	k = cv2.waitKey(1)

	return output_image
