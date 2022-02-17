import cv2
from PIL import Image
import os
from math import sqrt as s
import numpy as np
import mediapipe as mp


def generatePP(imgFile, maxiFile = "maxi.jpg"):
	imgSource = Image.open(imgFile)
	maxi = Image.open(maxiFile)

	# in Inches
	DISPLAY_SIZE_DIAGONAL = 14


	# Maxi dimensions
	# depends upon the display dimension in inches
	maxi_width, maxi_height = maxi.size
	maxi_pixels_height = maxi_height
	maxi_pixels_width = maxi_width
	maxi_pixels_diagonal = s((maxi_pixels_width)**2 + (maxi_pixels_height)**2)
	Maxi_PPI = maxi_pixels_diagonal / DISPLAY_SIZE_DIAGONAL
	maxi_crop_width = 1.96 * Maxi_PPI
	maxi_crop_height = 2.60 * Maxi_PPI


	# Camera/Video or Image's dimention
	source_width, source_height = imgSource.size
	source_pixels_height = source_height
	source_pixels_width = source_width
	source_pixels_diagonal = s((source_pixels_width)**2 + (source_pixels_height)**2)
	source_PPI = source_pixels_diagonal / DISPLAY_SIZE_DIAGONAL
	# source_crop_width = 1.96 * source_PPI
	# source_crop_height = 2.60 * source_PPI


	# img_crop = Image.resize((maxi_crop_width, maxi_crop_height), resample=None, box=None, reducing_gap=None)
	leftCord = int((source_width // 2) - (maxi_crop_width / 2))
	rightCord = int((source_width // 2) + (maxi_crop_width / 2))
	upCord = int((source_height // 2) - (maxi_crop_height / 2))
	downCord = int((source_height // 2) + (maxi_crop_height / 2))


	coord = leftCord, upCord, rightCord, downCord
	
	img_crop = imgSource.crop(coord)

	img_crop.save("crop_img_{}.jpg".format(imgFile))

	crop_img = cv2.imread("crop_img_{}.jpg".format(imgFile))

	border  = cv2.copyMakeBorder(crop_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value = [0, 0, 0])
	
	cv2.imwrite("border_crop_img_{}.jpg".format(imgFile), border)

	img_crop = Image.open("border_crop_img_{}.jpg".format(imgFile))

	resized_width, resized_height = img_crop.size

	Xgap = (maxi_pixels_width - (resized_width * 4)) / 5
	Ygap = (maxi_pixels_height - (resized_height * 2)) / 3

	x = Xgap
	y = Ygap
	i = 1
	while i < 5:
		maxi.paste(img_crop, (int(x), int(y)))
		x += (Xgap+resized_width)
		i += 1

	x = Xgap
	y = Ygap+(resized_height+Ygap)
	i = 1
	while i < 5:
		maxi.paste(img_crop, (int(x), int(y)))
		x += (Xgap+resized_width)
		i += 1

	os.remove("border_crop_img_{}.jpg".format(imgFile))
	os.remove("crop_img_{}.jpg".format(imgFile))
	os.remove(imgFile)
	maxi.save("./Passport Photo/{}_pp.jpg".format(imgFile))


def RamoveBackground(frameImg, color):
	image_index = 0
	mp_selfie_segmentation = mp.solutions.selfie_segmentation
	selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
		model_selection=1)

	image_path = 'BackGroundImages'
	images = os.listdir(image_path)
	image_index = 0
	if color == "white":
		image_index = 1
	bg_image = cv2.imread(image_path+'/'+images[image_index])
	results = selfie_segmentation.process(frameImg)

	mask = results.segmentation_mask
	height, width, channel = frameImg.shape

	condition = np.stack((results.segmentation_mask, ) * 3, axis=-1) > 0.5
	bg_image = cv2.resize(bg_image, (width, height))
	output_image = np.where(condition, frameImg, bg_image)

	k = cv2.waitKey(1)

	return output_image
