import cv2
from segmentation import kmeans
from models import Image
from segmentation import remove_red_cells
from segmentation import background_removal
from base import load
from segmentation import otsu
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt


path = 'bases/ALL_IDB1/'
images = load(path)
#images = [images[3]]



for image in images:
	lab_image = cv2.cvtColor(image.image, cv2.COLOR_BGR2LAB)
	hsv_image = cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV)
	yuv_image = cv2.cvtColor(image.image, cv2.COLOR_BGR2YUV)
	gray_image = cv2.cvtColor(image.image, cv2.COLOR_BGR2GRAY)
	blue, green, red = cv2.split(image.image)
	hue, saturation, value = cv2.split(hsv_image)
	l, a, b = cv2.split(lab_image)
	y, u, v = cv2.split(yuv_image)
	'''
	#cv2.imwrite('temp/temp/blue_' + image.name, blue)
	cv2.imwrite('temp/temp/green_' + image.name, green)
	#cv2.imwrite('temp/temp/red_' + image.name, red)
	cv2.imwrite('temp/temp/hue_' + image.name, hue)
	cv2.imwrite('temp/temp/saturation_' + image.name, saturation)
	#cv2.imwrite('temp/temp/value_' + image.name, value)
	cv2.imwrite('temp/temp/l_' + image.name, l)
	cv2.imwrite('temp/temp/a_' + image.name, a)
	cv2.imwrite('temp/temp/b_' + image.name, b)
	cv2.imwrite('temp/temp/y_' + image.name, y)
	cv2.imwrite('temp/temp/u_' + image.name, u)
	#cv2.imwrite('temp/temp/v_' + image.name, v)
	cv2.imwrite('temp/temp/gray_' + image.name, gray_image)
	#cv2.imwrite('temp/temp/' + image.name, segmented)
	
	'''

	'''
	hue_binary = otsu(hue)
	opening = cv2.morphologyEx(hue_binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
	erosion = cv2.erode(hue_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 1)

	norm_4 = kmeans(u, K = 4)
	binary_4 = otsu(norm_4)
	
	cv2.imwrite('temp/temp/u_' + image.name, u)
	cv2.imwrite('temp/temp/norm_4_' + image.name, norm_4)
	cv2.imwrite('temp/temp/otsu_4_' + image.name, binary_4)
	cv2.imwrite('temp/temp/hue_binary_' + image.name, hue_binary)
	cv2.imwrite('temp/temp/opening_' + image.name, opening)
	cv2.imwrite('temp/temp/erosion_' + image.name, erosion)
	'''

	added_image = cv2.add(b, u)
	added_binary = added_image.copy()
	added_binary[added_image < 255] = 0
	closing = cv2.morphologyEx(added_binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
	opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
	opening = cv2.bitwise_not(opening)
	im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours_image = np.zeros(image.image.shape[:2], np.uint8)
	contours_image[contours_image == 0] = 255
	cv2.drawContours(contours_image, contours, -1, 0, 1)
	for contour in contours:
		area = cv2.contourArea(contour)
		if 6000 < area and area < 1000000:
			M = cv2.moments(contour)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			center = (cX, cY)
			#contours_image.itemset(cY, cX, 0)
			temp = contours_image.copy()
			mask = np.zeros((image.image.shape[0] + 2 , image.image.shape[1] + 2) , np.uint8)
			before = np.count_nonzero(temp == 0)
			cv2.floodFill(temp , mask , center , 0)
			after = np.count_nonzero(temp == 0)
			#print(before)
			#print(after)
			#plt.imshow(temp, cmap = 'gray', interpolation = 'bicubic')
			#plt.xticks([]), plt.yticks([])
			#plt.show()
			if (after - before) > 1000000:
				pass
			else:
				contours_image = temp
			#cv2.circle(contours_image, center,int(radius), 127 ,3)
	
	#cv2.imwrite('temp/temp/contours_' + image.name, contours_image)
	mask = cv2.morphologyEx(contours_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
	#cv2.imwrite('temp/temp/mask_' + image.name, mask)
	mask = cv2.bitwise_not(mask)
	mask[mask == 255] = 1
	segmented = image.image * cv2.merge((mask, mask, mask))
			
	#cv2.imwrite('temp/temp/b_' + image.name, b)
	#cv2.imwrite('temp/temp/binary_' + image.name, added_binary)
	#cv2.imwrite('temp/temp/sum_' + image.name, added_image)
	#cv2.imwrite('temp/temp/opening_' + image.name, opening)
	#cv2.imwrite('temp/temp/closing_' + image.name, closing)
	#cv2.imwrite('temp/temp/contours_' + image.name, contours_image)
	#cv2.imwrite('temp/temp/mask_' + image.name, mask)
	cv2.imwrite('temp/temp/' + image.name, segmented)


	print(image.name)
