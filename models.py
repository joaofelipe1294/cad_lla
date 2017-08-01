import cv2

class Image:
	""" Classe usada como base para trabalhar com as imagens, 
	abstrai as informacoes relevantes de uma imagm"""

	def __init__(self, image_path):
		self.path = image_path
		self.label = image_path[-5]
		self.image = cv2.imread(image_path)
		self.name = image_path[-11:]