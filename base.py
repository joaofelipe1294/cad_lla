import os
import sys
import cv2
import numpy as np
from models import Image


#############################################################################################
#                                          LOAD                                             # 
#############################################################################################
# funcao que carrega as imagens de uma base (pasta)											#
# @recives																					#
# 	- uma string com o caminho da pasta que contem as imagens 							    #
#    																						#
# @returns                                                                                  #
# 	- tres listas, a primeira contem as imagens carregadas, objetos np.array, a segunda     #
# 	  contem inteiros 1 ou 0 referentes a classe da imagem,  a terceira contem strings      #
#     com o nome dos arquivos referentes as imagens                                         #
#############################################################################################

def load(base_path):
	images = []
	print('Carregando imagens de ' + base_path + ' ...')
	paths = os.listdir(base_path)
	try:
		paths.remove('.DS_Store')
	except Exception as e:
		pass
	paths.sort()
	for path in paths:
		image = Image(base_path + path)
		images.append(image)
	print('Imagens carregadas')
	return images
