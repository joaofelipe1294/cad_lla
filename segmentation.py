import numpy as np
import cv2

#############################################################################################
#                                         KMEANS                                            # 
#############################################################################################
# funcao que altera a quantizalção de uma imagem RGB         								#
# @recives																					#
# 	- image --> um objeto np.array com a imagem 									        #
#   - K     --> numero de classes que serao divididas as imagens 						    #
# 																							#
# @returns                                                                                  #
# 	- objeto np.array com a quantização reaorganizada                                		#
#############################################################################################

def kmeans(image , K = 4):
	if len(image.shape) == 3:	
		Z = image.reshape((-1,3)) #converte imagem em um vetor
	else:
		Z = image.reshape((-1)) #converte imagem em um vetor
	#Z = rgb_image.reshape((-1,3)) #converte imagem em um vetor
	Z = np.float32(Z) #converte o vetor para tipo float32 
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #define o criterio de parada para o kmeans
	ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS) #aplica o algoritmo do kmeans
	center = np.uint8(center) #converte novamente o vetor para np.uint8
	res = center[label.flatten()] 
	res2 = res.reshape((image.shape)) #converte a imagem novamente para seu formato original
	return res2

#############################################################################################
#                                           OTSU                                            # 
#############################################################################################
# funcao que aplica a tecnica de Otsu para binarizar uma imagem em tons de cinza			#
# @recives																					#
# 	- gray_image --> um objeto np.array com a imagem em tons de cinza					    #
# 																							#
# @returns                                                                                  #
# 	- imagem binaria resultado da aplicacao da tecnica de Otsu                       		#
#############################################################################################

def otsu(gray_image):
	blur_image = cv2.GaussianBlur(gray_image , (5,5) , 0)                                       #borra a imagem aplicando um filtro gaussiano , necessario para que o threshold OTSU funcione
	otsu_image = cv2.threshold(blur_image , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]   #aplica o threshold de otsu na imagem borrada
	return otsu_image

#############################################################################################
#                                     REMOVE_RED_CELLS                                      # 
#############################################################################################
# funcao que remove as hemacias de uma imagem RGB            								#
# @recives																					#
# 	- rgb_image --> um objeto np.array com a imagem  									    #
# 																							#
# @returns                                                                                  #
# 	- mascara composta de 0 e 1, onde as hemacias possuem valores iguais a 0 e os demais 1  #
#############################################################################################

def remove_red_cells(rgb_image):
	height, width = rgb_image.shape[:2]
	blue_chanel = cv2.split(rgb_image)[0]                       #obtem canal azul RGB
	hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)      #converte a imagem de RGB para HSV
	saturation = cv2.split(hsv_image)[1]           			    #obtem canal de saturacao
	saturation_binary = otsu(saturation)                        #aplica threshold na saturacao
	blue_chanel_binary = otsu(blue_chanel)                      #aplica threshold no canal B (azul)
	sum_image = blue_chanel_binary + saturation_binary          #soma os threshold da saturacao ao threshold do canal azul para remover a celula central da imagem , mantem apenas as hemacias em preto e o fundo branco
	closing = cv2.morphologyEx(sum_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))  #aplica fechamento para remover ruidos (alguns contornos da celila principal podem ter ficado na imagem resultante da som)
	seeds = np.argwhere(closing == 255)  #recupera os indices do valore diferentes de 0
	np.random.shuffle(seeds)             #embaralha a lista com os indices de pxs diferentes de 0
	index = 0                            #indice utilizado da interacao dos indices de pixels brancos
	copy_img = closing.copy()            #cria uma copia da imagem resultante do fechamento
	while True:
		mask = np.zeros((height + 2 , width + 2) , np.uint8) #mascara utilizada no filtro de crescimento de regioes
		cv2.floodFill(copy_img , mask , tuple([seeds[index][1], seeds[index][0]]) , 0 , loDiff = 2 , upDiff = 2)  #aplica a funcao cv2.floodFill() para cada semente criada
		if height * width - np.count_nonzero(copy_img) > height * width * 0.8: #verifica se o numero de pixels pretos presentes na imagem eh maior do que 80%, logo o px usado como semente esta no centro de uma das hemacias 
			break
		else:
			copy_img = closing.copy()           #caso a operacao de crescimento de regioes tenha dado errado eh recetada a imagem 
			index += 1                          #pega novas coordenadas para o ponto semente
	mask = cv2.bitwise_not(copy_img)            #inverte a imagem que teve sucesso no crescimento de regioes
	mask[mask == 255] = 1                       #prera uma mascara com valores 1 e 0
	centerless = closing * mask                 #remove o centro das hemacias a partir da multiplicacao entre a mascara e a imagem resultante do fechamento
	erosion = cv2.erode(centerless , cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) , iterations = 1) #aplica erosao para que a area preta cresca, dando uma margem de seguranca
	mask = cv2.threshold(erosion,10,1,cv2.THRESH_BINARY)[1]
	return mask

#############################################################################################
#                                      BACKGROUD_REMOVAL     (RECOMENTAR)                               # 
#############################################################################################
# funcao que remove o fundo presente na imagem, trabalha no canal de saturacao da imagem    #
# requantizada para apenas 4 classes                                                        #
# @recives																					#
# 	- rgb_image --> um objeto np.array com a imagem em tons de cinza					    #
# 																							#
# @returns                                                                                  #
# 	- retorna uma imagem usada como mascara para remocao do fundo, eh composta por 0 e 1    #
#     os pixels referentes ao fundo possuem valor igual a 0 e os demais igual a 1           #
#                                                                                           #
# ALGORITMO                                                                                 #
#     1-) alterar quantizacao da imagem original RGB utilizando algoritmo KMeans, sendo K=4 #
#     2-) obter o canal referente a saturacao                                               #
#     3-) obter o menor valor presente no canal de saturacao                                #
#     4-) criar a mascara utilizando os valores dois pixels da imagem de saturacao como     #
#         referencia, a mascara tera o valor 1 nos indices que tiverem seu valor maior do   #
#         que o menor valor presente na imagem de saturacao (obtido no passo anterior)      #
#############################################################################################


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def mask_builder(image):
	lba_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lba_image)
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	cl = clahe.apply(l)
	limg = cv2.merge((cl,a,b))
	final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	norm_image = kmeans(final)
	mask = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
	max_value = mask.max()
	mask[mask == max_value] = 0
	mask[mask > 0] = 255
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
	count = np.count_nonzero(mask == 0)
	return mask, count

def background_removal(image):
	masks = []
	counts = []
	for gamma_value in np.arange(1., 4.5, .5):
		ajusted = adjust_gamma(image, gamma_value)
		mask, count = mask_builder(ajusted)
		masks.append(mask)
		counts.append(count)
	mask = masks[counts.index(max(counts))]
	mask[mask == 255] = 1
	segmented = image * cv2.merge((mask, mask, mask))
	return segmented


	